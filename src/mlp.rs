//! Small multi-layer perceptron (MLP) inference on the GPU.
//!
//! Each layer computes `Y = act(X * W + bias)`:
//! - `X`: batch x inputs, row-major
//! - `W`: inputs x outputs, row-major (so a row of X times W gives a row of Y)
//! - `bias`: outputs, added to every row
//!
//! Weights are uploaded once in [`Mlp::new`]. [`Mlp::forward`] then runs the
//! whole network in a single GPU submission: activations ping-pong between two
//! buffers, one dispatch per layer, and only the final layer is read back.
//!
//! Bindings use "dual bind groups" (dual descriptor sets), split by how often
//! they change:
//! - group 1, per layer: params, weights and bias, built once in `Mlp::new`
//! - group 0, activations: two prebuilt sets (ping->pong and pong->ping) that
//!   alternate layer by layer, cached with their buffers per batch size

use crate::matrix_multiplier::{
    pop_error_scopes, push_error_scopes, GpuRun, MatrixMultiplier, MAX_DIMENSION, WORKGROUP_SIZE,
};
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;
use wgpu::util::DeviceExt;

/// Matches `struct LayerParams` in `shader_dense.wgsl` / `shader_softmax.wgsl`
/// (group 1, binding 0). Has no batch size, so it never changes after `Mlp::new`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LayerParams {
    k: u32,
    n: u32,
    activation: u32,
    _pad: u32,
}

/// Matches `struct Batch` in the same shaders (group 0, binding 2), padded to
/// 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BatchParams {
    m: u32,
    _pad: [u32; 3],
}

type BoxError = Box<dyn std::error::Error>;

/// Activation applied after `X * W + bias`. Values match `params.activation`
/// in `shader_dense.wgsl`.
// All variants are API choices; the demo only uses Relu and Softmax, the tests
// cover the rest.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Activation {
    None = 0,
    Relu = 1,
    /// `1 / (1 + e^-v)`, squashing any value into (0, 1).
    Sigmoid = 2,
    /// Row-wise: `e^v / sum(e^row)`, turning each row into probabilities that
    /// sum to 1. Usually only on the output layer. Unlike the others it needs the
    /// whole row, so on the GPU it runs as a separate dispatch after the layer
    /// (`shader_softmax.wgsl`); the dense shader just writes the raw values.
    Softmax = 3,
}

/// Sigmoid input is clamped to this range on both GPU and CPU. Outside it the
/// result is already 0 or 1 to f32 precision, and clamping keeps `exp` from
/// overflowing, which WGSL leaves undefined.
pub const SIGMOID_CLAMP: f32 = 30.0;

impl Activation {
    /// The code the dense shader expects in its uniform.
    pub fn shader_code(self) -> u32 {
        self as u32
    }

    /// Applies an element-wise activation to one value on the CPU.
    ///
    /// # Panics
    /// For `Softmax`, which depends on the whole row: use [`apply_row`](Self::apply_row).
    pub fn apply(self, v: f32) -> f32 {
        match self {
            Activation::None => v,
            Activation::Relu => v.max(0.0),
            Activation::Sigmoid => {
                let v = v.clamp(-SIGMOID_CLAMP, SIGMOID_CLAMP);
                1.0 / (1.0 + (-v).exp())
            }
            Activation::Softmax => panic!("softmax is row-wise; use Activation::apply_row"),
        }
    }

    /// Applies the activation to one output row in place on the CPU (the
    /// reference forward pass). Works for every activation, including softmax.
    pub fn apply_row(self, row: &mut [f32]) {
        match self {
            Activation::Softmax => {
                // Subtract the max first so exp() stays <= 1 and cannot overflow.
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
            other => {
                for v in row.iter_mut() {
                    *v = other.apply(*v);
                }
            }
        }
    }
}

/// One fully connected layer, as host data.
#[derive(Clone, Debug)]
pub struct DenseLayer {
    /// `inputs x outputs`, row-major.
    pub weights: Vec<f32>,
    /// `outputs` values, one per output column.
    pub bias: Vec<f32>,
    pub inputs: usize,
    pub outputs: usize,
    pub activation: Activation,
}

/// Checks that a stack of layers forms a valid network:
/// - at least one layer
/// - every width is non-zero and at most `MAX_DIMENSION`
/// - `weights` is `inputs * outputs` long and `bias` is `outputs` long
/// - each layer's `inputs` equals the previous layer's `outputs`
pub fn validate_layers(layers: &[DenseLayer]) -> Result<(), BoxError> {
    if layers.is_empty() {
        return Err("Network must have at least one layer".into());
    }

    for (i, layer) in layers.iter().enumerate() {
        let (inp, out) = (layer.inputs, layer.outputs);

        if inp == 0 || out == 0 {
            return Err(format!("Layer {i}: inputs ({inp}) and outputs ({out}) must be non-zero").into());
        }
        if inp > MAX_DIMENSION || out > MAX_DIMENSION {
            return Err(format!(
                "Layer {i}: width exceeds MAX_DIMENSION ({MAX_DIMENSION}): inputs={inp}, outputs={out}"
            )
            .into());
        }
        if layer.weights.len() != inp * out {
            return Err(format!(
                "Layer {i}: weights length ({}) does not match inputs * outputs ({inp} * {out} = {})",
                layer.weights.len(),
                inp * out
            )
            .into());
        }
        if layer.bias.len() != out {
            return Err(format!(
                "Layer {i}: bias length ({}) does not match outputs ({out})",
                layer.bias.len()
            )
            .into());
        }
        if i > 0 {
            let prev_out = layers[i - 1].outputs;
            if inp != prev_out {
                return Err(format!(
                    "Layer {i}: inputs ({inp}) does not match layer {}'s outputs ({prev_out})",
                    i - 1
                )
                .into());
            }
        }
    }

    Ok(())
}

/// Runs the same network on the CPU, layer by layer, as a reference for checking
/// [`Mlp::forward`]. Plain loops, no tiling: slow but obviously correct.
///
/// Note: the GPU sums each dot product in a different order (tile by tile), so
/// compare with a relative tolerance, not exact equality.
pub fn cpu_forward(layers: &[DenseLayer], input: &[f32], batch: usize) -> Result<Vec<f32>, BoxError> {
    validate_layers(layers)?;
    if batch == 0 {
        return Err("Batch size must be non-zero".into());
    }
    if input.len() != batch * layers[0].inputs {
        return Err(format!(
            "Input length ({}) does not match batch * inputs ({batch} * {} = {})",
            input.len(),
            layers[0].inputs,
            batch * layers[0].inputs
        )
        .into());
    }

    let mut x = input.to_vec();
    for layer in layers {
        let (k, n) = (layer.inputs, layer.outputs);
        let mut y = vec![0.0f32; batch * n];
        for row in 0..batch {
            for col in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[row * k + i] * layer.weights[i * n + col];
                }
                y[row * n + col] = acc + layer.bias[col];
            }
            layer.activation.apply_row(&mut y[row * n..(row + 1) * n]);
        }
        x = y;
    }
    Ok(x)
}

/// One layer whose weights and bias already live on the GPU.
struct GpuLayer {
    weights: wgpu::Buffer,
    bias: wgpu::Buffer,
    /// Bind group 1: params + weights + bias, built once in `Mlp::new`.
    group: wgpu::BindGroup,
    inputs: usize,
    outputs: usize,
    activation: Activation,
}

/// The ping-pong pair, built for one batch size and reused by every
/// [`Mlp::forward`] call with that batch size.
struct Activations {
    batch: usize,
    ping: wgpu::Buffer,
    pong: wgpu::Buffer,
    /// Bind group 0, two variants:
    /// `sets[0]` = { in: ping, out: pong } for even layers,
    /// `sets[1]` = { in: pong, out: ping } for odd layers.
    sets: [wgpu::BindGroup; 2],
}

/// A network ready for inference: weights are uploaded once in [`Mlp::new`]
/// and reused by every forward pass, so inference only moves activations.
pub struct Mlp<'e> {
    engine: &'e MatrixMultiplier,
    layers: Vec<GpuLayer>,
    /// Cached ping-pong pair; rebuilt when the batch size changes.
    activations: Option<Activations>,
}

impl<'e> Mlp<'e> {
    /// Validates `layers`, uploads every weight matrix and bias vector to the
    /// GPU of `engine`, and builds each layer's bind group. GPU errors (e.g. out
    /// of memory) are returned as `Err`.
    pub async fn new(engine: &'e MatrixMultiplier, layers: &[DenseLayer]) -> Result<Self, BoxError> {
        validate_layers(layers)?;

        let device = &engine.device;
        push_error_scopes(device);

        let gpu_layers = layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let upload = |what: &str, data: &[f32]| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("Layer {i} {what}")),
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                };
                let weights = upload("Weights", &layer.weights);
                let bias = upload("Bias", &layer.bias);

                // Everything about this layer that never changes, bound once.
                let params = LayerParams {
                    k: layer.inputs as u32,
                    n: layer.outputs as u32,
                    activation: layer.activation.shader_code(),
                    _pad: 0,
                };
                let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {i} Params")),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Layer {i} (group 1)")),
                    layout: &engine.layer_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: weights.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: bias.as_entire_binding() },
                    ],
                });

                GpuLayer {
                    weights,
                    bias,
                    group,
                    inputs: layer.inputs,
                    outputs: layer.outputs,
                    activation: layer.activation,
                }
            })
            .collect();

        pop_error_scopes(device).await?;

        Ok(Self {
            engine,
            layers: gpu_layers,
            activations: None,
        })
    }

    /// Batch size the cached ping-pong pair was built for, if any.
    pub fn cached_batch(&self) -> Option<usize> {
        self.activations.as_ref().map(|a| a.batch)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Features per input row (the first layer's `inputs`).
    pub fn input_size(&self) -> usize {
        self.layers[0].inputs
    }

    /// Values per output row (the last layer's `outputs`).
    pub fn output_size(&self) -> usize {
        self.layers[self.layers.len() - 1].outputs
    }

    /// Widest activation in the chain (input included); sizes the ping-pong buffers.
    pub fn max_width(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.outputs)
            .fold(self.input_size(), usize::max)
    }

    /// Bytes of weights + biases held on the GPU.
    pub fn gpu_param_bytes(&self) -> u64 {
        self.layers.iter().map(|l| l.weights.size() + l.bias.size()).sum()
    }

    /// Runs the network on `batch` rows of `input` (`batch x input_size`, row-major)
    /// and returns `batch x output_size` values.
    ///
    /// Everything happens in one submission:
    /// 1. upload the input into the `ping` buffer
    /// 2. layer i reads one buffer and writes the other (ping -> pong -> ping ...),
    ///    by binding activation set `i % 2` and its own layer group; wgpu makes
    ///    each dispatch wait for the previous one's writes
    /// 3. read back only the buffer the last layer wrote
    ///
    /// The ping-pong buffers and their two bind groups are cached per batch
    /// size (hence `&mut self`): a call with the same batch size as the last one
    /// builds no activation buffers or bind groups. It still allocates the
    /// small readback and timestamp staging buffers in `run_pass`.
    ///
    /// `kernel_time` covers the whole chain of layers on the GPU.
    pub async fn forward(&mut self, input: &[f32], batch: usize) -> Result<GpuRun, BoxError> {
        self.check_input(input, batch)?;

        let engine = self.engine;
        push_error_scopes(&engine.device);

        if self.cached_batch() != Some(batch) {
            self.activations = Some(self.build_activations(batch));
        }
        let output_size = self.output_size();
        let (layers, acts) = (&self.layers, self.activations.as_ref().expect("built above"));

        // Staged now, executed at the start of the submit in run_pass. The
        // previous call already waited for its readback, so ping is free.
        engine.queue.write_buffer(&acts.ping, 0, bytemuck::cast_slice(input));

        // Even layers write pong, odd layers write ping.
        let last = layers.len() - 1;
        let output = if last.is_multiple_of(2) { &acts.pong } else { &acts.ping };

        let result = engine
            .run_pass(
                |pass| {
                    for (i, layer) in layers.iter().enumerate() {
                        pass.set_bind_group(0, &acts.sets[i % 2], &[]);
                        pass.set_bind_group(1, &layer.group, &[]);

                        pass.set_pipeline(&engine.dense_pipeline);
                        pass.dispatch_workgroups(
                            (layer.outputs as u32).div_ceil(WORKGROUP_SIZE),
                            (batch as u32).div_ceil(WORKGROUP_SIZE),
                            1,
                        );

                        // Softmax needs whole rows: the dense shader wrote raw
                        // values, now normalize them in place, one workgroup
                        // per row, with the same two groups.
                        if layer.activation == Activation::Softmax {
                            pass.set_pipeline(&engine.softmax_pipeline);
                            pass.dispatch_workgroups(1, batch as u32, 1);
                        }
                    }
                },
                output,
                batch * output_size,
            )
            .await;

        // If the GPU reported an error (e.g. out of memory while building the
        // pair), don't keep possibly invalid buffers around for the next call.
        if result.is_err() {
            self.activations = None;
        }
        result
    }

    /// Creates the ping and pong buffers for `batch` rows of the widest layer,
    /// the batch-size uniform, and the two activation bind groups.
    fn build_activations(&self, batch: usize) -> Activations {
        let device = &self.engine.device;
        let act_bytes = (batch * self.max_width() * size_of::<f32>()) as u64;
        let make_act = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: act_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let ping = make_act("Activations Ping");
        let pong = make_act("Activations Pong");

        let batch_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Batch Size"),
            contents: bytemuck::bytes_of(&BatchParams { m: batch as u32, _pad: [0; 3] }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let make_set = |label: &str, input: &wgpu::Buffer, output: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.engine.act_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: batch_uniform.as_entire_binding() },
                ],
            })
        };
        let sets = [
            make_set("Activations Set A (ping -> pong)", &ping, &pong),
            make_set("Activations Set B (pong -> ping)", &pong, &ping),
        ];

        Activations { batch, ping, pong, sets }
    }

    fn check_input(&self, input: &[f32], batch: usize) -> Result<(), BoxError> {
        if batch == 0 || batch > MAX_DIMENSION {
            return Err(format!("Batch size must be 1..={MAX_DIMENSION}, got {batch}").into());
        }
        let expected = batch * self.input_size();
        if input.len() != expected {
            return Err(format!(
                "Input length ({}) does not match batch * input_size ({batch} * {} = {expected})",
                input.len(),
                self.input_size()
            )
            .into());
        }
        Ok(())
    }
}

/// Deterministic pseudo-random values in [-1, 1) (xorshift), so demo and test
/// data are the same on every run.
pub(crate) fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Builds a network with deterministic random weights for the demo and tests.
/// `widths` lists the layer sizes, input first (so `widths.len() - 1` layers).
/// `hidden` is used on every layer but the last, which uses `output`.
///
/// Weights are scaled by `1/sqrt(inputs)` and biases by 0.1, so activations
/// stay around 1 through the chain instead of growing layer by layer.
pub(crate) fn random_layers(
    widths: &[usize],
    seed: u32,
    hidden: Activation,
    output: Activation,
) -> Vec<DenseLayer> {
    let last = widths.len() - 2;
    widths
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let (inputs, outputs) = (w[0], w[1]);
            let scale = 1.0 / (inputs as f32).sqrt();
            let s = seed + 10 * i as u32;
            DenseLayer {
                weights: pseudo_random(inputs * outputs, s).iter().map(|v| v * scale).collect(),
                bias: pseudo_random(outputs, s + 1).iter().map(|v| v * 0.1).collect(),
                inputs,
                outputs,
                activation: if i == last { output } else { hidden },
            }
        })
        .collect()
}

/// Index of the largest value in `row` (the predicted class).
fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best })
        .0
}

/// `cargo run -- mlp`: builds a small 64 -> 128 -> 128 -> 10 network with fixed
/// random weights (ReLU hidden layers, softmax output), runs a batch of 32 on the
/// GPU, checks it against the CPU, and prints timings and the most probable class
/// of the first few rows.
pub fn run_demo() -> Result<(), BoxError> {
    use crate::matrix_multiplier::PipelineMode;
    use std::time::{Duration, Instant};

    const WIDTHS: [usize; 4] = [64, 128, 128, 10];
    const BATCH: usize = 32;
    const CALLS: u32 = 200;

    // ReLU on hidden layers, softmax on the last one so each row comes out as
    // 10 class probabilities.
    let layers = random_layers(&WIDTHS, 1000, Activation::Relu, Activation::Softmax);
    let input = pseudo_random(BATCH * WIDTHS[0], 7);

    // The multiply mode doesn't matter here: the network uses the dense pipeline.
    let engine = pollster::block_on(MatrixMultiplier::new(PipelineMode::Tiled))?;
    println!("Using GPU adapter: {:?}", engine.adapter_info().name);

    let mut net = pollster::block_on(Mlp::new(&engine, &layers))?;
    let widths: Vec<String> = WIDTHS.iter().map(|w| w.to_string()).collect();
    println!(
        "Network: {} ({} layers, {:.1} KB of weights on the GPU), batch {BATCH}",
        widths.join(" -> "),
        net.num_layers(),
        net.gpu_param_bytes() as f64 / 1024.0
    );

    let cpu_start = Instant::now();
    let expected = cpu_forward(&layers, &input, BATCH)?;
    let cpu_time = cpu_start.elapsed();

    // The first call pays one-off warm-up and builds the ping-pong pair.
    let first_start = Instant::now();
    let first = pollster::block_on(net.forward(&input, BATCH))?;
    let first_wall = first_start.elapsed();

    if !crate::probs_eq(&first.output, &expected) {
        return Err("MLP inference failed verification: GPU output differs from CPU".into());
    }
    println!("\nSUCCESS: GPU inference matches the CPU reference!");

    let outputs = net.output_size();
    println!("\n--- Predicted class (most probable of {outputs}, after softmax), first 8 rows ---");
    for row in 0..8 {
        let probs = &first.output[row * outputs..(row + 1) * outputs];
        let cpu_probs = &expected[row * outputs..(row + 1) * outputs];
        println!(
            "row {row}: GPU class {}  CPU class {}  (p = {:.4}, row sums to {:.6})",
            argmax(probs),
            argmax(cpu_probs),
            probs[argmax(probs)],
            probs.iter().sum::<f32>()
        );
    }

    // Steady state: repeated calls reuse the cached ping-pong pair.
    let mut device_total = Duration::ZERO;
    let steady_start = Instant::now();
    for _ in 0..CALLS {
        let run = pollster::block_on(net.forward(&input, BATCH))?;
        device_total += run.kernel_time.unwrap_or_default();
    }
    let steady_wall = steady_start.elapsed() / CALLS;

    let ms = crate::ms;
    println!("\n--- Inference Timing ({} layers in one submit) ---", net.num_layers());
    println!("CPU forward:                        {:>10.3} ms", ms(cpu_time));
    println!("GPU first call, total (wall):       {:>10.3} ms", ms(first_wall));
    println!("GPU later calls, total (wall):      {:>10.3} ms  (average of {CALLS})", ms(steady_wall));
    if first.kernel_time.is_some() {
        println!("GPU later calls, chain (device):    {:>10.3} ms", ms(device_total / CALLS));
    } else {
        println!("GPU later calls, chain (device):           n/a  (adapter lacks TIMESTAMP_QUERY)");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(inputs: usize, outputs: usize) -> DenseLayer {
        DenseLayer {
            weights: vec![0.0; inputs * outputs],
            bias: vec![0.0; outputs],
            inputs,
            outputs,
            activation: Activation::Relu,
        }
    }

    #[test]
    fn accepts_valid_stack() {
        assert!(validate_layers(&[layer(64, 128), layer(128, 128), layer(128, 10)]).is_ok());
        assert!(validate_layers(&[layer(3, 1)]).is_ok());
    }

    #[test]
    fn rejects_empty_network() {
        assert!(validate_layers(&[]).is_err());
    }

    #[test]
    fn rejects_zero_or_oversized_width() {
        assert!(validate_layers(&[layer(0, 4)]).is_err());
        assert!(validate_layers(&[layer(4, 0)]).is_err());
        assert!(validate_layers(&[layer(MAX_DIMENSION + 1, 1)]).is_err());
    }

    #[test]
    fn rejects_wrong_weight_or_bias_length() {
        let mut bad_w = layer(4, 3);
        bad_w.weights.pop();
        assert!(validate_layers(&[bad_w]).is_err());

        let mut bad_b = layer(4, 3);
        bad_b.bias.push(0.0);
        assert!(validate_layers(&[bad_b]).is_err());
    }

    #[test]
    fn rejects_mismatched_chain() {
        let err = validate_layers(&[layer(8, 16), layer(15, 4)]).unwrap_err();
        assert!(err.to_string().contains("Layer 1"), "unexpected error: {err}");
    }

    /// Same 2 -> 2 (ReLU) -> 3 network as the GPU test `mlp_forward_hand_computed`.
    #[test]
    fn cpu_forward_hand_computed() {
        #[rustfmt::skip]
        let layers = [
            DenseLayer {
                weights: vec![1.0, -1.0,
                              2.0,  1.0],
                bias: vec![-6.0, 0.5],
                inputs: 2, outputs: 2,
                activation: Activation::Relu,
            },
            DenseLayer {
                weights: vec![1.0, 0.0, 2.0,
                              1.0, 1.0, 0.0],
                bias: vec![0.0, 0.0, 0.0],
                inputs: 2, outputs: 3,
                activation: Activation::None,
            },
        ];
        let out = cpu_forward(&layers, &[1.0, 2.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(out, vec![1.5, 1.5, 0.0, 0.5, 0.5, 0.0]);
    }

    #[test]
    fn cpu_forward_rejects_bad_input() {
        let layers = [layer(4, 2)];
        assert!(cpu_forward(&layers, &[1.0; 4], 0).is_err());
        assert!(cpu_forward(&layers, &[1.0; 7], 2).is_err());
        assert!(cpu_forward(&[], &[1.0; 4], 1).is_err());
    }

    #[test]
    fn activation_codes_and_cpu_apply() {
        assert_eq!(Activation::None.shader_code(), 0);
        assert_eq!(Activation::Relu.shader_code(), 1);
        assert_eq!(Activation::Sigmoid.shader_code(), 2);
        assert_eq!(Activation::Softmax.shader_code(), 3);
        assert_eq!(Activation::Relu.apply(-2.0), 0.0);
        assert_eq!(Activation::Relu.apply(3.0), 3.0);
        assert_eq!(Activation::None.apply(-2.0), -2.0);
    }

    #[test]
    fn softmax_cpu_values() {
        let softmax = |v: &[f32]| {
            let mut row = v.to_vec();
            Activation::Softmax.apply_row(&mut row);
            row
        };

        // Equal inputs -> uniform probabilities.
        assert_eq!(softmax(&[2.0, 2.0, 2.0, 2.0]), vec![0.25; 4]);
        // A single value is certain.
        assert_eq!(softmax(&[-7.0]), vec![1.0]);

        // Known values: softmax([1,2,3]) = [0.0900, 0.2447, 0.6652].
        let p = softmax(&[1.0, 2.0, 3.0]);
        let expected = [0.090_030_57, 0.244_728_48, 0.665_240_9];
        assert!(p.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-6), "{p:?}");
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // Shifting a row doesn't change it, and huge values don't overflow
        // (e^1000 would be inf without the max subtraction).
        let shifted = softmax(&[1001.0, 1002.0, 1003.0]);
        assert!(shifted.iter().zip(&p).all(|(a, b)| (a - b).abs() < 1e-6), "{shifted:?}");
    }

    #[test]
    fn apply_row_matches_apply_for_elementwise() {
        let input = [-2.0, -0.5, 0.0, 0.5, 2.0];
        for act in [Activation::None, Activation::Relu, Activation::Sigmoid] {
            let mut row = input.to_vec();
            act.apply_row(&mut row);
            let expected: Vec<f32> = input.iter().map(|&v| act.apply(v)).collect();
            assert_eq!(row, expected, "{act:?}");
        }
    }

    #[test]
    #[should_panic(expected = "row-wise")]
    fn softmax_apply_single_value_panics() {
        Activation::Softmax.apply(1.0);
    }

    #[test]
    fn sigmoid_cpu_values() {
        let s = |v| Activation::Sigmoid.apply(v);
        assert_eq!(s(0.0), 0.5);
        // sigmoid(ln 3) = 1 / (1 + 1/3) = 0.75
        assert!((s(3.0f32.ln()) - 0.75).abs() < 1e-6);
        // Symmetry: sigmoid(-v) = 1 - sigmoid(v)
        assert!((s(-1.5) - (1.0 - s(1.5))).abs() < 1e-6);
        // Extreme inputs saturate without overflow or NaN.
        assert_eq!(s(1e30), 1.0);
        assert!(s(-1e30) >= 0.0 && s(-1e30) < 1e-12);
        assert!(!s(f32::MAX).is_nan() && !s(f32::MIN).is_nan());
    }
}
