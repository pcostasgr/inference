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

use crate::matrix_multiplier::{
    pop_error_scopes, push_error_scopes, GpuRun, MatrixMultiplier, MAX_DIMENSION, WORKGROUP_SIZE,
};
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;
use wgpu::util::DeviceExt;

/// Matches `struct Params` in `shader_dense.wgsl` (16 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DenseParams {
    m: u32,
    k: u32,
    n: u32,
    activation: u32,
}

type BoxError = Box<dyn std::error::Error>;

/// Activation applied after `X * W + bias`. Values match `params.activation`
/// in `shader_dense.wgsl`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Activation {
    None = 0,
    Relu = 1,
    /// `1 / (1 + e^-v)`, squashing any value into (0, 1).
    // Part of the API; only the tests use it so far (the demo uses ReLU).
    #[allow(dead_code)]
    Sigmoid = 2,
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

    /// Applies the activation on the CPU (for the reference forward pass).
    pub fn apply(self, v: f32) -> f32 {
        match self {
            Activation::None => v,
            Activation::Relu => v.max(0.0),
            Activation::Sigmoid => {
                let v = v.clamp(-SIGMOID_CLAMP, SIGMOID_CLAMP);
                1.0 / (1.0 + (-v).exp())
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
                y[row * n + col] = layer.activation.apply(acc + layer.bias[col]);
            }
        }
        x = y;
    }
    Ok(x)
}

/// One layer whose weights and bias already live on the GPU.
struct GpuLayer {
    weights: wgpu::Buffer,
    bias: wgpu::Buffer,
    inputs: usize,
    outputs: usize,
    activation: Activation,
}

/// A network ready for inference: weights are uploaded once in [`Mlp::new`]
/// and reused by every forward pass, so inference only moves activations.
pub struct Mlp<'e> {
    engine: &'e MatrixMultiplier,
    layers: Vec<GpuLayer>,
}

impl<'e> Mlp<'e> {
    /// Validates `layers` and uploads every weight matrix and bias vector to the
    /// GPU of `engine`. GPU errors (e.g. out of memory) are returned as `Err`.
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
                GpuLayer {
                    weights: upload("Weights", &layer.weights),
                    bias: upload("Bias", &layer.bias),
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
        })
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
    /// 2. layer i reads one buffer and writes the other (ping -> pong -> ping ...);
    ///    wgpu makes each dispatch wait for the previous one's writes
    /// 3. read back only the buffer the last layer wrote
    ///
    /// `kernel_time` covers the whole chain of layers on the GPU.
    pub async fn forward(&self, input: &[f32], batch: usize) -> Result<GpuRun, BoxError> {
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

        let engine = self.engine;
        let device = &engine.device;
        push_error_scopes(device);

        // Two activation buffers, each big enough for the widest layer.
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

        // Staged now, executed at the start of the submit in run_pass.
        engine.queue.write_buffer(&ping, 0, bytemuck::cast_slice(input));

        // One params uniform + bind group per layer. Even layers read ping and
        // write pong; odd layers the reverse.
        let bind_groups: Vec<wgpu::BindGroup> = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let (src, dst) = if i.is_multiple_of(2) { (&ping, &pong) } else { (&pong, &ping) };
                let params = DenseParams {
                    m: batch as u32,
                    k: layer.inputs as u32,
                    n: layer.outputs as u32,
                    activation: layer.activation.shader_code(),
                };
                let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {i} Params")),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Layer {i} Bind Group")),
                    layout: &engine.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: layer.weights.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: dst.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: layer.bias.as_entire_binding() },
                    ],
                })
            })
            .collect();

        let last = self.layers.len() - 1;
        // Same parity rule as above: the last layer wrote pong if its index is even.
        let output = if last.is_multiple_of(2) { &pong } else { &ping };

        engine
            .run_pass(
                |pass| {
                    pass.set_pipeline(&engine.dense_pipeline);
                    for (layer, bind_group) in self.layers.iter().zip(&bind_groups) {
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.dispatch_workgroups(
                            (layer.outputs as u32).div_ceil(WORKGROUP_SIZE),
                            (batch as u32).div_ceil(WORKGROUP_SIZE),
                            1,
                        );
                    }
                },
                output,
                batch * self.output_size(),
            )
            .await
    }
}

/// Deterministic pseudo-random values in [-1, 1) (xorshift), so the demo network
/// is the same on every run.
fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
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

/// Index of the largest value in `row` (the predicted class).
fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best })
        .0
}

/// `cargo run -- mlp`: builds a small 64 -> 128 -> 128 -> 10 network with fixed
/// random weights, runs a batch of 32 on the GPU, checks it against the CPU, and
/// prints timings and the predicted class of the first few rows.
pub fn run_demo() -> Result<(), BoxError> {
    use crate::matrix_multiplier::PipelineMode;
    use std::time::Instant;

    const WIDTHS: [usize; 4] = [64, 128, 128, 10];
    const BATCH: usize = 32;

    // ReLU on hidden layers, raw scores (logits) out of the last one.
    // Weights scaled by 1/sqrt(inputs) so activations stay around 1.
    let last = WIDTHS.len() - 2;
    let layers: Vec<DenseLayer> = WIDTHS
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let (inputs, outputs) = (w[0], w[1]);
            let scale = 1.0 / (inputs as f32).sqrt();
            let seed = 1000 + 10 * i as u32;
            DenseLayer {
                weights: pseudo_random(inputs * outputs, seed).iter().map(|v| v * scale).collect(),
                bias: pseudo_random(outputs, seed + 1).iter().map(|v| v * 0.1).collect(),
                inputs,
                outputs,
                activation: if i == last { Activation::None } else { Activation::Relu },
            }
        })
        .collect();
    let input = pseudo_random(BATCH * WIDTHS[0], 7);

    // The multiply mode doesn't matter here: the network uses the dense pipeline.
    let engine = pollster::block_on(MatrixMultiplier::new(PipelineMode::Tiled))?;
    println!("Using GPU adapter: {:?}", engine.adapter_info().name);

    let net = pollster::block_on(Mlp::new(&engine, &layers))?;
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

    // The first call pays one-off driver/pipeline warm-up; the second shows the
    // steady-state cost of an inference call.
    let mut runs = Vec::new();
    for _ in 0..2 {
        let start = Instant::now();
        let run = pollster::block_on(net.forward(&input, BATCH))?;
        runs.push((start.elapsed(), run));
    }

    for (_, run) in &runs {
        if !crate::approx_eq(&run.output, &expected) {
            return Err("MLP inference failed verification: GPU output differs from CPU".into());
        }
    }
    println!("\nSUCCESS: GPU inference matches the CPU reference!");

    let outputs = net.output_size();
    let output = &runs[1].1.output;
    println!("\n--- Predicted class (argmax of {outputs} scores), first 8 rows ---");
    for row in 0..8 {
        let scores = &output[row * outputs..(row + 1) * outputs];
        let cpu_scores = &expected[row * outputs..(row + 1) * outputs];
        println!(
            "row {row}: GPU class {}  CPU class {}  (top score {:.4})",
            argmax(scores),
            argmax(cpu_scores),
            scores[argmax(scores)]
        );
    }

    let ms = crate::ms;
    println!("\n--- Inference Timing ({} layers in one submit) ---", net.num_layers());
    println!("CPU forward:                    {:>10.3} ms", ms(cpu_time));
    for (label, (wall, run)) in ["first call", "second call"].iter().zip(&runs) {
        println!("GPU {label:<12} total (wall):   {:>10.3} ms", ms(*wall));
        match run.kernel_time {
            Some(t) => println!("GPU {label:<12} chain (device): {:>10.3} ms", ms(t)),
            None => println!("GPU {label:<12} chain (device):        n/a"),
        }
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
        assert_eq!(Activation::Relu.apply(-2.0), 0.0);
        assert_eq!(Activation::Relu.apply(3.0), 3.0);
        assert_eq!(Activation::None.apply(-2.0), -2.0);
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
