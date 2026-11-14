use bytemuck::{Pod, Zeroable};
use std::mem::size_of;
use std::time::Duration;
use wgpu::util::DeviceExt;

/// Maximum dimension allowed for any axis (M, K, N).
/// A 4096 x 4096 f32 matrix is 64 MB, which fits within the default
/// 128 MB `max_storage_buffer_binding_size` limit. Note this does NOT
/// guarantee the dispatch finishes before the Windows TDR timeout (~2 s);
/// large naive multiplies on integrated GPUs can still exceed it.
pub const MAX_DIMENSION: usize = 4096;

/// Workgroup edge length (threads per axis). Injected into the WGSL
/// shaders as `BLOCK_SIZE`, so this is the single source of truth.
///
/// Cannot exceed 16 with default GPU limits: the shaders use
/// `BLOCK_SIZE * BLOCK_SIZE` threads per workgroup, and the default
/// `max_compute_invocations_per_workgroup` is 256.
pub const WORKGROUP_SIZE: u32 = 16;
const _: () = assert!(WORKGROUP_SIZE * WORKGROUP_SIZE <= 256);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Dimensions {
    m: u32,
    k: u32,
    n: u32,
    /// 1 when the multiply shaders should add D (multiply-add); also keeps 16-byte size.
    has_d: u32,
}

/// Pipeline mode selected by the caller
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PipelineMode {
    Naive,
    Tiled,
}

type BoxError = Box<dyn std::error::Error>;

/// Result of one GPU operation (multiply, add, multiply-add or a network
/// forward pass), with its on-GPU execution time when available.
pub struct GpuRun {
    pub output: Vec<f32>,
    /// Time the compute pass took on the GPU, measured with timestamp queries.
    /// `None` if the adapter doesn't support `TIMESTAMP_QUERY`.
    pub kernel_time: Option<Duration>,
}

/// Reusable GPU engine: the device and queue, the chosen multiply pipeline, the
/// element-wise add pipeline, and the dense-layer and softmax pipelines that
/// [`crate::mlp`] uses.
pub struct MatrixMultiplier {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    add_pipeline: wgpu::ComputePipeline,
    /// Layout for multiply / add: uniform, A, B, C (out), D.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Network layout, group 0: the ping-pong activation pair (in, out, batch size).
    pub(crate) act_layout: wgpu::BindGroupLayout,
    /// Network layout, group 1: one layer's constants (params, weights, bias).
    pub(crate) layer_layout: wgpu::BindGroupLayout,
    /// Tiled `Y = act(X * W + bias)` (`shader_dense.wgsl`).
    pub(crate) dense_pipeline: wgpu::ComputePipeline,
    /// In-place row-wise softmax, run after a dense layer (`shader_softmax.wgsl`).
    pub(crate) softmax_pipeline: wgpu::ComputePipeline,
    adapter_info: wgpu::AdapterInfo,
    /// Two-entry timestamp query set (pass begin/end), if supported.
    query_set: Option<wgpu::QuerySet>,
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Pushes validation + out-of-memory error scopes; pair with [`pop_error_scopes`].
pub(crate) fn push_error_scopes(device: &wgpu::Device) {
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
}

/// Pops the scopes pushed by [`push_error_scopes`], returning the first error captured.
pub(crate) async fn pop_error_scopes(device: &wgpu::Device) -> Result<(), BoxError> {
    let validation = device.pop_error_scope().await;
    let oom = device.pop_error_scope().await;
    match validation.or(oom) {
        Some(err) => Err(format!("wgpu error: {err}").into()),
        None => Ok(()),
    }
}

impl MatrixMultiplier {
    /// Initializes the GPU context and compiles the pipelines: the multiply
    /// kernel selected by `mode`, plus add, dense layer and softmax.
    pub async fn new(mode: PipelineMode) -> Result<Self, BoxError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find an appropriate GPU adapter")?;
        let adapter_info = adapter.get_info();

        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features,
                    ..Default::default()
                },
                None,
            )
            .await?;

        let query_set = timestamps.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Kernel Timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });

        let (label, body) = match mode {
            PipelineMode::Naive => ("Naive Matrix Multiply", include_str!("shader.wgsl")),
            PipelineMode::Tiled => ("Tiled Matrix Multiply", include_str!("shader_tiled.wgsl")),
        };

        push_error_scopes(&device);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, true), // D for multiply-add; a 1-element placeholder otherwise
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = create_pipeline(&device, &pipeline_layout, label, body);
        let add_pipeline = create_pipeline(
            &device,
            &pipeline_layout,
            "Matrix Add",
            include_str!("shader_add.wgsl"),
        );

        // Dual bind group layout for the MLP: activations in group 0 (two
        // prebuilt sets that alternate), per-layer constants in group 1.
        let act_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Network: Activations (group 0)"),
            entries: &[
                storage_entry(0, true),  // X: this layer's input (ping or pong)
                storage_entry(1, false), // Y: this layer's output (the other one)
                uniform_entry(2),        // batch size
            ],
        });
        let layer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Network: Layer (group 1)"),
            entries: &[
                uniform_entry(0),       // K, N, activation
                storage_entry(1, true), // weights
                storage_entry(2, true), // bias
            ],
        });
        let network_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Network Pipeline Layout"),
            bind_group_layouts: &[&act_layout, &layer_layout],
            push_constant_ranges: &[],
        });
        let dense_pipeline = create_pipeline(
            &device,
            &network_layout,
            "Dense Layer",
            include_str!("shader_dense.wgsl"),
        );
        let softmax_pipeline = create_pipeline(
            &device,
            &network_layout,
            "Row Softmax",
            include_str!("shader_softmax.wgsl"),
        );

        pop_error_scopes(&device).await?;

        Ok(Self {
            device,
            queue,
            pipeline,
            add_pipeline,
            bind_group_layout,
            act_layout,
            layer_layout,
            dense_pipeline,
            softmax_pipeline,
            adapter_info,
            query_set,
        })
    }

    /// Information about the GPU adapter this engine runs on.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Whether operations can report on-GPU time (`GpuRun::kernel_time`).
    pub fn has_kernel_timer(&self) -> bool {
        self.query_set.is_some()
    }

    /// Multiplies matrix `A` (m x k) and matrix `B` (k x n) using the active pipeline.
    /// Validates dimensions against MAX_DIMENSION and slice lengths.
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn multiply(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, BoxError> {
        Ok(self.multiply_timed(matrix_a, matrix_b, m, k, n).await?.output)
    }

    /// Like [`multiply`](Self::multiply), but also returns the GPU kernel time.
    pub async fn multiply_timed(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuRun, BoxError> {
        self.matmul(matrix_a, matrix_b, None, m, k, n).await
    }

    /// Fused multiply-add: C = A (m x k) * B (k x n) + D (m x n), in a single kernel.
    /// Cheaper than `multiply` then `add`, which round-trips A*B through the CPU.
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn multiply_add(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        matrix_d: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, BoxError> {
        Ok(self
            .multiply_add_timed(matrix_a, matrix_b, matrix_d, m, k, n)
            .await?
            .output)
    }

    /// Like [`multiply_add`](Self::multiply_add), but also returns the GPU kernel time.
    pub async fn multiply_add_timed(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        matrix_d: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuRun, BoxError> {
        self.matmul(matrix_a, matrix_b, Some(matrix_d), m, k, n).await
    }

    /// Shared validation + dispatch for multiply (`d = None`) and multiply-add.
    async fn matmul(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        matrix_d: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuRun, BoxError> {
        if m == 0 || k == 0 || n == 0 {
            return Err("Matrix dimensions must be non-zero".into());
        }

        if m > MAX_DIMENSION || k > MAX_DIMENSION || n > MAX_DIMENSION {
            return Err(format!(
                "Dimension exceeds MAX_DIMENSION ({MAX_DIMENSION}): M={m}, K={k}, N={n}"
            )
            .into());
        }

        if matrix_a.len() != m * k {
            return Err(format!(
                "Matrix A slice length ({}) does not match m * k ({m} * {k} = {})",
                matrix_a.len(),
                m * k
            )
            .into());
        }

        if matrix_b.len() != k * n {
            return Err(format!(
                "Matrix B slice length ({}) does not match k * n ({k} * {n} = {})",
                matrix_b.len(),
                k * n
            )
            .into());
        }

        if let Some(d) = matrix_d
            && d.len() != m * n
        {
            return Err(format!(
                "Matrix D slice length ({}) does not match m * n ({m} * {n} = {})",
                d.len(),
                m * n
            )
            .into());
        }

        // Dimensions are bounded by MAX_DIMENSION, so these casts cannot truncate.
        let dims = Dimensions {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            has_d: matrix_d.is_some() as u32,
        };

        self.dispatch(&self.pipeline, dims, matrix_a, matrix_b, matrix_d)
            .await
    }

    /// Adds two `rows x cols` matrices element-wise on the GPU: C = A + B.
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn add(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, BoxError> {
        Ok(self.add_timed(matrix_a, matrix_b, rows, cols).await?.output)
    }

    /// Like [`add`](Self::add), but also returns the GPU kernel time.
    pub async fn add_timed(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<GpuRun, BoxError> {
        if rows == 0 || cols == 0 {
            return Err("Matrix dimensions must be non-zero".into());
        }

        if rows > MAX_DIMENSION || cols > MAX_DIMENSION {
            return Err(format!(
                "Dimension exceeds MAX_DIMENSION ({MAX_DIMENSION}): rows={rows}, cols={cols}"
            )
            .into());
        }

        let len = rows * cols;
        if matrix_a.len() != len || matrix_b.len() != len {
            return Err(format!(
                "Slice lengths (A={}, B={}) must both equal rows * cols ({rows} * {cols} = {len})",
                matrix_a.len(),
                matrix_b.len()
            )
            .into());
        }

        // k is unused by the add shader; dispatch only reads m (rows) and n (cols).
        let dims = Dimensions {
            m: rows as u32,
            k: 0,
            n: cols as u32,
            has_d: 0,
        };

        self.dispatch(&self.add_pipeline, dims, matrix_a, matrix_b, None)
            .await
    }

    /// Uploads A, B (and D if given), runs `pipeline` over an `m x n` output grid,
    /// and reads C back. Inputs must already be validated against `dims`.
    async fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        dims: Dimensions,
        matrix_a: &[f32],
        matrix_b: &[f32],
        matrix_d: Option<&[f32]>,
    ) -> Result<GpuRun, BoxError> {
        let (m, n) = (dims.m as usize, dims.n as usize);

        push_error_scopes(&self.device);

        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dimensions Uniform Buffer"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A Storage Buffer"),
            contents: bytemuck::cast_slice(matrix_a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B Storage Buffer"),
            contents: bytemuck::cast_slice(matrix_b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Binding 4 must always be bound; without D, a 1-element placeholder the
        // shaders never read (they check `has_d`).
        let d_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix D Storage Buffer"),
            contents: bytemuck::cast_slice(matrix_d.unwrap_or(&[0.0])),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let c_buffer_size = (m * n * size_of::<f32>()) as u64;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C Storage Buffer"),
            size: c_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: d_buffer.as_entire_binding() },
            ],
        });

        self.run_pass(
            |pass| {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    dims.n.div_ceil(WORKGROUP_SIZE),
                    dims.m.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            },
            &c_buffer,
            m * n,
        )
        .await
    }

    /// Records one timed compute pass (via `record`), submits it, and reads back the
    /// first `out_len` floats of `output`.
    ///
    /// The caller must have called [`push_error_scopes`] before creating the
    /// resources used here; this pops them so GPU errors surface as `Err`.
    /// Dispatches recorded in the same pass that write then read a buffer are
    /// synchronized by wgpu automatically.
    pub(crate) async fn run_pass(
        &self,
        record: impl FnOnce(&mut wgpu::ComputePass<'_>),
        output: &wgpu::Buffer,
        out_len: usize,
    ) -> Result<GpuRun, BoxError> {
        let out_bytes = (out_len * size_of::<f32>()) as u64;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer for Readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Command Encoder"),
        });

        {
            let timestamp_writes =
                self.query_set
                    .as_ref()
                    .map(|query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    });
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matrix Compute Pass"),
                timestamp_writes,
            });
            record(&mut compute_pass);
        }

        encoder.copy_buffer_to_buffer(output, 0, &staging_buffer, 0, out_bytes);

        // Resolve the two u64 timestamps and copy them somewhere mappable.
        const TS_BYTES: u64 = 2 * size_of::<u64>() as u64;
        let ts_staging = self.query_set.as_ref().map(|query_set| {
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestamp Resolve Buffer"),
                size: TS_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestamp Staging Buffer"),
                size: TS_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.resolve_query_set(query_set, 0..2, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, TS_BYTES);
            staging
        });

        self.queue.submit(Some(encoder.finish()));

        let result_rx = map_read(&staging_buffer);
        let ts_rx = ts_staging.as_ref().map(map_read);

        let _ = self.device.poll(wgpu::Maintain::wait());

        // Always pop the scopes so they don't leak, and report GPU errors first:
        // they explain a failed map better than the map error itself.
        pop_error_scopes(&self.device).await?;

        let output = read_mapped::<f32>(&staging_buffer, result_rx)?;
        let kernel_time = match (ts_staging.as_ref(), ts_rx) {
            (Some(buf), Some(rx)) => {
                let ts = read_mapped::<u64>(buf, rx)?;
                let ticks = ts[1].checked_sub(ts[0]);
                let period_ns = self.queue.get_timestamp_period() as f64;
                ticks.map(|t| Duration::from_nanos((t as f64 * period_ns) as u64))
            }
            _ => None,
        };

        Ok(GpuRun { output, kernel_time })
    }
}

/// Compiles a WGSL compute shader (with `BLOCK_SIZE` injected) into a pipeline.
fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    label: &str,
    body: &str,
) -> wgpu::ComputePipeline {
    let source = format!("const BLOCK_SIZE: u32 = {WORKGROUP_SIZE}u;\n{body}");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

type MapReceiver = std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

/// Starts mapping the whole buffer for reading; completes on the next `device.poll`.
fn map_read(buffer: &wgpu::Buffer) -> MapReceiver {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        // The receiver only disappears if we've already returned; nothing to do then.
        let _ = tx.send(result);
    });
    rx
}

/// Waits for a [`map_read`] to finish, copies the contents out and unmaps.
fn read_mapped<T: Pod>(buffer: &wgpu::Buffer, rx: MapReceiver) -> Result<Vec<T>, BoxError> {
    rx.recv()
        .map_err(|_| "Buffer map callback was never invoked")?
        .map_err(|e| format!("Failed to map staging buffer: {e}"))?;
    let data = buffer.slice(..).get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    Ok(out)
}
