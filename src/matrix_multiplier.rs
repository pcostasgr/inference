use bytemuck::{Pod, Zeroable};
use std::mem::size_of;
use wgpu::util::DeviceExt;

/// Maximum dimension allowed for any axis (M, K, N).
/// A 4096 x 4096 f32 matrix is 64 MB, which fits within the default
/// 128 MB `max_storage_buffer_binding_size` limit. Note this does NOT
/// guarantee the dispatch finishes before the Windows TDR timeout (~2 s);
/// large naive multiplies on integrated GPUs can still exceed it.
pub const MAX_DIMENSION: usize = 4096;

/// Workgroup edge length (threads per axis). Injected into the WGSL
/// shaders as `BLOCK_SIZE`, so this is the single source of truth.
pub const WORKGROUP_SIZE: u32 = 16;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Dimensions {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

/// Pipeline mode selected by the caller
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PipelineMode {
    Naive,
    Tiled,
}

type BoxError = Box<dyn std::error::Error>;

/// Reusable GPU engine that holds the GPU device, queue, and a single chosen compute pipeline.
pub struct MatrixMultiplier {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    adapter_info: wgpu::AdapterInfo,
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
fn push_error_scopes(device: &wgpu::Device) {
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
}

/// Pops the scopes pushed by [`push_error_scopes`], returning the first error captured.
async fn pop_error_scopes(device: &wgpu::Device) -> Result<(), BoxError> {
    let validation = device.pop_error_scope().await;
    let oom = device.pop_error_scope().await;
    match validation.or(oom) {
        Some(err) => Err(format!("wgpu error: {err}").into()),
        None => Ok(()),
    }
}

impl MatrixMultiplier {
    /// Initializes GPU context and compiles only the selected pipeline mode.
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        let (label, body) = match mode {
            PipelineMode::Naive => ("Naive Matrix Multiply Shader", include_str!("shader.wgsl")),
            PipelineMode::Tiled => ("Tiled Matrix Multiply Shader", include_str!("shader_tiled.wgsl")),
        };
        let shader_source = format!("const BLOCK_SIZE: u32 = {WORKGROUP_SIZE}u;\n{body}");

        push_error_scopes(&device);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        pop_error_scopes(&device).await?;

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            adapter_info,
        })
    }

    /// Information about the GPU adapter this engine runs on.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Multiplies matrix `A` (m x k) and matrix `B` (k x n) using the active pipeline.
    /// Validates dimensions against MAX_DIMENSION and slice lengths.
    pub async fn multiply(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, BoxError> {
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

        // Dimensions are bounded by MAX_DIMENSION, so these casts cannot truncate.
        let dims = Dimensions {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };

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

        let c_buffer_size = (m * n * size_of::<f32>()) as u64;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C Storage Buffer"),
            size: c_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer for Readback"),
            size: c_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matrix Multiply Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = dims.n.div_ceil(WORKGROUP_SIZE);
            let workgroups_y = dims.m.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, c_buffer_size);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            // The receiver only disappears if we've already returned; nothing to do then.
            let _ = tx.send(result);
        });

        let _ = self.device.poll(wgpu::Maintain::wait());

        // Always pop the scopes so they don't leak, and report GPU errors first:
        // they explain a failed map better than the map error itself.
        pop_error_scopes(&self.device).await?;
        rx.recv()
            .map_err(|_| "Buffer map callback was never invoked")?
            .map_err(|e| format!("Failed to map staging buffer: {e}"))?;

        let data = buffer_slice.get_mapped_range();
        let result_c: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result_c)
    }
}
