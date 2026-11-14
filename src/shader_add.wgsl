// Element-wise matrix addition: C = A + B, all M x N.
// BLOCK_SIZE is injected by the host (WORKGROUP_SIZE in matrix_multiplier.rs).

struct Dimensions {
    M: u32,
    K: u32, // unused by addition; kept so all shaders share one uniform layout
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> matrixA: array<f32>;
@group(0) @binding(2) var<storage, read> matrixB: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrixC: array<f32>;

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    let idx = row * dims.N + col;
    matrixC[idx] = matrixA[idx] + matrixB[idx];
}
