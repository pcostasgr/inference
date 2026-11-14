// BLOCK_SIZE is injected by the host (WORKGROUP_SIZE in matrix_multiplier.rs).

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    has_d: u32, // 1 = multiply-add (C = A*B + D), 0 = plain multiply
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> matrixA: array<f32>;
@group(0) @binding(2) var<storage, read> matrixB: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrixC: array<f32>;
@group(0) @binding(4) var<storage, read> matrixD: array<f32>; // M x N; placeholder when has_d == 0

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        let a_idx = row * dims.K + k;
        let b_idx = k * dims.N + col;
        sum = sum + matrixA[a_idx] * matrixB[b_idx];
    }

    let c_idx = row * dims.N + col;
    if (dims.has_d != 0u) {
        sum = sum + matrixD[c_idx];
    }
    matrixC[c_idx] = sum;
}
