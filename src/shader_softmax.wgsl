// Row-wise softmax, in place: for each of the M rows of Y (M x N),
//   Y[r][c] = exp(Y[r][c] - max_r) / sum_c' exp(Y[r][c'] - max_r)
//
// Softmax needs the whole row, so it can't be fused into the dense shader's
// per-element write. It runs right after a dense layer with the same two bind
// groups as shader_dense.wgsl: Y is that layer's output (group 0, binding 1),
// the batch size comes from group 0, and the row width N from the layer's
// group 1.
//
// One workgroup per row (dispatch 1 x M x 1); its threads stride across the
// row, then combine partial max / sum in shared memory. Subtracting the row
// max first keeps exp() <= 1, so it can never overflow.
// BLOCK_SIZE is injected by the host (WORKGROUP_SIZE in matrix_multiplier.rs).

struct Batch {
    M: u32,
}

struct LayerParams {
    K: u32,          // unused
    N: u32,
    activation: u32, // unused (always softmax here)
    _pad: u32,
}

@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> batch: Batch;
@group(1) @binding(0) var<uniform> layer: LayerParams;

const WG: u32 = BLOCK_SIZE * BLOCK_SIZE; // 256 threads per row
var<workgroup> scratch: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32
) {
    // Same for every thread in the group, so the barriers stay uniform.
    let row = wg_id.y;
    if (row >= batch.M) {
        return;
    }
    let base = row * layer.N;

    // 1. Row max.
    var local_max = -3.402823e38;
    for (var c = lid; c < layer.N; c = c + WG) {
        local_max = max(local_max, Y[base + c]);
    }
    scratch[lid] = local_max;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s / 2u) {
        if (lid < s) {
            scratch[lid] = max(scratch[lid], scratch[lid + s]);
        }
        workgroupBarrier();
    }
    let row_max = scratch[0];
    workgroupBarrier(); // everyone reads row_max before scratch is reused

    // 2. Row sum of exp(v - max).
    var local_sum = 0.0;
    for (var c = lid; c < layer.N; c = c + WG) {
        local_sum = local_sum + exp(Y[base + c] - row_max);
    }
    scratch[lid] = local_sum;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s / 2u) {
        if (lid < s) {
            scratch[lid] = scratch[lid] + scratch[lid + s];
        }
        workgroupBarrier();
    }
    let total = scratch[0];

    // 3. Normalize the columns this thread owns.
    for (var c = lid; c < layer.N; c = c + WG) {
        Y[base + c] = exp(Y[base + c] - row_max) / total;
    }
}
