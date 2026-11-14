// Row-wise softmax, in place: for each of the M rows of Y (M x N),
//   Y[r][c] = exp(Y[r][c] - max_r) / sum_c' exp(Y[r][c'] - max_r)
// Subtracting the row max first keeps exp() <= 1, so it can never overflow.
//
// Softmax needs the whole row, so it can't be fused into the dense shader's
// per-element write. It runs as a second dispatch on the dense layer's output,
// with the same bind group: params at binding 0, Y at binding 3 (the other
// bindings are unused here).
//
// One workgroup per row (dispatch 1 x M x 1). Its WG threads stride across the
// row's N columns, then combine their partial max / sum in shared memory.
// BLOCK_SIZE is injected by the host (WORKGROUP_SIZE in matrix_multiplier.rs).

struct Params {
    M: u32,
    K: u32,          // unused
    N: u32,
    activation: u32, // unused (always softmax here)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

const WG: u32 = BLOCK_SIZE * BLOCK_SIZE; // 256 threads per row
var<workgroup> scratch: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32
) {
    // workgroup_id is the same for every thread in the group, so this return
    // keeps the barriers below in uniform control flow.
    let row = wg_id.y;
    if (row >= params.M) {
        return;
    }
    let base = row * params.N;

    // 1. Row max: each thread scans its columns, then a tree reduction.
    var local_max = -3.402823e38; // lowest f32
    for (var c = lid; c < params.N; c = c + WG) {
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
    // Everyone must read row_max before scratch is reused for the sum.
    workgroupBarrier();

    // 2. Row sum of exp(v - max), same pattern.
    var local_sum = 0.0;
    for (var c = lid; c < params.N; c = c + WG) {
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

    // 3. Normalize. Each thread writes only the columns it read, and every read
    //    of other threads' columns finished before the barriers above.
    for (var c = lid; c < params.N; c = c + WG) {
        Y[base + c] = exp(Y[base + c] - row_max) / total;
    }
}
