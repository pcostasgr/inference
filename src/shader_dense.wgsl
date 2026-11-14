// Dense (fully connected) layer: Y = act(X * W + bias), act = none | ReLU | sigmoid
//   X: M x K   (batch x inputs)       row-major
//   W: K x N   (inputs x outputs)     row-major
//   bias: N, broadcast over every row
//   Y: M x N   (batch x outputs)
// Same tiled algorithm as shader_tiled.wgsl, with a bias + activation epilogue.
// BLOCK_SIZE is injected by the host (WORKGROUP_SIZE in matrix_multiplier.rs).

struct Params {
    M: u32,
    K: u32,
    N: u32,
    activation: u32, // 0 = none, 1 = ReLU, 2 = sigmoid,
                     // 3 = softmax: written raw here, normalized per row by
                     //     shader_softmax.wgsl in the next dispatch
}

// Must match SIGMOID_CLAMP in mlp.rs. Beyond +-30 sigmoid is 0 or 1 in f32,
// and clamping keeps exp() from overflowing (undefined in WGSL).
const SIGMOID_CLAMP: f32 = 30.0;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

var<workgroup> tileX: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;
var<workgroup> tileW: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let num_tiles = (params.K + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let x_col = t * BLOCK_SIZE + local_id.x;
        let w_row = t * BLOCK_SIZE + local_id.y;

        if (row < params.M && x_col < params.K) {
            tileX[local_id.y][local_id.x] = X[row * params.K + x_col];
        } else {
            tileX[local_id.y][local_id.x] = 0.0;
        }

        if (w_row < params.K && col < params.N) {
            tileW[local_id.y][local_id.x] = W[w_row * params.N + col];
        } else {
            tileW[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
            acc = acc + tileX[local_id.y][k] * tileW[k][local_id.x];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        var v = acc + bias[col];
        if (params.activation == 1u) {
            v = max(v, 0.0);
        } else if (params.activation == 2u) {
            let c = clamp(v, -SIGMOID_CLAMP, SIGMOID_CLAMP);
            v = 1.0 / (1.0 + exp(-c));
        }
        Y[row * params.N + col] = v;
    }
}
