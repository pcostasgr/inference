// Dense (fully connected) layer: Y = act(X * W + bias)
//   X: M x K (batch x inputs), W: K x N (inputs x outputs), bias: N, all row-major.
// Tiled like shader_tiled.wgsl, with a bias + activation epilogue.
//
// "Dual bind group" layout: the bindings are split by how often they change,
// so the host can build them ahead of time:
//
//   @group(0)  activations - the ping-pong pair, built once per batch size:
//              set A = { X: ping, Y: pong },  set B = { X: pong, Y: ping }
//   @group(1)  this layer's constants - built once when the network is created
//
// Layer i binds group 0 = (i even ? A : B) and group 1 = layer i's group.
// BLOCK_SIZE is injected by the host (WORKGROUP_SIZE in matrix_multiplier.rs).

struct Batch {
    M: u32,          // rows (batch size)
}

struct LayerParams {
    K: u32,          // inputs
    N: u32,          // outputs
    activation: u32, // 0 = none, 1 = ReLU, 2 = sigmoid, 3 = softmax (raw here,
                     // normalized by shader_softmax.wgsl next)
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> batch: Batch;

@group(1) @binding(0) var<uniform> layer: LayerParams;
@group(1) @binding(1) var<storage, read> W: array<f32>;
@group(1) @binding(2) var<storage, read> bias: array<f32>;

// Must match SIGMOID_CLAMP in mlp.rs.
const SIGMOID_CLAMP: f32 = 30.0;

var<workgroup> tileX: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;
var<workgroup> tileW: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let num_tiles = (layer.K + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let x_col = t * BLOCK_SIZE + local_id.x;
        let w_row = t * BLOCK_SIZE + local_id.y;

        if (row < batch.M && x_col < layer.K) {
            tileX[local_id.y][local_id.x] = X[row * layer.K + x_col];
        } else {
            tileX[local_id.y][local_id.x] = 0.0;
        }

        if (w_row < layer.K && col < layer.N) {
            tileW[local_id.y][local_id.x] = W[w_row * layer.N + col];
        } else {
            tileW[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
            acc = acc + tileX[local_id.y][k] * tileW[k][local_id.x];
        }

        workgroupBarrier();
    }

    if (row < batch.M && col < layer.N) {
        var v = acc + bias[col];
        if (layer.activation == 1u) {
            v = max(v, 0.0);
        } else if (layer.activation == 2u) {
            let c = clamp(v, -SIGMOID_CLAMP, SIGMOID_CLAMP);
            v = 1.0 / (1.0 + exp(-c));
        }
        Y[row * layer.N + col] = v;
    }
}
