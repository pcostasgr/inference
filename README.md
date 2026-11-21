# inference

GPU matrix math and a small neural-network inference chain in Rust, using
[wgpu](https://wgpu.rs) compute shaders written in WGSL. It runs on any GPU that
wgpu supports (Vulkan, DirectX 12 or Metal). It was developed on Windows with an
AMD Radeon RX 560 and an Intel UHD Graphics 630.

Every GPU result is checked against a plain CPU implementation, and every
operation can report how long it took, both overall and on the GPU itself.

## What it does

| Operation | API | Shader |
|---|---|---|
| Matrix multiply, `C = A·B` | `multiply`, `multiply_timed` | `shader.wgsl` (naive) or `shader_tiled.wgsl` (tiled) |
| Element-wise add, `C = A + B` | `add`, `add_timed` | `shader_add.wgsl` |
| Fused multiply-add, `C = A·B + D` | `multiply_add`, `multiply_add_timed` | the multiply shader, with D added in its final write |
| MLP inference, `Y = act(X·W + b)` per layer | `Mlp::new`, `Mlp::forward` | `shader_dense.wgsl` (+ `shader_softmax.wgsl`) |

### Matrix engine (`src/matrix_multiplier.rs`)

`MatrixMultiplier` holds the GPU device and the compiled pipelines. All multiply
kernels are compiled up front. You pick one when you create the engine, and
`set_mode` switches between them on the same device:

- **Naive:** each GPU thread computes one output value straight from global memory.
- **Tiled:** each 16×16 workgroup first loads tiles of A and B into fast shared
  memory, then multiplies from there. It reads global memory far less.
- **TiledSwizzled:** the tiled kernel with XOR-swizzled tile storage
  (element (r, c) stored at column `c ^ r`), a standard trick for avoiding
  shared-memory bank conflicts. It's kept as an experiment: this kernel's reads
  were already conflict-free, so the swizzle only adds index math to the inner
  loop. On a Radeon RX 560 it measured **9–25% slower** than `Tiled` (see
  "Measuring kernel changes" below).

Each call uploads the inputs, runs one compute pass and reads the result back.
The `*_timed` versions also return the on-GPU kernel time, measured with
timestamp queries when the GPU supports them.

### MLP inference chain (`src/mlp.rs`)

A small multi-layer perceptron. Each layer computes `Y = act(X·W + bias)`, where
`act` is one of:

- `None`: the raw value.
- `Relu`: `max(v, 0)`.
- `Sigmoid`: `1 / (1 + e^-v)`, squashing values into (0, 1). The input is clamped to
  ±30 first (on both GPU and CPU). Beyond that the result is already 0 or 1 in f32,
  and the clamp keeps `exp` from overflowing.
- `Softmax`: `e^v / sum(e^row)`, turning each row into probabilities that sum to 1.
  It's usually used on the output layer. Softmax needs the whole row, not one value,
  so the dense shader writes raw values and a second dispatch (`shader_softmax.wgsl`)
  normalizes each row in place, still within the same submission. It subtracts the
  row's maximum before `exp`, so it can't overflow.

- `Mlp::new` checks that the layer shapes chain together, then **uploads the
  weights once**. Later inference calls reuse them.
- `Mlp::forward` runs **every layer in one GPU submission**. Activations
  alternate between two buffers ("ping-pong"): layer *i* reads one and writes
  the other. wgpu makes each layer wait for the previous one, and only the final
  layer's output is read back.
- `cpu_forward` runs the same network on the CPU, as a reference.

```rust
let engine = pollster::block_on(MatrixMultiplier::new(PipelineMode::Tiled))?;
let mut net = pollster::block_on(Mlp::new(&engine, &layers))?; // weights -> GPU once
let run = pollster::block_on(net.forward(&input, batch))?;     // all layers, one submit
println!("{:?} on the GPU", run.kernel_time);
```

#### Dual bind groups

The network shaders split their bindings into two groups ("dual descriptor
sets"), by how often they change, so the bindings are set up ahead of time
instead of on every call:

```
group 0 (activations): set A = { in: ping, out: pong, batch }   built once per batch size,
                       set B = { in: pong, out: ping, batch }   cached in the Mlp
group 1 (per layer):   { params, weights, bias }                built once in Mlp::new
```

Layer *i* binds `set A` or `set B` (alternating) as group 0, and its own group 1.
A call with the same batch size as the last one builds no activation buffers or
bind groups. It uploads the input, records the dispatches, submits once and
reads back; only the small readback and timestamp staging buffers are still
allocated per call. Changing the batch size rebuilds the pair once. Because of
that cache, `forward` takes `&mut self`.

An earlier version rebuilt one bind group per layer on every call. On the
3-layer demo network the two measured the same (within 0.2% over 200 calls):
per-call time there is dominated by the submit and readback, not binding setup.

## Running it

Requires a recent stable Rust toolchain (edition 2024) and a GPU with
Vulkan or DirectX 12 drivers.

```bash
cargo run                        # 4x4 demo: multiply, add, multiply-add, with timings
cargo run --release -- mlp       # inference demo: 64 -> 128 -> 128 -> 10 (softmax), batch 32
cargo run --release -- bench     # every multiply kernel, 64..512
cargo run --release -- swizzle   # A/B test: swizzled vs plain tiled kernel
cargo test                       # GPU and CPU tests
```

Use `--release` for timings. Unoptimized Rust is slow at the CPU reference loops,
so debug builds make the CPU look about 40× slower than it is.

The program prints which GPU adapter it picked. It asks for the high-performance
GPU, so on a machine with integrated and discrete graphics it uses the discrete
card if that card's driver is working.

## Example results

On a Radeon RX 560, release build, MLP demo (64 → 128 → 128 → 10, batch 32):

| | time |
|---|---|
| CPU forward | 0.89 ms |
| GPU, first call (includes one-time setup) | 4.37 ms |
| GPU, later calls (upload + 3 layers + softmax + readback), average of 200 | 1.51 ms |
| GPU, the layers on the device | 0.18 ms |

The GPU does the layer math in a fraction of a millisecond. The ~1.3 ms around it
is the fixed cost of each call: uploading, submitting, waiting and reading back.
At this small size, that makes the CPU faster overall. The GPU pulls ahead as the
batch or the layers grow, because that fixed cost stays the same while the work
increases.

## Measuring kernel changes

GPU clocks change between runs, so separate `bench` runs of the same code can
disagree by up to 2×. That's too noisy to judge a kernel tweak.
`cargo run --release -- swizzle` measures the swizzled kernel against the plain
one properly:

- **Paired rounds:** each round runs both kernels back to back on the same
  device and records their ratio, so both see the same clock state.
- **Alternating order:** plain first in one round, swizzled first in the next.
- **Volume:** 60 rounds per size after 5 warm-up rounds, timed on the GPU.
- **A/A control:** the plain kernel against itself, with the same procedure.
  Its ratio shows the noise floor. A difference only counts if it's larger
  than that and its middle 50% is entirely on one side of 1.0.

Result on a Radeon RX 560 (ratio < 1 would mean the swizzle is faster):

| N | swizzle / plain [25th–75th percentile] | A/A control | verdict |
|---|---|---|---|
| 64 | 1.176 [1.100–1.204] | 1.004 [0.993–1.022] | 18% slower |
| 128 | 1.105 [1.084–1.177] | 0.997 [0.992–1.003] | 10% slower |
| 256 | 1.090 [1.085–1.098] | 0.999 [0.994–1.003] | 9% slower |
| 512 | 1.249 [1.212–1.281] | 1.000 [0.999–1.002] | 25% slower |

To test another kernel change the same way, add it as a `PipelineMode` and
call `paired()` in `src/bench.rs` with the two modes.

## Project layout

```
src/
  main.rs               entry point, the 4x4 demo, CPU references, GPU tests
  matrix_multiplier.rs  GPU engine: device, pipelines, run_pass(), multiply/add/madd
  mlp.rs                MLP types, validation, Mlp::new/forward, cpu_forward, mlp demo
  bench.rs              naive vs tiled multiply benchmark
  shader.wgsl           naive multiply (+ optional D)
  shader_tiled.wgsl     tiled multiply (+ optional D, + optional XOR swizzle)
  shader_add.wgsl       element-wise add
  shader_dense.wgsl     tiled dense layer: X·W + bias, then none, ReLU or sigmoid
  shader_softmax.wgsl   row-wise softmax in place, one workgroup per row
```

The multiply and add shaders share one bind-group layout: a uniform with the
dimensions, inputs A and B, output C, and an optional fourth matrix D. The dense
and softmax shaders use the two-group layout described under "Dual bind groups".

The 16×16 workgroup size is defined once in Rust (`WORKGROUP_SIZE`) and injected
into every shader as `BLOCK_SIZE`. It can't go above 16 with default GPU limits,
which allow 256 threads per workgroup.

## Limits and notes

- **Matrix size:** every dimension is limited to 4096 (`MAX_DIMENSION`), so one
  f32 matrix is at most 64 MB. That fits the default 128 MB storage-buffer limit.
- **Long dispatches:** a large multiply, especially with the naive kernel on an
  integrated GPU, can run long enough to trigger the Windows GPU timeout (TDR).
  The benchmark stops at 512×512 for this reason.
- **Precision:** the GPU sums dot products in a different order than the CPU,
  so results are compared with a relative tolerance, not for exact equality.
  Sigmoid and softmax outputs use a tighter check suited to values in [0, 1].
- **One GPU device at a time:** the GPU tests share a lock (`GPU_LOCK`), so only
  one wgpu device exists at once even though `cargo test` runs tests in parallel.
  The test machine blue-screened during some parallel runs. The crash logs
  pointed at a Windows file-system driver, not the GPU driver, so the cause is
  unconfirmed; the lock is a precaution.
- **Kernel timing** needs the `TIMESTAMP_QUERY` feature. Without it,
  `kernel_time` is `None` and only wall-clock times are shown.
