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

// Workgroup shared memory (on-chip L1/shared scratchpad)
var<workgroup> tileA: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;
var<workgroup> tileB: array<array<f32, BLOCK_SIZE>, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;

    // Total number of 16x16 tiles along the shared K dimension
    let num_tiles = (dims.K + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let tiled_col_A = t * BLOCK_SIZE + local_id.x;
        let tiled_row_B = t * BLOCK_SIZE + local_id.y;

        // Collaboratively load Tile A into shared workgroup memory
        if (row < dims.M && tiled_col_A < dims.K) {
            tileA[local_id.y][local_id.x] = matrixA[row * dims.K + tiled_col_A];
        } else {
            tileA[local_id.y][local_id.x] = 0.0;
        }

        // Collaboratively load Tile B into shared workgroup memory
        if (tiled_row_B < dims.K && col < dims.N) {
            tileB[local_id.y][local_id.x] = matrixB[tiled_row_B * dims.N + col];
        } else {
            tileB[local_id.y][local_id.x] = 0.0;
        }

        // Synchronize: wait until all threads in workgroup have populated their tile slots
        workgroupBarrier();

        // Accumulate dot products from fast on-chip shared tiles
        for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
            acc = acc + tileA[local_id.y][k] * tileB[k][local_id.x];
        }

        // Synchronize: make sure all threads finish computing before loading the next tile
        workgroupBarrier();
    }

    // Write final accumulated result to output matrix C, fusing in D for multiply-add
    if (row < dims.M && col < dims.N) {
        let idx = row * dims.N + col;
        if (dims.has_d != 0u) {
            acc = acc + matrixD[idx];
        }
        matrixC[idx] = acc;
    }
}
