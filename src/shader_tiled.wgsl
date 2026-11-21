// BLOCK_SIZE is injected by the host (WORKGROUP_SIZE in matrix_multiplier.rs),
// and SWIZZLE (see swz() below).

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

// XOR swizzle: logical element (r, c) of a tile is stored in column c ^ r.
//
// Shared memory is split into banks; threads that hit different addresses in
// the same bank at the same time are serialized (a bank conflict). With a
// plain [r][c] layout, a column of a tile (fixed c, varying r) sits in rows
// BLOCK_SIZE floats apart, which can land on the same bank. XORing the column
// with the row spreads each column across all columns of the tile, so a
// column read touches BLOCK_SIZE different banks.
//
// For a fixed row, c -> c ^ r is a permutation of 0..BLOCK_SIZE (BLOCK_SIZE is
// a power of two, checked on the Rust side), so every element still has
// exactly one slot. Every store and load must go through swz() so they agree.
//
// SWIZZLE is injected by the host (PipelineMode::Tiled = false,
// PipelineMode::TiledSwizzled = true), so the two variants can be compared.
// It's a constant, so the compiler drops the unused branch.
//
// Measured result (Radeon RX 560, `cargo run --release -- swizzle`): the
// swizzle is 9-25% slower. This kernel's reads were already conflict-free
// (a row of threads reads one shared element of tileA, and reads across a
// row of tileB), so the swizzle only adds an XOR and address math to every
// multiply-add in the inner loop.
fn swz(r: u32, c: u32) -> u32 {
    if (SWIZZLE) {
        return c ^ (r & (BLOCK_SIZE - 1u));
    }
    return c;
}

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

        // Each thread stores tile element (local_id.y, local_id.x), at its
        // swizzled column.
        let store_col = swz(local_id.y, local_id.x);

        // Collaboratively load Tile A into shared workgroup memory
        if (row < dims.M && tiled_col_A < dims.K) {
            tileA[local_id.y][store_col] = matrixA[row * dims.K + tiled_col_A];
        } else {
            tileA[local_id.y][store_col] = 0.0;
        }

        // Collaboratively load Tile B into shared workgroup memory
        if (tiled_row_B < dims.K && col < dims.N) {
            tileB[local_id.y][store_col] = matrixB[tiled_row_B * dims.N + col];
        } else {
            tileB[local_id.y][store_col] = 0.0;
        }

        // Synchronize: wait until all threads in workgroup have populated their tile slots
        workgroupBarrier();

        // Accumulate dot products from fast on-chip shared tiles:
        // A(local_id.y, k) * B(k, local_id.x), both read through the swizzle.
        for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
            acc = acc + tileA[local_id.y][swz(local_id.y, k)] * tileB[k][swz(k, local_id.x)];
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
