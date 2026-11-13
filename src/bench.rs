//! Simple timing benchmark: `cargo run --release -- bench`.
//!
//! Reports, per pipeline mode and square size N:
//! - wall: end-to-end `multiply` call (upload + dispatch + readback), host clock
//! - kernel: compute pass only, measured on the GPU with timestamp queries
//! - GFLOP/s: 2*N^3 / kernel time (falls back to wall time without timestamps)

use crate::matrix_multiplier::{MatrixMultiplier, PipelineMode};
use std::time::{Duration, Instant};

// Kept small on purpose: 1024+ (especially the naive kernel) can run long enough
// on an integrated GPU to trigger a driver reset / system hang.
const SIZES: &[usize] = &[64, 128, 256, 512];
const WARMUP: usize = 1;
const ITERS: usize = 5;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        println!("note: debug build; use `cargo run --release -- bench` for meaningful host timings\n");
    }

    for mode in [PipelineMode::Naive, PipelineMode::Tiled] {
        let engine = pollster::block_on(MatrixMultiplier::new(mode))?;
        println!(
            "== {mode:?} on {} (kernel timer: {}) ==",
            engine.adapter_info().name,
            if engine.has_kernel_timer() { "yes" } else { "unsupported" }
        );
        println!(
            "{:>6} {:>12} {:>12} {:>10}",
            "N", "wall ms", "kernel ms", "GFLOP/s"
        );

        for &n in SIZES {
            let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.5).collect();
            let b: Vec<f32> = (0..n * n).map(|i| (i % 5) as f32 * 0.25).collect();

            let mut walls = Vec::with_capacity(ITERS);
            let mut kernels = Vec::with_capacity(ITERS);
            for i in 0..WARMUP + ITERS {
                let start = Instant::now();
                let run = pollster::block_on(engine.multiply_timed(&a, &b, n, n, n))?;
                let wall = start.elapsed();
                if i >= WARMUP {
                    walls.push(wall);
                    kernels.extend(run.kernel_time);
                }
            }

            let wall = median(walls);
            let kernel = (!kernels.is_empty()).then(|| median(kernels));
            let flops = 2.0 * (n as f64).powi(3);
            let gflops = flops / kernel.unwrap_or(wall).as_secs_f64() / 1e9;

            println!(
                "{:>6} {:>12.3} {:>12} {:>10.1}",
                n,
                ms(wall),
                kernel.map_or("-".into(), |k| format!("{:.3}", ms(k))),
                gflops
            );
        }
        println!();
    }

    println!("(median of {ITERS} runs after {WARMUP} warmup)");
    Ok(())
}
