//! Timing benchmarks.
//!
//! `cargo run --release -- bench`: every multiply kernel at each square size N.
//! - wall: end-to-end `multiply` call (upload + dispatch + readback), host clock
//! - kernel: compute pass only, measured on the GPU with timestamp queries
//! - GFLOP/s: 2*N^3 / kernel time (falls back to wall time without timestamps)
//!
//! `cargo run --release -- swizzle`: A/B test of the XOR-swizzled tiled kernel
//! against the plain one; see [`run_swizzle_ab`].

use crate::matrix_multiplier::{MatrixMultiplier, PipelineMode};
use std::time::{Duration, Instant};

type BoxError = Box<dyn std::error::Error>;

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

/// Deterministic N x N test matrices.
fn square_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..n * n).map(|i| (i % 7) as f32 * 0.5).collect();
    let b = (0..n * n).map(|i| (i % 5) as f32 * 0.25).collect();
    (a, b)
}

pub fn run() -> Result<(), BoxError> {
    if cfg!(debug_assertions) {
        println!("note: debug build; use `cargo run --release -- bench` for meaningful host timings\n");
    }

    // One engine (one device) for every kernel: set_mode switches between them.
    let mut engine = pollster::block_on(MatrixMultiplier::new(PipelineMode::Naive))?;
    let timer = if engine.has_kernel_timer() { "yes" } else { "unsupported" };

    for mode in PipelineMode::ALL {
        engine.set_mode(mode);
        println!("== {mode:?} on {} (kernel timer: {timer}) ==", engine.adapter_info().name);
        println!("{:>6} {:>12} {:>12} {:>10}", "N", "wall ms", "kernel ms", "GFLOP/s");

        for &n in SIZES {
            let (a, b) = square_inputs(n);

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
    println!("note: GPU clocks vary between runs; for a fair kernel comparison use `-- swizzle`");
    Ok(())
}

// ---------------------------------------------------------------------------
// A/B test: XOR-swizzled vs plain tiled kernel
// ---------------------------------------------------------------------------

const AB_ROUNDS: usize = 60;
const AB_WARMUP_ROUNDS: usize = 5;

/// Value at fraction `p` (0..=1) of an already-sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// Median and interquartile range (25th..75th percentile) of `v`.
struct Spread {
    p25: f64,
    median: f64,
    p75: f64,
}

fn spread(mut v: Vec<f64>) -> Spread {
    v.sort_by(f64::total_cmp);
    Spread {
        p25: percentile(&v, 0.25),
        median: percentile(&v, 0.5),
        p75: percentile(&v, 0.75),
    }
}

/// Paired measurement of kernel `x` vs kernel `y` on N x N inputs.
///
/// Each round runs both kernels back to back and records `y / x` for that
/// round. Kernels that run right after each other share the GPU's clock and
/// power state, so the per-round ratio cancels the slow drift that makes
/// separate benchmark runs disagree by up to 2x. The order alternates every
/// round (x then y, then y then x) so neither kernel always runs second.
///
/// Returns (x kernel ms, y kernel ms, per-round y/x ratio), warm-up excluded.
fn paired(
    engine: &mut MatrixMultiplier,
    a: &[f32],
    b: &[f32],
    n: usize,
    x: PipelineMode,
    y: PipelineMode,
) -> Result<(Spread, Spread, Spread), BoxError> {
    let mut xs = Vec::with_capacity(AB_ROUNDS);
    let mut ys = Vec::with_capacity(AB_ROUNDS);
    let mut ratios = Vec::with_capacity(AB_ROUNDS);

    let time = |engine: &mut MatrixMultiplier, mode: PipelineMode| -> Result<f64, BoxError> {
        engine.set_mode(mode);
        let run = pollster::block_on(engine.multiply_timed(a, b, n, n, n))?;
        let t = run.kernel_time.ok_or("kernel timer needed for the A/B test")?;
        Ok(ms(t))
    };

    for round in 0..AB_WARMUP_ROUNDS + AB_ROUNDS {
        let (tx, ty) = if round % 2 == 0 {
            let tx = time(engine, x)?;
            (tx, time(engine, y)?)
        } else {
            let ty = time(engine, y)?;
            (time(engine, x)?, ty)
        };
        if round >= AB_WARMUP_ROUNDS {
            xs.push(tx);
            ys.push(ty);
            ratios.push(ty / tx);
        }
    }

    Ok((spread(xs), spread(ys), spread(ratios)))
}

/// `cargo run --release -- swizzle`
///
/// For each size, two paired experiments on the same device:
/// - **A/B:** plain tiled vs XOR-swizzled tiled. Ratio = swizzled / plain.
/// - **A/A control:** plain tiled vs plain tiled. The kernels are identical,
///   so any spread in this ratio is pure measurement noise. It shows how big
///   an A/B difference has to be before it means anything.
///
/// A result is called a real difference only if the A/B median is further
/// from 1.0 than the A/A control's interquartile range (and at least 2%),
/// and the A/B interquartile range doesn't include 1.0.
pub fn run_swizzle_ab() -> Result<(), BoxError> {
    if cfg!(debug_assertions) {
        println!("note: debug build; kernel times are still GPU-measured, but use --release\n");
    }

    let mut engine = pollster::block_on(MatrixMultiplier::new(PipelineMode::Tiled))?;
    if !engine.has_kernel_timer() {
        return Err("This GPU doesn't support timestamp queries, which the A/B test needs".into());
    }

    println!("== XOR swizzle A/B on {} ==", engine.adapter_info().name);
    println!(
        "{AB_ROUNDS} paired rounds per size (+{AB_WARMUP_ROUNDS} warm-up), order alternating; kernel times on the GPU\n"
    );
    println!(
        "{:>5} {:>11} {:>11}   {:<27} {:<27} verdict",
        "N", "plain ms", "swizzle ms", "swizzle/plain [p25-p75]", "A/A control [p25-p75]"
    );

    for &n in SIZES {
        let (a, b) = square_inputs(n);
        let (plain, swz, ab) =
            paired(&mut engine, &a, &b, n, PipelineMode::Tiled, PipelineMode::TiledSwizzled)?;
        let (_, _, aa) = paired(&mut engine, &a, &b, n, PipelineMode::Tiled, PipelineMode::Tiled)?;

        // How far from 1.0 the identical-kernel ratio wanders: the noise floor.
        let noise = (aa.p75 - 1.0).abs().max((1.0 - aa.p25).abs()).max(0.02);
        let effect = ab.median - 1.0;
        let iqr_excludes_one = ab.p25 > 1.0 || ab.p75 < 1.0;
        let verdict = if effect.abs() > noise && iqr_excludes_one {
            if effect < 0.0 {
                format!("swizzle {:.0}% faster", -effect * 100.0)
            } else {
                format!("swizzle {:.0}% slower", effect * 100.0)
            }
        } else {
            "no clear difference".to_string()
        };

        println!(
            "{:>5} {:>11.4} {:>11.4}   {:<27} {:<27} {}",
            n,
            plain.median,
            swz.median,
            format!("{:.3} [{:.3}-{:.3}]", ab.median, ab.p25, ab.p75),
            format!("{:.3} [{:.3}-{:.3}]", aa.median, aa.p25, aa.p75),
            verdict
        );
    }

    println!("\nratio < 1 means the swizzled kernel was faster in the same round");
    Ok(())
}
