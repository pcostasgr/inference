mod matrix_multiplier;
use matrix_multiplier::{MatrixMultiplier, PipelineMode, MAX_DIMENSION};

fn cpu_matrix_multiply(a: &[f32], b: &[f32], m: usize, k_dim: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k_dim {
                sum += a[i * k_dim + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Compares with a tolerance relative to each value's magnitude, since GPU and CPU
/// accumulate in different orders and f32 error grows with K and value size.
fn approx_eq(gpu: &[f32], cpu: &[f32]) -> bool {
    gpu.len() == cpu.len()
        && gpu
            .iter()
            .zip(cpu)
            .all(|(g, c)| (g - c).abs() <= 1e-4 * c.abs().max(1.0))
}

fn print_matrix(mat: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            print!("{:8.2} ", mat[r * cols + c]);
        }
        println!();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Matrix Size Limit: Max dimension = {MAX_DIMENSION}");

    let selected_mode = PipelineMode::Tiled;
    println!("Initializing wgpu with mode: {selected_mode:?}...");
    let engine = pollster::block_on(MatrixMultiplier::new(selected_mode))?;
    println!("Using GPU adapter: {:?}", engine.adapter_info().name);

    let m = 4usize;
    let k = 4usize;
    let n = 4usize;

    #[rustfmt::skip]
    let matrix_a: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
    ];

    #[rustfmt::skip]
    let matrix_b: Vec<f32> = vec![
        7.0, 6.0, 5.0, 4.0,
        3.0, 2.0, 1.0, 0.0,
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ];

    let cpu_expected = cpu_matrix_multiply(&matrix_a, &matrix_b, m, k, n);
    let gpu_result = pollster::block_on(engine.multiply(&matrix_a, &matrix_b, m, k, n))?;

    println!("\n--- Matrix A ({m}x{k}) ---");
    print_matrix(&matrix_a, m, k);

    println!("\n--- Matrix B ({k}x{n}) ---");
    print_matrix(&matrix_b, k, n);

    println!("\n--- GPU Result Matrix C (Mode: {selected_mode:?}) ---");
    print_matrix(&gpu_result, m, n);

    println!("\n--- CPU Expected Matrix C ---");
    print_matrix(&cpu_expected, m, n);

    if !approx_eq(&gpu_result, &cpu_expected) {
        return Err("Computation failed verification: GPU result differs from CPU".into());
    }
    println!("\nSUCCESS: GPU computation matches CPU calculation!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in [-1, 1) so failures are reproducible.
    fn test_matrix(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Shapes chosen to hit multiple tiles and partial edge tiles on every axis.
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (4, 4, 4),
        (16, 16, 16),
        (17, 17, 17),
        (33, 17, 45),
        (1, 100, 1),
        (100, 1, 100),
        (64, 130, 31),
    ];

    fn check_mode(mode: PipelineMode) {
        let engine = pollster::block_on(MatrixMultiplier::new(mode)).expect("GPU init failed");
        for (i, &(m, k, n)) in SHAPES.iter().enumerate() {
            let a = test_matrix(m * k, 2 * i as u32 + 1);
            let b = test_matrix(k * n, 2 * i as u32 + 2);
            let expected = cpu_matrix_multiply(&a, &b, m, k, n);
            let got = pollster::block_on(engine.multiply(&a, &b, m, k, n))
                .unwrap_or_else(|e| panic!("{mode:?} {m}x{k}x{n}: {e}"));
            assert!(approx_eq(&got, &expected), "{mode:?} mismatch for {m}x{k}x{n}");
        }
    }

    #[test]
    fn naive_matches_cpu() {
        check_mode(PipelineMode::Naive);
    }

    #[test]
    fn tiled_matches_cpu() {
        check_mode(PipelineMode::Tiled);
    }

    #[test]
    fn rejects_bad_input() {
        let engine =
            pollster::block_on(MatrixMultiplier::new(PipelineMode::Tiled)).expect("GPU init failed");
        let one = [1.0f32];
        assert!(pollster::block_on(engine.multiply(&one, &one, 0, 1, 1)).is_err());
        assert!(pollster::block_on(engine.multiply(&one, &one, 2, 1, 1)).is_err());
        let too_big = MAX_DIMENSION + 1;
        assert!(pollster::block_on(engine.multiply(&one, &one, too_big, 1, 1)).is_err());
    }
}
