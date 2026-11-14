mod bench;
mod matrix_multiplier;
mod mlp;
use matrix_multiplier::{MatrixMultiplier, PipelineMode, MAX_DIMENSION};
use std::time::{Duration, Instant};

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

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

fn cpu_matrix_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
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
    match std::env::args().nth(1).as_deref() {
        Some("bench") => return bench::run(),
        Some("mlp") => return mlp::run_demo(),
        _ => {}
    }

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

    let cpu_start = Instant::now();
    let cpu_expected = cpu_matrix_multiply(&matrix_a, &matrix_b, m, k, n);
    let cpu_time = cpu_start.elapsed();

    let gpu_start = Instant::now();
    let gpu_run = pollster::block_on(engine.multiply_timed(&matrix_a, &matrix_b, m, k, n))?;
    let gpu_wall = gpu_start.elapsed();
    let gpu_result = gpu_run.output;

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

    println!("\n--- Timing ---");
    println!("CPU multiply:           {:>10.3} ms", ms(cpu_time));
    println!("GPU total (wall):       {:>10.3} ms  (upload + dispatch + readback)", ms(gpu_wall));
    match gpu_run.kernel_time {
        Some(t) => println!("GPU kernel (on-device): {:>10.3} ms", ms(t)),
        None => println!("GPU kernel (on-device):        n/a  (adapter lacks TIMESTAMP_QUERY)"),
    }

    // --- Element-wise addition: C = A + B (both m x n here, since A and B are square) ---
    let cpu_add_start = Instant::now();
    let cpu_sum = cpu_matrix_add(&matrix_a, &matrix_b);
    let cpu_add_time = cpu_add_start.elapsed();

    let gpu_add_start = Instant::now();
    let add_run = pollster::block_on(engine.add_timed(&matrix_a, &matrix_b, m, n))?;
    let gpu_add_wall = gpu_add_start.elapsed();

    println!("\n--- GPU Result A + B ---");
    print_matrix(&add_run.output, m, n);

    if !approx_eq(&add_run.output, &cpu_sum) {
        return Err("Addition failed verification: GPU result differs from CPU".into());
    }
    println!("\nSUCCESS: GPU addition matches CPU calculation!");

    println!("\n--- Addition Timing ---");
    println!("CPU add:                {:>10.3} ms", ms(cpu_add_time));
    println!("GPU total (wall):       {:>10.3} ms  (upload + dispatch + readback)", ms(gpu_add_wall));
    match add_run.kernel_time {
        Some(t) => println!("GPU kernel (on-device): {:>10.3} ms", ms(t)),
        None => println!("GPU kernel (on-device):        n/a  (adapter lacks TIMESTAMP_QUERY)"),
    }


    // --- Fused multiply-add: C = A * B + D, in one kernel ---
    let matrix_d: Vec<f32> = (0..m * n).map(|i| i as f32).collect();

    let cpu_madd_start = Instant::now();
    let cpu_madd = cpu_matrix_add(&cpu_matrix_multiply(&matrix_a, &matrix_b, m, k, n), &matrix_d);
    let cpu_madd_time = cpu_madd_start.elapsed();

    let gpu_madd_start = Instant::now();
    let madd_run =
        pollster::block_on(engine.multiply_add_timed(&matrix_a, &matrix_b, &matrix_d, m, k, n))?;
    let gpu_madd_wall = gpu_madd_start.elapsed();

    println!("\n--- Matrix D ({m}x{n}) ---");
    print_matrix(&matrix_d, m, n);

    println!("\n--- GPU Result A * B + D (fused) ---");
    print_matrix(&madd_run.output, m, n);

    if !approx_eq(&madd_run.output, &cpu_madd) {
        return Err("Multiply-add failed verification: GPU result differs from CPU".into());
    }
    println!("\nSUCCESS: GPU multiply-add matches CPU calculation!");

    println!("\n--- Multiply-Add Timing ---");
    println!("CPU multiply-add:       {:>10.3} ms", ms(cpu_madd_time));
    println!("GPU total (wall):       {:>10.3} ms  (upload + dispatch + readback)", ms(gpu_madd_wall));
    match madd_run.kernel_time {
        Some(t) => println!("GPU kernel (on-device): {:>10.3} ms", ms(t)),
        None => println!("GPU kernel (on-device):        n/a  (adapter lacks TIMESTAMP_QUERY)"),
    }

    println!("\nRun `cargo run --release -- bench` to time larger matrices.");

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

    /// Only one GPU device may exist at a time: opening several at once
    /// (cargo runs tests in parallel threads) crashes the Intel UHD 630 driver.
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Runs `f` with a fresh engine while holding [`GPU_LOCK`]; the engine
    /// (and its device) is dropped before the lock is released.
    fn with_engine(mode: PipelineMode, f: impl FnOnce(&MatrixMultiplier)) {
        // A panicking test poisons the lock; the GPU is still usable afterwards.
        let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let engine = pollster::block_on(MatrixMultiplier::new(mode)).expect("GPU init failed");
        f(&engine);
    }

    fn check_mode(mode: PipelineMode) {
        with_engine(mode, |engine| {
            for (i, &(m, k, n)) in SHAPES.iter().enumerate() {
                let a = test_matrix(m * k, 2 * i as u32 + 1);
                let b = test_matrix(k * n, 2 * i as u32 + 2);
                let expected = cpu_matrix_multiply(&a, &b, m, k, n);
                let got = pollster::block_on(engine.multiply(&a, &b, m, k, n))
                    .unwrap_or_else(|e| panic!("{mode:?} {m}x{k}x{n}: {e}"));
                assert!(approx_eq(&got, &expected), "{mode:?} mismatch for {m}x{k}x{n}");
            }
        });
    }

    #[test]
    fn naive_matches_cpu() {
        check_mode(PipelineMode::Naive);
    }

    #[test]
    fn tiled_matches_cpu() {
        check_mode(PipelineMode::Tiled);
    }

    fn check_madd(mode: PipelineMode) {
        with_engine(mode, |engine| {
            for (i, &(m, k, n)) in SHAPES.iter().enumerate() {
                let a = test_matrix(m * k, 200 + 3 * i as u32);
                let b = test_matrix(k * n, 201 + 3 * i as u32);
                let d = test_matrix(m * n, 202 + 3 * i as u32);
                let expected = cpu_matrix_add(&cpu_matrix_multiply(&a, &b, m, k, n), &d);
                let got = pollster::block_on(engine.multiply_add(&a, &b, &d, m, k, n))
                    .unwrap_or_else(|e| panic!("{mode:?} madd {m}x{k}x{n}: {e}"));
                assert!(approx_eq(&got, &expected), "{mode:?} madd mismatch for {m}x{k}x{n}");
            }
        });
    }

    #[test]
    fn naive_madd_matches_cpu() {
        check_madd(PipelineMode::Naive);
    }

    #[test]
    fn tiled_madd_matches_cpu() {
        check_madd(PipelineMode::Tiled);
    }

    #[test]
    fn madd_rejects_wrong_d_len() {
        with_engine(PipelineMode::Tiled, |engine| {
            let a = [1.0f32; 6]; // 2x3
            let b = [1.0f32; 12]; // 3x4
            let d_bad = [0.0f32; 7]; // should be 2x4 = 8
            assert!(pollster::block_on(engine.multiply_add(&a, &b, &d_bad, 2, 3, 4)).is_err());
        });
    }

    fn dense(inputs: usize, outputs: usize, seed: u32) -> mlp::DenseLayer {
        mlp::DenseLayer {
            weights: test_matrix(inputs * outputs, seed),
            bias: test_matrix(outputs, seed + 1),
            inputs,
            outputs,
            activation: mlp::Activation::Relu,
        }
    }

    #[test]
    fn mlp_uploads_weights_once() {
        with_engine(PipelineMode::Tiled, |engine| {
            let layers = [dense(64, 128, 1), dense(128, 33, 3), dense(33, 10, 5)];
            let net = pollster::block_on(mlp::Mlp::new(engine, &layers)).expect("upload failed");

            assert_eq!(net.num_layers(), 3);
            assert_eq!(net.input_size(), 64);
            assert_eq!(net.output_size(), 10);
            assert_eq!(net.max_width(), 128);

            let floats: usize = layers.iter().map(|l| l.weights.len() + l.bias.len()).sum();
            assert_eq!(net.gpu_param_bytes(), (floats * 4) as u64);
        });
    }

    #[test]
    fn mlp_rejects_invalid_layers() {
        with_engine(PipelineMode::Tiled, |engine| {
            let mismatched = [dense(8, 16, 1), dense(15, 4, 3)];
            assert!(pollster::block_on(mlp::Mlp::new(engine, &mismatched)).is_err());
        });
    }

    fn identity_layer(width: usize) -> mlp::DenseLayer {
        let mut weights = vec![0.0; width * width];
        for i in 0..width {
            weights[i * width + i] = 1.0;
        }
        mlp::DenseLayer {
            weights,
            bias: vec![0.0; width],
            inputs: width,
            outputs: width,
            activation: mlp::Activation::None,
        }
    }

    /// Identity layers must pass the input through unchanged, whether the
    /// ping-pong ends in pong (odd layer count) or ping (even).
    #[test]
    fn mlp_forward_identity_chain_any_length() {
        with_engine(PipelineMode::Tiled, |engine| {
            let (width, batch) = (5, 3);
            let input = test_matrix(batch * width, 42);
            for n_layers in 1..=4 {
                let layers = vec![identity_layer(width); n_layers];
                let net = pollster::block_on(mlp::Mlp::new(engine, &layers)).unwrap();
                let out = pollster::block_on(net.forward(&input, batch)).unwrap().output;
                assert_eq!(out, input, "identity chain of {n_layers} layers changed the input");
            }
        });
    }

    /// Small 2 -> 2 (ReLU) -> 3 network worked out by hand, batch of 2.
    #[test]
    fn mlp_forward_hand_computed() {
        with_engine(PipelineMode::Tiled, |engine| {
            #[rustfmt::skip]
            let layers = [
                mlp::DenseLayer {
                    weights: vec![1.0, -1.0,
                                  2.0,  1.0],
                    bias: vec![-6.0, 0.5],
                    inputs: 2, outputs: 2,
                    activation: mlp::Activation::Relu,
                },
                mlp::DenseLayer {
                    weights: vec![1.0, 0.0, 2.0,
                                  1.0, 1.0, 0.0],
                    bias: vec![0.0, 0.0, 0.0],
                    inputs: 2, outputs: 3,
                    activation: mlp::Activation::None,
                },
            ];
            let net = pollster::block_on(mlp::Mlp::new(engine, &layers)).unwrap();

            // Row 0: [1,2]·W1 = [5,1]; +bias = [-1,1.5]; ReLU = [0,1.5]; ·W2 = [1.5,1.5,0]
            // Row 1: [0,0]·W1 = [0,0]; +bias = [-6,0.5]; ReLU = [0,0.5]; ·W2 = [0.5,0.5,0]
            let input = [1.0, 2.0, 0.0, 0.0];
            let run = pollster::block_on(net.forward(&input, 2)).unwrap();
            assert_eq!(run.output, vec![1.5, 1.5, 0.0, 0.5, 0.5, 0.0]);

            // Weights stay on the GPU: a second call gives the same answer.
            let again = pollster::block_on(net.forward(&input, 2)).unwrap();
            assert_eq!(again.output, run.output);
        });
    }

    /// Random network with the given widths: ReLU on hidden layers, none on the
    /// output. Weights are scaled by 1/sqrt(inputs) so activations stay around 1
    /// through the chain instead of growing layer by layer.
    fn random_net(widths: &[usize], seed: u32, hidden: mlp::Activation) -> Vec<mlp::DenseLayer> {
        let last = widths.len() - 2;
        widths
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                let (inputs, outputs) = (w[0], w[1]);
                let scale = 1.0 / (inputs as f32).sqrt();
                let s = seed + 10 * i as u32;
                mlp::DenseLayer {
                    weights: test_matrix(inputs * outputs, s).iter().map(|v| v * scale).collect(),
                    bias: test_matrix(outputs, s + 1),
                    inputs,
                    outputs,
                    activation: if i == last { mlp::Activation::None } else { hidden },
                }
            })
            .collect()
    }

    /// (batch, layer widths): widths and batches off the 16-wide tile grid, chains
    /// of 1–4 layers (both ping-pong endings), widening and narrowing layers.
    const MLP_SHAPES: &[(usize, &[usize])] = &[
        (1, &[3, 1]),
        (16, &[16, 16]),
        (33, &[64, 128, 10]),
        (17, &[100, 33, 47, 5]),
        (5, &[7, 130, 1, 20, 9]),
    ];

    /// Every shape, once with ReLU hidden layers and once with sigmoid.
    #[test]
    fn mlp_forward_matches_cpu() {
        with_engine(PipelineMode::Tiled, |engine| {
            for hidden in [mlp::Activation::Relu, mlp::Activation::Sigmoid] {
                for (i, &(batch, widths)) in MLP_SHAPES.iter().enumerate() {
                    let layers = random_net(widths, 300 + 50 * i as u32, hidden);
                    let input = test_matrix(batch * widths[0], 900 + i as u32);

                    let expected = mlp::cpu_forward(&layers, &input, batch).unwrap();
                    let net = pollster::block_on(mlp::Mlp::new(engine, &layers)).unwrap();
                    let got = pollster::block_on(net.forward(&input, batch))
                        .unwrap_or_else(|e| panic!("{hidden:?} {widths:?} batch {batch}: {e}"))
                        .output;

                    assert!(
                        approx_eq(&got, &expected),
                        "GPU/CPU mismatch for {hidden:?} hidden layers, widths {widths:?}, batch {batch}"
                    );
                }
            }
        });
    }

    /// One 1 -> 2 sigmoid layer, so each output is sigmoid(x - 2) and sigmoid(2 - x):
    /// exact midpoint, ordinary values, and inputs far past the clamp.
    #[test]
    fn mlp_forward_sigmoid_values() {
        with_engine(PipelineMode::Tiled, |engine| {
            let layers = [mlp::DenseLayer {
                weights: vec![1.0, -1.0],
                bias: vec![-2.0, 2.0],
                inputs: 1,
                outputs: 2,
                activation: mlp::Activation::Sigmoid,
            }];
            let net = pollster::block_on(mlp::Mlp::new(engine, &layers)).unwrap();

            // x = 2 -> pre-activations [0, 0]; x = 0 -> [-2, 2];
            // x = +-1000 -> far beyond the +-30 clamp (saturates, no NaN).
            let input = [2.0, 0.0, 1000.0, -1000.0];
            let out = pollster::block_on(net.forward(&input, 4)).unwrap().output;

            assert_eq!(&out[0..2], &[0.5, 0.5], "sigmoid(0) must be exactly 0.5");
            assert!(out.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)), "{out:?}");
            assert!(approx_eq(&out, &mlp::cpu_forward(&layers, &input, 4).unwrap()), "{out:?}");
            assert!(out[4] > 0.999_999 && out[5] < 1e-6, "large x must saturate: {out:?}");
            assert!(out[6] < 1e-6 && out[7] > 0.999_999, "small x must saturate: {out:?}");
        });
    }

    #[test]
    fn mlp_forward_rejects_bad_input() {
        with_engine(PipelineMode::Tiled, |engine| {
            let net = pollster::block_on(mlp::Mlp::new(engine, &[dense(4, 2, 1)])).unwrap();
            assert!(pollster::block_on(net.forward(&[1.0; 4], 0)).is_err());
            assert!(pollster::block_on(net.forward(&[1.0; 7], 2)).is_err());
            assert!(pollster::block_on(net.forward(&[1.0; 4], MAX_DIMENSION + 1)).is_err());
        });
    }

    /// Addition shapes: single element, exact tile, partial tiles, thin rows/columns.
    const ADD_SHAPES: &[(usize, usize)] = &[(1, 1), (16, 16), (17, 33), (1, 300), (300, 1), (100, 70)];

    #[test]
    fn add_matches_cpu() {
        with_engine(PipelineMode::Tiled, |engine| {
            for (i, &(rows, cols)) in ADD_SHAPES.iter().enumerate() {
                let a = test_matrix(rows * cols, 100 + 2 * i as u32);
                let b = test_matrix(rows * cols, 101 + 2 * i as u32);
                let got = pollster::block_on(engine.add(&a, &b, rows, cols))
                    .unwrap_or_else(|e| panic!("add {rows}x{cols}: {e}"));
                // Each element is a single f32 add, so GPU and CPU must agree exactly.
                assert_eq!(got, cpu_matrix_add(&a, &b), "add mismatch for {rows}x{cols}");
            }
        });
    }

    #[test]
    fn add_rejects_bad_input() {
        with_engine(PipelineMode::Tiled, |engine| {
            let four = [1.0f32; 4];
            let three = [1.0f32; 3];
            assert!(pollster::block_on(engine.add(&four, &four, 0, 4)).is_err());
            assert!(pollster::block_on(engine.add(&four, &three, 2, 2)).is_err());
            assert!(pollster::block_on(engine.add(&four, &four, MAX_DIMENSION + 1, 1)).is_err());
        });
    }

    #[test]
    fn rejects_bad_input() {
        with_engine(PipelineMode::Tiled, |engine| {
            let one = [1.0f32];
            assert!(pollster::block_on(engine.multiply(&one, &one, 0, 1, 1)).is_err());
            assert!(pollster::block_on(engine.multiply(&one, &one, 2, 1, 1)).is_err());
            let too_big = MAX_DIMENSION + 1;
            assert!(pollster::block_on(engine.multiply(&one, &one, too_big, 1, 1)).is_err());
        });
    }
}
