use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, RmsNorm};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn run_layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor) {
    let _ = LayerNorm::new(weight.clone(), bias.clone(), 1e-5).forward(input);
}

fn run_layer_norm_2d(input: &Tensor, weight: &Tensor, bias: &Tensor) {
    let _ = candle_nn::ops::layer_norm_2d(input, weight, bias, 1e-6);
}

fn run_layer_norm_2d_tensor_ops(input: &Tensor, weight: &Tensor, bias: &Tensor) {
    let channels = weight.dims1().unwrap();
    let mean = input.mean_keepdim(1).unwrap();
    let centered = input.broadcast_sub(&mean).unwrap();
    let var = centered.sqr().unwrap().mean_keepdim(1).unwrap();
    let normed = centered
        .broadcast_div(&(var + 1e-6).unwrap().sqrt().unwrap())
        .unwrap();
    let normed = normed
        .broadcast_mul(&weight.reshape((1, channels, 1, 1)).unwrap())
        .unwrap();
    let _ = normed.broadcast_add(&bias.reshape((1, channels, 1, 1)).unwrap());
}

fn run_rms_norm(input: &Tensor, weight: &Tensor) {
    let _ = RmsNorm::new(weight.clone(), 1e-5).forward(input);
}

const B: usize = 1;
const M: usize = 1024;
const K: usize = 1024;

fn run_layer_norm_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let elements = B * M * K;

    let weight = Tensor::arange(0.0, elements as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = weight.ones_like().unwrap();
    let input = weight.ones_like().unwrap();

    let flops = elements * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_layer_norm(black_box(&input), black_box(&weight), black_box(&bias));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_layer_norm_2d_benchmark(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    channels: usize,
    height: usize,
    width: usize,
    name: &str,
) {
    let input = Tensor::ones((1, channels, height, width), dtype, device).unwrap();
    let weight = Tensor::ones(channels, dtype, device).unwrap();
    let bias = Tensor::zeros(channels, dtype, device).unwrap();

    let bytes = input.elem_count() * dtype.size_in_bytes();
    let mut group =
        c.benchmark_group(device.bench_name(format!("sam3_layer_norm_2d_{dtype:?}_{name}")));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("fused", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_layer_norm_2d(black_box(&input), black_box(&weight), black_box(&bias));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.bench_function("tensor_ops", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_layer_norm_2d_tensor_ops(
                    black_box(&input),
                    black_box(&weight),
                    black_box(&bias),
                );
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_rms_norm_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let elements = B * M * K;

    let weight = Tensor::arange(0.0, elements as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let input = weight.ones_like().unwrap();

    let flops = elements * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_rms_norm(black_box(&input), black_box(&weight));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    let sam3_layer_norm_2d_only =
        std::env::var_os("CANDLE_NN_BENCH_SAM3_LAYER_NORM_2D_ONLY").is_some();
    for d in device.devices {
        if !sam3_layer_norm_2d_only {
            run_rms_norm_benchmark(c, &d, DType::F32, "rms_norm_f32");
            run_rms_norm_benchmark(c, &d, DType::BF16, "rms_norm_bf16");
            run_rms_norm_benchmark(c, &d, DType::F16, "rms_norm_f16");
            run_layer_norm_benchmark(c, &d, DType::F32, "layer_norm_f32");
            run_layer_norm_benchmark(c, &d, DType::BF16, "layer_norm_bf16");
            run_layer_norm_benchmark(c, &d, DType::F16, "layer_norm_f16");
        }
        if !matches!(&d, Device::Metal(_)) {
            run_layer_norm_2d_benchmark(c, &d, DType::F32, 4, 128, 128, "downsample_4x128x128");
            run_layer_norm_2d_benchmark(c, &d, DType::F32, 16, 64, 64, "downsample_16x64x64");
            run_layer_norm_2d_benchmark(c, &d, DType::F32, 64, 32, 32, "downsample_64x32x32");
            run_layer_norm_2d_benchmark(c, &d, DType::F32, 256, 16, 16, "downsample_256x16x16");
        }
    }
}

criterion_group!(benches, criterion_benchmark);
