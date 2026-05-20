use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{Device, Result, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

const MAX_AREA: usize = 16;
const HOLE_FILL_LOGIT: f32 = 0.1;
const SPRINKLE_REMOVE_LOGIT: f32 = -0.1;

fn make_cleanup_logits(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    let plane_count = batch * channels;
    let spatial = height * width;
    let mut data = vec![-1f32; plane_count * spatial];

    for plane_idx in 0..plane_count {
        let offset = plane_idx * spatial;
        for row in height / 4..height * 3 / 4 {
            for col in width / 4..width * 3 / 4 {
                data[offset + row * width + col] = 1.0;
            }
        }

        let holes = [
            (height / 2, width / 2),
            (height / 2, width / 2 + 1),
            (height / 2 + 1, width / 2),
            (height / 2 + 1, width / 2 + 1),
        ];
        for (row, col) in holes {
            data[offset + row * width + col] = -1.0;
        }

        let sprinkles = [
            (height / 8, width / 8),
            (height / 8, width / 8 + 1),
            (height * 7 / 8, width * 7 / 8),
        ];
        for (row, col) in sprinkles {
            data[offset + row * width + col] = 1.0;
        }
    }

    Tensor::from_vec(data, (batch, channels, height, width), &Device::Cpu)?.to_device(device)
}

fn cleanup_mask_logits(input: &Tensor) -> Result<Tensor> {
    candle_nn::ops::cleanup_mask_logits_small_components_2d(
        input,
        MAX_AREA,
        HOLE_FILL_LOGIT,
        SPRINKLE_REMOVE_LOGIT,
    )
}

fn cleanup_mask_logits_cpu_roundtrip(input: &Tensor, device: &Device) -> Result<Tensor> {
    let input_cpu = input.to_device(&Device::Cpu)?;
    cleanup_mask_logits(&input_cpu)?.to_device(device)
}

fn run_cleanup_mask_logits_benchmark(
    c: &mut Criterion,
    device: &Device,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    name: &str,
) {
    let input = make_cleanup_logits(batch, channels, height, width, device).unwrap();
    let bytes = input.elem_count() * std::mem::size_of::<f32>();
    let mut group =
        c.benchmark_group(device.bench_name(format!("sam3_cleanup_mask_logits_{name}")));
    group.throughput(Throughput::Bytes(bytes as u64));

    let op_name = if matches!(device, Device::Cuda(_)) {
        "cuda_kernel"
    } else {
        "exact_cpu"
    };
    group.bench_function(op_name, |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = cleanup_mask_logits(black_box(&input)).unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });

    if matches!(device, Device::Cuda(_)) {
        group.bench_function("cpu_roundtrip", |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = cleanup_mask_logits_cpu_roundtrip(black_box(&input), device).unwrap();
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    for d in device.devices {
        run_cleanup_mask_logits_benchmark(c, &d, 4, 1, 256, 256, "4x1x256x256");
    }
}

criterion_group!(benches, criterion_benchmark);
