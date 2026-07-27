use candle_core::{backend::BackendDevice, DType, Device, Tensor};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn run(x: &Tensor, kernel: &Tensor) {
    x.conv_transpose2d(kernel, 0, 0, 2, 1).unwrap();
}

fn run_sam3_neck_benchmark(
    c: &mut Criterion,
    device: &Device,
    c_in: usize,
    c_out: usize,
    h_in: usize,
) {
    let x = Tensor::zeros((1, c_in, h_in, h_in), DType::F32, device).unwrap();
    let kernel = Tensor::zeros((c_in, c_out, 2, 2), DType::F32, device).unwrap();
    let h_out = h_in * 2;
    let bytes =
        (x.elem_count() + kernel.elem_count() + c_out * h_out * h_out) * DType::F32.size_in_bytes();
    let name = format!("cuda_sam3_neck_{c_in}x{h_in}_to_{c_out}x{h_out}_f32");

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                run(black_box(&x), black_box(&kernel));
            }
            match device {
                Device::Cuda(device) => device.synchronize().unwrap(),
                _ => unreachable!("conv_transpose2d benchmark requires CUDA"),
            }
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = Device::new_cuda(0).unwrap();
    run_sam3_neck_benchmark(c, &device, 1024, 512, 72);
    run_sam3_neck_benchmark(c, &device, 512, 256, 144);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
