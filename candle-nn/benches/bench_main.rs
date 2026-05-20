mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::cleanup::benches,
    benchmarks::norm::benches,
    benchmarks::softmax::benches,
    benchmarks::conv::benches
);
