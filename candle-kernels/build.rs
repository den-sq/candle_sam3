use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

// Match the CUTLASS submodule revision used by the vendored PyTorch 2.7
// memory-efficient attention kernel.
const CUTLASS_COMMIT: &str = "afa1772203677c5118fcd82537a9c8fefbcc7008";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo:rerun-if-env-changed=CANDLE_DISABLE_F32_SM75_SDPA");
    println!("cargo:rustc-check-cfg=cfg(candle_f32_sm75_sdpa)");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&[
            "cutlass_sdpa.cu",
            "cutlass_sdpa_stub.cu",
            "moe_*.cu",
            "mmvq_gguf.cu",
            "mmq_*.cu",
        ]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/mmvq_gguf.cu",
            "src/mmq_gguf/mmq_quantize.cu",
            "src/mmq_gguf/mmq_instance_q4_0.cu",
            "src/mmq_gguf/mmq_instance_q4_1.cu",
            "src/mmq_gguf/mmq_instance_q5_0.cu",
            "src/mmq_gguf/mmq_instance_q5_1.cu",
            "src/mmq_gguf/mmq_instance_q8_0.cu",
            "src/mmq_gguf/mmq_instance_q2_k.cu",
            "src/mmq_gguf/mmq_instance_q3_k.cu",
            "src/mmq_gguf/mmq_instance_q4_k.cu",
            "src/mmq_gguf/mmq_instance_q5_k.cu",
            "src/mmq_gguf/mmq_instance_q6_k.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;

    let sdpa_available = compute_cap == 75 && env::var_os("CANDLE_DISABLE_F32_SM75_SDPA").is_none();
    let sdpa_source = if sdpa_available {
        "src/cutlass_sdpa.cu"
    } else {
        "src/cutlass_sdpa_stub.cu"
    };
    let mut sdpa_builder = KernelBuilder::new()
        .source_files(vec![sdpa_source])
        .out_dir(&out_dir)
        .arg("-std=c++17")
        .arg("-O3");
    if sdpa_available {
        sdpa_builder = sdpa_builder
            .include_path("src")
            .with_cutlass(Some(CUTLASS_COMMIT))
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda");
    }
    if !is_target_msvc {
        sdpa_builder = sdpa_builder.arg("-Xcompiler").arg("-fPIC");
    }
    sdpa_builder.build_lib(out_dir.join("libcutlass_sdpa.a"))?;
    if sdpa_available {
        println!("cargo:rustc-cfg=candle_f32_sm75_sdpa");
    }

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=cutlass_sdpa");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
