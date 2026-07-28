use std::env;
use std::path::{Path, PathBuf};

fn cuda_root() -> Option<PathBuf> {
    [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ]
    .iter()
    .find_map(|name| env::var_os(name).map(PathBuf::from))
    .or_else(|| {
        let default = Path::new("/usr/local/cuda");
        default.exists().then(|| default.to_path_buf())
    })
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/models/sam3/profiling_nvtx.c");
    if env::var_os("CARGO_FEATURE_SAM3_NVTX").is_none() {
        return;
    }

    let root = cuda_root().unwrap_or_else(|| {
        panic!(
            "the sam3-nvtx feature requires CUDA_HOME, CUDA_PATH, CUDA_ROOT, \
             CUDA_TOOLKIT_ROOT_DIR, or /usr/local/cuda"
        )
    });
    let include = root.join("include");
    let header = include.join("nvtx3").join("nvToolsExt.h");
    assert!(
        header.is_file(),
        "the sam3-nvtx feature requires {}, but it was not found",
        header.display()
    );

    cc::Build::new()
        .file("src/models/sam3/profiling_nvtx.c")
        .include(include)
        .warnings(false)
        .compile("candle_sam3_nvtx");
}
