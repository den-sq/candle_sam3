#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::sam3;

#[derive(Parser, Debug)]
struct Args {
    /// Optional path to the upstream `sam3.pt` checkpoint.
    #[arg(long)]
    checkpoint: Option<String>,

    /// Optional image path. Present for the eventual image-grounding API.
    #[arg(long)]
    image: Option<String>,

    /// Optional text prompt. Present for the eventual image-grounding API.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    print_config: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let config = sam3::Config::default();

    println!("sam3 scaffold example");
    println!("device: {device:?}");
    println!(
        "image MVP target: {}x{}",
        config.image.image_size, config.image.image_size
    );
    println!("milestones:");
    for step in sam3::Sam3ImageModel::scaffold_milestones() {
        println!("  - {step}");
    }

    if args.print_config {
        println!("{config:#?}");
    }

    if let Some(checkpoint) = args.checkpoint.as_ref() {
        let vb = VarBuilder::from_pth_with_state(checkpoint, DType::F32, "model", &device)?;
        let _model = sam3::Sam3ImageModel::new(&config, vb)?;
        println!("checkpoint opened with state key `model`; scaffold model instantiated");
    }

    if args.image.is_some() || args.prompt.is_some() {
        anyhow::bail!(
            "the sam3 example currently exposes the scaffold only; image preprocessing, text tokenization, grounding, and rendering are not implemented yet"
        );
    }

    Ok(())
}
