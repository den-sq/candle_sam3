#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::DType;
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
    let checkpoint_source = args
        .checkpoint
        .as_ref()
        .map(|path| sam3::Sam3CheckpointSource::upstream_pth(path));

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

    let model = if let Some(checkpoint) = checkpoint_source.as_ref() {
        let model =
            sam3::Sam3ImageModel::from_checkpoint_source(&config, checkpoint, DType::F32, &device)?;
        println!(
            "checkpoint opened with state key `{}` and image-model namespace remap",
            sam3::UPSTREAM_SAM3_STATE_KEY
        );
        Some(model)
    } else {
        None
    };

    if let Some(image_path) = args.image.as_ref() {
        let Some(model) = model.as_ref() else {
            anyhow::bail!(
                "loading an image into typed SAM3 state currently requires `--checkpoint <sam3.pt>`"
            );
        };
        let (image, _h, _w) =
            candle_examples::load_image(image_path, Some(config.image.image_size))?;
        let image = image.to_device(&device)?;
        let mut state = model.set_image(&image)?;
        if let Some(prompt) = args.prompt.as_ref() {
            state = state.with_text_prompt(prompt.clone());
        }
        println!("prepared typed image state:\n{state:#?}");
    }

    if args.prompt.is_some() {
        anyhow::bail!(
            "the sam3 example now prepares typed state and checkpoint loading, but actual grounding and rendering are not implemented yet"
        );
    }

    Ok(())
}
