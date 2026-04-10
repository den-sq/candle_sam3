use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::sam3;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,

    #[arg(long)]
    image: PathBuf,

    #[arg(long)]
    box_cx: f32,

    #[arg(long)]
    box_cy: f32,

    #[arg(long)]
    box_w: f32,

    #[arg(long)]
    box_h: f32,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value = "cuda")]
    device: String,

    #[arg(long)]
    dummy_image: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup debug exporter
    let exporter = sam3::DebugExporter::new(&args.output)?;
    sam3::set_exporter(Some(exporter));

    // Load model
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::cuda_if_available(0)?,
        other => candle::bail!("unsupported device: {}. expected cpu or cuda", other),
    };
    println!("[sam3_debug] device = {:?}", device);
    let config = sam3::Config::default();
    println!(
        "[sam3_debug] loading model checkpoint from {:?}...",
        args.checkpoint
    );
    let model =
        sam3::Sam3ImageModel::from_upstream_pth(&config, &args.checkpoint, DType::F32, &device)?;
    println!("[sam3_debug] model loaded");

    // Load image
    let image = if args.dummy_image {
        println!(
            "[sam3_debug] creating dummy image tensor on device {:?}...",
            device
        );
        Tensor::zeros((1, 3, 224, 224), DType::F32, &device)?
    } else {
        println!("[sam3_debug] loading image tensor from {:?}...", args.image);
        let image = load_image_tensor(&args.image, &device)?;
        println!("[sam3_debug] image loaded: {:?}", image.dims());
        image
    };

    // Create geometry prompt
    let geometry_prompt = sam3::GeometryPrompt {
        boxes_cxcywh: Some(Tensor::new(
            &[args.box_cx, args.box_cy, args.box_w, args.box_h],
            &device,
        )?),
        box_labels: Some(Tensor::new(&[1u32], &device)?),
        points_xy: None,
        point_labels: None,
        masks: None,
        mask_labels: None,
    };

    // Run the geometry encoder on the loaded image and prompt
    let visual = model.encode_image_features(&image)?;
    let _encoded_prompt = model.encode_geometry_prompt(&geometry_prompt, &visual)?;

    // Finalize export
    sam3::finish()?;

    println!("Debug export complete: {}", args.output.display());

    Ok(())
}

fn load_image_tensor(path: &PathBuf, device: &Device) -> Result<Tensor> {
    // Simple image loading - assumes preprocessed tensor saved as .safetensors
    // In real implementation, this would load and preprocess an image file
    let tensors = candle::safetensors::load(path, device)?;
    let image = tensors
        .get("inputs.image")
        .or_else(|| tensors.get("image"))
        .ok_or_else(|| {
            candle::Error::Msg("image tensor not found in safetensors bundle".to_owned())
        })?;
    Ok(image.clone())
}
