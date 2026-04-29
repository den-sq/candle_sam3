use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use candle::Device;
use candle_transformers::models::sam3;
use image::GenericImageView;
use serde_json::json;

use super::{BatchBox, BatchJob, RenderStyle};

const IMAGE1_URL: &str = "http://images.cocodataset.org/val2017/000000077595.jpg";
const IMAGE2_URL: &str = "http://images.cocodataset.org/val2017/000000136466.jpg";

pub(crate) fn run(
    model: &sam3::Sam3ImageModel,
    tokenizer_path: Option<&str>,
    _notebook_asset_root: Option<&str>,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let example_root = output_dir.join("sam3_image_batched_inference");
    let asset_cache_dir = example_root.join("_assets");
    std::fs::create_dir_all(&asset_cache_dir)?;

    let image1_path =
        ensure_cached_image(&asset_cache_dir, IMAGE1_URL, "000000077595.jpg", "image 1")?;
    let image2_path =
        ensure_cached_image(&asset_cache_dir, IMAGE2_URL, "000000136466.jpg", "image 2")?;
    let image2_dims = image::ImageReader::open(&image2_path)?
        .decode()
        .map_err(anyhow::Error::from)?
        .dimensions();

    std::fs::write(
        example_root.join("notebook_match.json"),
        serde_json::to_string_pretty(&json!({
            "notebook": "sam3_image_batched_inference.ipynb",
            "inputs": [
                {
                    "name": "image1",
                    "source_url": IMAGE1_URL,
                    "cached_path": image1_path.display().to_string(),
                },
                {
                    "name": "image2",
                    "source_url": IMAGE2_URL,
                    "cached_path": image2_path.display().to_string(),
                }
            ],
        }))?,
    )?;

    let image1 = image1_path.display().to_string();
    let image2 = image2_path.display().to_string();
    let jobs = vec![
        BatchJob {
            name: Some("image1_text_cat".to_string()),
            image: image1.clone(),
            prompt: Some("cat".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![],
        },
        BatchJob {
            name: Some("image1_text_laptop".to_string()),
            image: image1,
            prompt: Some("laptop".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![],
        },
        BatchJob {
            name: Some("image2_text_pot".to_string()),
            image: image2.clone(),
            prompt: Some("pot".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![],
        },
        BatchJob {
            name: Some("image2_visual_left_dial".to_string()),
            image: image2.clone(),
            prompt: Some("visual".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![normalized_xyxy_box(
                [59.0, 144.0, 76.0, 163.0],
                image2_dims,
                1,
            )],
        },
        BatchJob {
            name: Some("image2_visual_dial_and_button".to_string()),
            image: image2.clone(),
            prompt: Some("visual".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![
                normalized_xyxy_box([59.0, 144.0, 76.0, 163.0], image2_dims, 1),
                normalized_xyxy_box([87.0, 148.0, 104.0, 159.0], image2_dims, 1),
            ],
        },
        BatchJob {
            name: Some("image2_text_handle".to_string()),
            image: image2.clone(),
            prompt: Some("handle".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![],
        },
        BatchJob {
            name: Some("image2_text_handle_with_negative_box".to_string()),
            image: image2,
            prompt: Some("handle".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![normalized_xyxy_box(
                [40.0, 183.0, 318.0, 204.0],
                image2_dims,
                0,
            )],
        },
    ];

    super::run_batch_jobs(
        model,
        tokenizer_path,
        "sam3_image_batched_inference.ipynb",
        &jobs,
        &example_root,
        RenderStyle::Combined,
        device,
    )
}

fn ensure_cached_image(
    cache_dir: &Path,
    url: &str,
    filename: &str,
    label: &str,
) -> Result<PathBuf> {
    let path = cache_dir.join(filename);
    if path.exists() {
        return Ok(path);
    }

    let status = Command::new("curl")
        .args(["-L", "--fail", "--output"])
        .arg(&path)
        .arg(url)
        .status()
        .with_context(|| format!("failed to launch curl while downloading {label} from {url}"))?;
    if !status.success() {
        bail!("curl failed while downloading {label} from {url}");
    }
    Ok(path)
}

fn normalized_xyxy_box(box_xyxy: [f32; 4], image_dims: (u32, u32), label: u32) -> BatchBox {
    let (width, height) = image_dims;
    let box_width = box_xyxy[2] - box_xyxy[0];
    let box_height = box_xyxy[3] - box_xyxy[1];
    let center_x = box_xyxy[0] + box_width * 0.5;
    let center_y = box_xyxy[1] + box_height * 0.5;
    BatchBox {
        cx: center_x / width as f32,
        cy: center_y / height as f32,
        w: box_width / width as f32,
        h: box_height / height as f32,
        label,
    }
}
