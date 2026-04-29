use std::path::Path;

use anyhow::{Context, Result};
use candle::Device;
use candle_transformers::models::sam3;
use serde_json::json;

use super::{BatchBox, BatchJob, RenderStyle};

pub(crate) fn run(
    model: &sam3::Sam3ImageModel,
    tokenizer_path: Option<&str>,
    notebook_asset_root: Option<&str>,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let asset_root = super::resolve_notebook_asset_root(notebook_asset_root)?;
    let image_path = asset_root.join("images/test_image.jpg");
    let image_path_str = image_path
        .to_str()
        .context("test_image.jpg path is not valid UTF-8")?;
    let example_root = output_dir.join("sam3_image_predictor_example");
    std::fs::create_dir_all(&example_root)?;
    std::fs::write(
        example_root.join("notebook_match.json"),
        serde_json::to_string_pretty(&json!({
            "notebook": "sam3_image_predictor_example.ipynb",
            "asset_root": asset_root.display().to_string(),
            "image_path": image_path.display().to_string(),
            "jobs": [
                "image_predictor_text_shoe",
                "image_predictor_single_positive_box",
                "image_predictor_positive_negative_boxes",
            ],
        }))?,
    )?;

    let jobs = vec![
        BatchJob {
            name: Some("image_predictor_text_shoe".to_string()),
            image: image_path_str.to_string(),
            prompt: Some("shoe".to_string()),
            smoke_image_size: None,
            points: vec![],
            boxes: vec![],
        },
        BatchJob {
            name: Some("image_predictor_single_positive_box".to_string()),
            image: image_path_str.to_string(),
            prompt: None,
            smoke_image_size: None,
            points: vec![],
            boxes: vec![BatchBox {
                cx: 0.41796875,
                cy: 0.6527777777777778,
                w: 0.0859375,
                h: 0.5,
                label: 1,
            }],
        },
        BatchJob {
            name: Some("image_predictor_positive_negative_boxes".to_string()),
            image: image_path_str.to_string(),
            prompt: None,
            smoke_image_size: None,
            points: vec![],
            boxes: vec![
                BatchBox {
                    cx: 0.41796875,
                    cy: 0.6527777777777778,
                    w: 0.0859375,
                    h: 0.5,
                    label: 1,
                },
                BatchBox {
                    cx: 0.333984375,
                    cy: 0.6493055555555556,
                    w: 0.08984375,
                    h: 0.5208333333333334,
                    label: 0,
                },
            ],
        },
    ];

    super::run_batch_jobs(
        model,
        tokenizer_path,
        "sam3_image_predictor_example.ipynb",
        &jobs,
        &example_root,
        RenderStyle::NotebookImagePredictor,
        device,
    )
}
