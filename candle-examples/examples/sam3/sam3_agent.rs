use std::path::Path;

use anyhow::{bail, Result};
use candle::Device;
use candle_transformers::models::sam3;

pub(crate) fn run(
    _model: &sam3::Sam3ImageModel,
    _tokenizer_path: Option<&str>,
    _notebook_asset_root: Option<&str>,
    _output_dir: &Path,
    _device: &Device,
) -> Result<()> {
    bail!(
        "sam3_agent.ipynb depends on an external multimodal LLM tool loop that is not implemented in the Candle SAM3 runtime yet"
    )
}
