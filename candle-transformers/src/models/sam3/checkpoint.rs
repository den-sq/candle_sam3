use std::path::{Path, PathBuf};

use candle::{DType, Device, Result};
use candle_nn::VarBuilder;

use super::{Config, Sam3ImageModel};

pub const UPSTREAM_SAM3_STATE_KEY: &str = "model";
pub const UPSTREAM_SAM3_DETECTOR_PREFIX: &str = "detector.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sam3CheckpointSource {
    UpstreamPth(PathBuf),
}

impl Sam3CheckpointSource {
    pub fn upstream_pth<P: Into<PathBuf>>(path: P) -> Self {
        Self::UpstreamPth(path.into())
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::UpstreamPth(path) => path.as_path(),
        }
    }

    pub fn load_var_builder(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
        match self {
            Self::UpstreamPth(path) => load_upstream_detector_var_builder(path, dtype, device),
        }
    }

    pub fn load_image_model(
        &self,
        config: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Sam3ImageModel> {
        Sam3ImageModel::new(config, self.load_var_builder(dtype, device)?)
    }
}

pub fn load_upstream_detector_var_builder<P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let vb = VarBuilder::from_pth_with_state(path, dtype, UPSTREAM_SAM3_STATE_KEY, device)?;
    Ok(vb.rename_f(map_image_tensor_to_upstream_checkpoint_name))
}

pub fn map_image_tensor_to_upstream_checkpoint_name(name: &str) -> String {
    if name.starts_with(UPSTREAM_SAM3_DETECTOR_PREFIX) {
        name.to_owned()
    } else {
        format!("{UPSTREAM_SAM3_DETECTOR_PREFIX}{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::map_image_tensor_to_upstream_checkpoint_name;

    #[test]
    fn adds_detector_prefix_for_image_model_names() {
        assert_eq!(
            map_image_tensor_to_upstream_checkpoint_name("backbone.vision_trunk.patch_embed"),
            "detector.backbone.vision_trunk.patch_embed"
        );
    }

    #[test]
    fn preserves_existing_detector_prefix() {
        assert_eq!(
            map_image_tensor_to_upstream_checkpoint_name(
                "detector.transformer.decoder.query_embed"
            ),
            "detector.transformer.decoder.query_embed"
        );
    }
}
