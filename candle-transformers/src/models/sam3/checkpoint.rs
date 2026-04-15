use std::path::{Path, PathBuf};

use candle::{DType, Device, Result};
use candle_nn::VarBuilder;

use super::{Config, Sam3ImageModel};

pub const UPSTREAM_SAM3_STATE_KEY: &str = "model";
pub const UPSTREAM_SAM3_DETECTOR_PREFIX: &str = "detector.";
pub const UPSTREAM_SAM3_TRACKER_PREFIX: &str = "tracker.";

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

    pub fn load_tracker_var_builder(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        match self {
            Self::UpstreamPth(path) => load_upstream_tracker_var_builder(path, dtype, device),
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
    let path = path.as_ref();
    let vb = match VarBuilder::from_pth_with_state(path, dtype, UPSTREAM_SAM3_STATE_KEY, device) {
        Ok(vb) => vb,
        Err(err) if should_fallback_to_direct_state_dict(&err) => {
            VarBuilder::from_pth(path, dtype, device)?
        }
        Err(err) => return Err(err),
    };
    Ok(vb.rename_f(map_image_tensor_to_upstream_checkpoint_name))
}

pub fn load_upstream_tracker_var_builder<P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let path = path.as_ref();
    let vb = match VarBuilder::from_pth_with_state(path, dtype, UPSTREAM_SAM3_STATE_KEY, device) {
        Ok(vb) => vb,
        Err(err) if should_fallback_to_direct_state_dict(&err) => {
            VarBuilder::from_pth(path, dtype, device)?
        }
        Err(err) => return Err(err),
    };
    Ok(vb.rename_f(map_tracker_tensor_to_upstream_checkpoint_name))
}

pub fn map_image_tensor_to_upstream_checkpoint_name(name: &str) -> String {
    if name.starts_with(UPSTREAM_SAM3_DETECTOR_PREFIX) {
        name.to_owned()
    } else {
        format!("{UPSTREAM_SAM3_DETECTOR_PREFIX}{name}")
    }
}

pub fn map_tracker_tensor_to_upstream_checkpoint_name(name: &str) -> String {
    if name.starts_with(UPSTREAM_SAM3_TRACKER_PREFIX) {
        name.to_owned()
    } else {
        format!("{UPSTREAM_SAM3_TRACKER_PREFIX}{name}")
    }
}

fn should_fallback_to_direct_state_dict(err: &candle::Error) -> bool {
    err.to_string()
        .contains(&format!("key {UPSTREAM_SAM3_STATE_KEY} not found"))
}

#[cfg(test)]
mod tests {
    use super::{
        map_image_tensor_to_upstream_checkpoint_name,
        map_tracker_tensor_to_upstream_checkpoint_name, should_fallback_to_direct_state_dict,
    };

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

    #[test]
    fn adds_tracker_prefix_for_tracker_names() {
        assert_eq!(
            map_tracker_tensor_to_upstream_checkpoint_name("sam_prompt_encoder.point_embeddings.0"),
            "tracker.sam_prompt_encoder.point_embeddings.0"
        );
    }

    #[test]
    fn preserves_existing_tracker_prefix() {
        assert_eq!(
            map_tracker_tensor_to_upstream_checkpoint_name("tracker.maskmem_tpos_enc"),
            "tracker.maskmem_tpos_enc"
        );
    }

    #[test]
    fn falls_back_when_checkpoint_is_not_wrapped_in_model_key() {
        let err = candle::Error::msg("key model not found");
        assert!(should_fallback_to_direct_state_dict(&err));
    }
}
