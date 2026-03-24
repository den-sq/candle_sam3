use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub image: ImageConfig,
    pub vision: VisionConfig,
    pub text: TextConfig,
    pub neck: NeckConfig,
    pub geometry: GeometryConfig,
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub segmentation: SegmentationConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            image: ImageConfig::default(),
            vision: VisionConfig::default(),
            text: TextConfig::default(),
            neck: NeckConfig::default(),
            geometry: GeometryConfig::default(),
            encoder: EncoderConfig::default(),
            decoder: DecoderConfig::default(),
            segmentation: SegmentationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageConfig {
    pub image_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            image_size: 1008,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub pretrain_image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub window_size: usize,
    pub global_attn_blocks: Vec<usize>,
    pub use_abs_pos: bool,
    pub tile_abs_pos: bool,
    pub use_rope: bool,
    pub use_interp_rope: bool,
    pub retain_cls_token: bool,
    pub ln_pre: bool,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            pretrain_image_size: 336,
            patch_size: 14,
            embed_dim: 1024,
            depth: 32,
            num_heads: 16,
            mlp_ratio: 4.625,
            window_size: 24,
            global_attn_blocks: vec![7, 15, 23, 31],
            use_abs_pos: true,
            tile_abs_pos: true,
            use_rope: true,
            use_interp_rope: true,
            retain_cls_token: false,
            ln_pre: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub d_model: usize,
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub context_length: usize,
    pub vocab_size: usize,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            width: 1024,
            heads: 16,
            layers: 24,
            context_length: 32,
            vocab_size: 49_408,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct NeckConfig {
    pub d_model: usize,
    pub scale_factors: [f32; 4],
    pub scalp: usize,
    pub add_sam2_neck: bool,
}

impl Default for NeckConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            scale_factors: [4.0, 2.0, 1.0, 0.5],
            scalp: 1,
            add_sam2_neck: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeometryConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub add_cls: bool,
    pub add_post_encode_proj: bool,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            num_layers: 3,
            add_cls: true,
            add_post_encode_proj: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub dim_feedforward: usize,
    pub pool_text_with_mask: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            num_layers: 6,
            num_heads: 8,
            dim_feedforward: 2048,
            pool_text_with_mask: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_queries: usize,
    pub num_heads: usize,
    pub dim_feedforward: usize,
    pub presence_token: bool,
    pub use_text_cross_attention: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            num_layers: 6,
            num_queries: 200,
            num_heads: 8,
            dim_feedforward: 2048,
            presence_token: true,
            use_text_cross_attention: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SegmentationConfig {
    pub enabled: bool,
    pub hidden_dim: usize,
    pub upsampling_stages: usize,
    pub aux_masks: bool,
    pub presence_head: bool,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hidden_dim: 256,
            upsampling_stages: 3,
            aux_masks: false,
            presence_head: false,
        }
    }
}
