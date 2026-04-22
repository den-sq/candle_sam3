use std::collections::HashMap;
use std::sync::Mutex;

use super::*;

#[derive(Debug)]
pub(super) struct TrackerSimpleMaskDownSampler {
    interpol_size: [usize; 2],
    convs: Vec<Conv2d>,
    norms: Vec<LayerNorm2d>,
    out_proj: Conv2d,
}

impl TrackerSimpleMaskDownSampler {
    pub(super) fn new(
        config: &Sam3TrackerMaskDownsamplerConfig,
        embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let total_stride = 16usize;
        let stride = config.stride;
        let mut num_layers = 0usize;
        let mut current_stride = 1usize;
        while current_stride < total_stride {
            current_stride *= stride;
            num_layers += 1;
        }
        if current_stride != total_stride {
            candle::bail!(
                "tracker simple mask downsampler expected total_stride {total_stride} to be divisible by stride {}, got effective stride {current_stride}",
                stride
            );
        }

        let encoder_vb = vb.pp("encoder");
        let mut convs = Vec::with_capacity(num_layers);
        let mut norms = Vec::with_capacity(num_layers);
        let mut mask_in_chans = 1usize;
        let mut mask_out_chans = 1usize;
        for layer_idx in 0..num_layers {
            mask_out_chans *= stride * stride;
            convs.push(candle_nn::conv2d(
                mask_in_chans,
                mask_out_chans,
                config.kernel_size,
                Conv2dConfig {
                    stride,
                    padding: config.padding,
                    ..Default::default()
                },
                encoder_vb.pp(layer_idx * 3),
            )?);
            norms.push(LayerNorm2d::new(
                mask_out_chans,
                1e-6,
                encoder_vb.pp(layer_idx * 3 + 1),
            )?);
            mask_in_chans = mask_out_chans;
        }
        let out_proj = candle_nn::conv2d(
            mask_out_chans,
            embed_dim,
            1,
            Default::default(),
            encoder_vb.pp(num_layers * 3),
        )?;
        Ok(Self {
            interpol_size: config.interpol_size,
            convs,
            norms,
            out_proj,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.to_dtype(DType::F32)?;
        let (_, _, height, width) = xs.dims4()?;
        if [height, width] != self.interpol_size {
            xs = resize_bilinear2d_antialias(&xs, self.interpol_size[0], self.interpol_size[1])?;
        }
        for (conv, norm) in self.convs.iter().zip(self.norms.iter()) {
            xs = conv.forward(&xs)?;
            xs = norm.forward(&xs)?;
            xs = xs.gelu_erf()?;
        }
        self.out_proj.forward(&xs)
    }
}

#[derive(Debug)]
struct TrackerCxBlock {
    dwconv: Conv2d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl TrackerCxBlock {
    fn new(config: &Sam3TrackerCxBlockConfig, vb: VarBuilder) -> Result<Self> {
        let groups = if config.use_dwconv { config.dim } else { 1 };
        let dwconv = candle_nn::conv2d(
            config.dim,
            config.dim,
            config.kernel_size,
            Conv2dConfig {
                padding: config.padding,
                groups,
                ..Default::default()
            },
            vb.pp("dwconv"),
        )?;
        let gamma = if config.layer_scale_init_value > 0.0 {
            Some(vb.get((config.dim,), "gamma")?.reshape((1, 1, 1, config.dim))?)
        } else {
            None
        };
        Ok(Self {
            dwconv,
            norm: candle_nn::layer_norm(config.dim, 1e-6, vb.pp("norm"))?,
            pwconv1: linear(vb.pp("pwconv1"), config.dim, 4 * config.dim, true)?,
            pwconv2: linear(vb.pp("pwconv2"), 4 * config.dim, config.dim, true)?,
            gamma,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.dwconv.forward(xs)?;
        xs = xs.permute((0, 2, 3, 1))?.contiguous()?;
        xs = self.norm.forward(&xs)?;
        xs = self.pwconv1.forward(&xs)?;
        xs = xs.gelu_erf()?;
        xs = self.pwconv2.forward(&xs)?;
        if let Some(gamma) = self.gamma.as_ref() {
            xs = xs.broadcast_mul(gamma)?;
        }
        xs = xs.permute((0, 3, 1, 2))?.contiguous()?;
        residual.broadcast_add(&xs)
    }
}

#[derive(Debug)]
struct TrackerSimpleFuser {
    layers: Vec<TrackerCxBlock>,
}

impl TrackerSimpleFuser {
    fn new(
        fuser_config: &Sam3TrackerFuserConfig,
        cx_block_config: &Sam3TrackerCxBlockConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(fuser_config.num_layers);
        for layer_idx in 0..fuser_config.num_layers {
            layers.push(TrackerCxBlock::new(
                cx_block_config,
                layers_vb.pp(layer_idx),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
pub(super) struct TrackerSimpleMaskEncoder {
    mask_downsampler: TrackerSimpleMaskDownSampler,
    pix_feat_proj: Conv2d,
    fuser: TrackerSimpleFuser,
    out_proj: Conv2d,
    position_num_pos_feats: usize,
    position_normalize: bool,
    position_scale: f32,
    position_temperature: f32,
    position_cache: Mutex<HashMap<MaskmemPositionCacheKey, Tensor>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MaskmemPositionCacheKey {
    device: String,
    dtype: String,
    channels: usize,
    height: usize,
    width: usize,
    num_pos_feats: usize,
    normalize: bool,
    scale_bits: u32,
    temperature_bits: u32,
}

impl TrackerSimpleMaskEncoder {
    pub(super) fn new(
        config: &Sam3TrackerMaskmemBackboneConfig,
        hidden_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            mask_downsampler: TrackerSimpleMaskDownSampler::new(
                &config.mask_downsampler,
                hidden_dim,
                vb.pp("mask_downsampler"),
            )?,
            pix_feat_proj: candle_nn::conv2d(
                hidden_dim,
                hidden_dim,
                1,
                Default::default(),
                vb.pp("pix_feat_proj"),
            )?,
            fuser: TrackerSimpleFuser::new(&config.fuser, &config.cx_block, vb.pp("fuser"))?,
            out_proj: candle_nn::conv2d(
                hidden_dim,
                config.out_dim,
                1,
                Default::default(),
                vb.pp("out_proj"),
            )?,
            position_num_pos_feats: config.position_encoding.num_pos_feats,
            position_normalize: config.position_encoding.normalize,
            position_scale: config
                .position_encoding
                .scale
                .unwrap_or(2.0 * std::f32::consts::PI),
            position_temperature: config.position_encoding.temperature,
            position_cache: Mutex::new(HashMap::new()),
        })
    }

    pub(super) fn forward(
        &self,
        pix_feat: &Tensor,
        masks: &Tensor,
        skip_mask_sigmoid: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mut masks = if skip_mask_sigmoid {
            masks.clone()
        } else {
            candle_nn::ops::sigmoid(masks)?
        };
        masks = self.mask_downsampler.forward(&masks)?;
        let pix_feat = if pix_feat.device().same_device(masks.device()) {
            pix_feat.clone()
        } else {
            pix_feat.to_device(masks.device())?
        };
        let mut xs = self.pix_feat_proj.forward(&pix_feat)?;
        xs = xs.broadcast_add(&masks)?;
        xs = self.fuser.forward(&xs)?;
        xs = self.out_proj.forward(&xs)?;
        let pos = self.cached_position_encoding(&xs)?;
        Ok((xs, pos))
    }

    fn cached_position_encoding(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, channels, height, width) = xs.dims4()?;
        let key = MaskmemPositionCacheKey {
            device: format!("{:?}", xs.device()),
            dtype: format!("{:?}", xs.dtype()),
            channels,
            height,
            width,
            num_pos_feats: self.position_num_pos_feats,
            normalize: self.position_normalize,
            scale_bits: self.position_scale.to_bits(),
            temperature_bits: self.position_temperature.to_bits(),
        };
        let cached = {
            let cache = self
                .position_cache
                .lock()
                .expect("maskmem position cache lock poisoned");
            cache.get(&key).cloned()
        };
        match cached {
            Some(pos) => Ok(pos),
            None => {
                let pos = build_2d_sine_position_encoding(
                    xs,
                    self.position_num_pos_feats,
                    self.position_normalize,
                    self.position_scale,
                    self.position_temperature,
                )?
                .to_dtype(xs.dtype())?;
                let mut cache = self
                    .position_cache
                    .lock()
                    .expect("maskmem position cache lock poisoned");
                Ok(cache.entry(key).or_insert_with(|| pos.clone()).clone())
            }
        }
    }
}
