use std::collections::BTreeSet;

use candle::{DType, Device, Result, Tensor};

use super::super::image::ImageSize;
use super::VideoSessionOptions;

/// Tensor-backed compatibility source.
///
/// File and codec ownership intentionally lives in callers. Use
/// [`Sam3VideoPredictor::start_session_with_frame_source`](super::Sam3VideoPredictor::start_session_with_frame_source)
/// to inject a lazy runtime adapter.
#[derive(Debug, Clone)]
pub enum VideoSource {
    TensorFrames(Vec<Tensor>),
}

impl VideoSource {
    pub(crate) fn into_frame_source(
        self,
        session_options: &VideoSessionOptions,
    ) -> Result<Box<dyn FrameSource>> {
        match self {
            Self::TensorFrames(frames) => Ok(Box::new(TensorFrameSource::new(
                frames,
                session_options.offload_frames_to_cpu,
            )?)),
        }
    }
}

/// Lazy, tensor-only video frame boundary.
///
/// Implementations own decode, resize, caching, and source lifetime. Returned
/// frames must be normalized RGB CHW or BCHW tensors ready for the SAM3 image
/// backbone. `memory_bytes` reports cached source storage on CPU and accelerator
/// devices respectively.
pub trait FrameSource: Send {
    fn frame_count(&self) -> usize;
    fn video_size(&self) -> ImageSize;
    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor>;
    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()>;
    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>);
    fn loaded_frame_count(&self) -> usize;
    fn memory_bytes(&self) -> (usize, usize);
    fn close(&mut self);
}

/// Apply SAM3 channel normalization to an RGB CHW or BCHW tensor in `[0, 1]`.
///
/// Decode and resize stay caller-owned. This helper locks the tensor-only
/// normalization contract shared by runtime adapters and model callers.
pub fn normalize_rgb_frame_for_sam3(
    image: &Tensor,
    image_mean: [f32; 3],
    image_std: [f32; 3],
) -> Result<Tensor> {
    let image = image.to_dtype(DType::F32)?;
    let (image, squeeze_batch) = match image.rank() {
        3 => (image.unsqueeze(0)?, true),
        4 => (image, false),
        rank => candle::bail!("expected RGB CHW or BCHW frame tensor, got rank {rank}"),
    };
    let channels = image.dim(1)?;
    if channels != 3 {
        candle::bail!("expected RGB frame tensor, got {channels} channels")
    }
    let device = image.device();
    let mean = Tensor::from_vec(image_mean.to_vec(), (1, 3, 1, 1), device)?;
    let std = Tensor::from_vec(image_std.to_vec(), (1, 3, 1, 1), device)?;
    let normalized = image.broadcast_sub(&mean)?.broadcast_div(&std)?;
    if squeeze_batch {
        normalized.squeeze(0)
    } else {
        Ok(normalized)
    }
}

struct TensorFrameSource {
    frames: Vec<Tensor>,
    video_size: ImageSize,
}

impl TensorFrameSource {
    fn new(frames: Vec<Tensor>, offload_to_cpu: bool) -> Result<Self> {
        if frames.is_empty() {
            candle::bail!("tensor frame source requires at least one frame")
        }
        let first = if offload_to_cpu && !matches!(frames[0].device(), Device::Cpu) {
            frames[0].to_device(&Device::Cpu)?
        } else {
            frames[0].clone()
        };
        let (channels, height, width) = match first.rank() {
            3 => first.dims3()?,
            4 => {
                let (_batch, channels, height, width) = first.dims4()?;
                (channels, height, width)
            }
            rank => candle::bail!("expected CHW or BCHW frame tensor, got rank {rank}"),
        };
        if channels != 3 {
            candle::bail!("tensor frame source expects RGB frames, got {channels} channels")
        }
        let frames = if offload_to_cpu {
            frames
                .into_iter()
                .map(|frame| {
                    if matches!(frame.device(), Device::Cpu) {
                        Ok(frame)
                    } else {
                        frame.to_device(&Device::Cpu)
                    }
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            frames
        };
        Ok(Self {
            frames,
            video_size: ImageSize::new(height, width),
        })
    }
}

impl FrameSource for TensorFrameSource {
    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.frames
            .get(frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {frame_idx} out of bounds")))?
            .to_device(target_device)
    }

    fn prefetch(&mut self, _frame_indices: &[usize]) -> Result<()> {
        Ok(())
    }

    fn evict_except(&mut self, _keep_frame_indices: &BTreeSet<usize>) {}

    fn loaded_frame_count(&self) -> usize {
        self.frames.len()
    }

    fn memory_bytes(&self) -> (usize, usize) {
        let mut cpu = 0usize;
        let mut device = 0usize;
        for frame in &self.frames {
            let bytes = frame
                .elem_count()
                .saturating_mul(frame.dtype().size_in_bytes());
            if matches!(frame.device(), Device::Cpu) {
                cpu = cpu.saturating_add(bytes);
            } else {
                device = device.saturating_add(bytes);
            }
        }
        (cpu, device)
    }

    fn close(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_frame_source_reports_cpu_memory_usage() -> Result<()> {
        let frame_a = Tensor::zeros((3, 2, 2), DType::F32, &Device::Cpu)?;
        let frame_b = Tensor::ones((3, 2, 2), DType::F32, &Device::Cpu)?;
        let source = TensorFrameSource::new(vec![frame_a, frame_b], true)?;

        let (cpu_bytes, device_bytes) = source.memory_bytes();

        assert_eq!(device_bytes, 0);
        assert_eq!(cpu_bytes, 2 * 3 * 2 * 2 * std::mem::size_of::<f32>());
        Ok(())
    }

    #[test]
    fn tensor_only_normalization_preserves_shape_and_values() -> Result<()> {
        let image = Tensor::from_vec(vec![0.25f32, 0.5, 0.75], (3, 1, 1), &Device::Cpu)?;
        let normalized = normalize_rgb_frame_for_sam3(&image, [0.1, 0.2, 0.3], [0.5, 0.5, 0.5])?;

        assert_eq!(normalized.dims(), &[3, 1, 1]);
        assert_eq!(
            normalized.flatten_all()?.to_vec1::<f32>()?,
            vec![0.3, 0.6, 0.9]
        );
        Ok(())
    }

    #[test]
    fn tensor_only_normalization_rejects_non_rgb_input() -> Result<()> {
        let image = Tensor::zeros((1, 2, 2), DType::F32, &Device::Cpu)?;
        let error = normalize_rgb_frame_for_sam3(&image, [0.0; 3], [1.0; 3]).unwrap_err();
        assert!(error.to_string().contains("expected RGB frame tensor"));
        Ok(())
    }
}
