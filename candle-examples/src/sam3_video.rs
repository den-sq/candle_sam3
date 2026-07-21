//! Caller-owned media adapters for the tensor-only SAM3 video API.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::sam3::{
    normalize_rgb_frame_for_sam3, FrameSource, ImageSize, VideoDebugArtifactSink,
};
use image::ImageReader;
use serde::Deserialize;

#[derive(Debug)]
struct FrameBlob {
    data: Vec<f32>,
    frame_size: ImageSize,
}

impl FrameBlob {
    fn to_tensor(&self, target_device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            self.data.clone(),
            (3, self.frame_size.height, self.frame_size.width),
            &Device::Cpu,
        )?
        .to_device(target_device)
    }

    fn memory_bytes(&self) -> usize {
        self.data.len().saturating_mul(std::mem::size_of::<f32>())
    }
}

#[derive(Debug)]
enum MediaSource {
    Images(Vec<PathBuf>),
    Video { path: PathBuf, frame_count: usize },
}

/// Lazy image-folder, image-file, or video-file adapter used by SAM3 examples.
#[derive(Debug)]
pub struct MediaFrameSource {
    source: MediaSource,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    video_size: ImageSize,
    cache: HashMap<usize, FrameBlob>,
}

impl MediaFrameSource {
    pub fn from_path(
        path: impl AsRef<Path>,
        image_size: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    ) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::from_images(sorted_image_paths(path)?, image_size, image_mean, image_std);
        }
        let extension = lowercase_extension(path);
        match extension.as_deref() {
            Some("jpg" | "jpeg" | "png") => {
                Self::from_images(vec![path.to_path_buf()], image_size, image_mean, image_std)
            }
            Some("mp4" | "avi" | "mov" | "mkv" | "webm") => {
                let metadata = probe_video_file(path)?;
                Ok(Self {
                    source: MediaSource::Video {
                        path: path.to_path_buf(),
                        frame_count: metadata.frame_count,
                    },
                    image_size,
                    image_mean,
                    image_std,
                    video_size: metadata.video_size,
                    cache: HashMap::new(),
                })
            }
            _ => candle::bail!("unsupported SAM3 media source path {}", path.display()),
        }
    }

    fn from_images(
        paths: Vec<PathBuf>,
        image_size: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    ) -> Result<Self> {
        let first = paths
            .first()
            .ok_or_else(|| candle::Error::Msg("image source requires at least one path".into()))?;
        let image = ImageReader::open(first)?
            .decode()
            .map_err(candle::Error::wrap)?;
        let video_size = ImageSize::new(image.height() as usize, image.width() as usize);
        Ok(Self {
            source: MediaSource::Images(paths),
            image_size,
            image_mean,
            image_std,
            video_size,
            cache: HashMap::new(),
        })
    }

    fn ensure_loaded(&mut self, frame_idx: usize) -> Result<()> {
        if self.cache.contains_key(&frame_idx) {
            return Ok(());
        }
        let blob = match &self.source {
            MediaSource::Images(paths) => {
                let path = paths.get(frame_idx).ok_or_else(|| {
                    candle::Error::Msg(format!("frame_idx {frame_idx} out of bounds"))
                })?;
                load_image_frame_blob(
                    path,
                    self.image_size,
                    self.image_mean,
                    self.image_std,
                    self.video_size,
                )?
            }
            MediaSource::Video { path, frame_count } => {
                if frame_idx >= *frame_count {
                    candle::bail!(
                        "frame_idx {frame_idx} out of bounds for video with {frame_count} frames"
                    )
                }
                decode_video_frame_blob(
                    path,
                    frame_idx,
                    self.image_size,
                    self.image_mean,
                    self.image_std,
                    self.video_size,
                )?
            }
        };
        self.cache.insert(frame_idx, blob);
        Ok(())
    }
}

impl FrameSource for MediaFrameSource {
    fn frame_count(&self) -> usize {
        match &self.source {
            MediaSource::Images(paths) => paths.len(),
            MediaSource::Video { frame_count, .. } => *frame_count,
        }
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.ensure_loaded(frame_idx)?;
        self.cache
            .get(&frame_idx)
            .expect("frame inserted by ensure_loaded")
            .to_tensor(target_device)
    }

    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()> {
        for frame_idx in frame_indices {
            self.ensure_loaded(*frame_idx)?;
        }
        Ok(())
    }

    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>) {
        self.cache
            .retain(|frame_idx, _| keep_frame_indices.contains(frame_idx));
    }

    fn loaded_frame_count(&self) -> usize {
        self.cache.len()
    }

    fn memory_bytes(&self) -> (usize, usize) {
        (self.cache.values().map(FrameBlob::memory_bytes).sum(), 0)
    }

    fn close(&mut self) {
        self.cache.clear();
    }
}

/// PNG implementation for the model crate's codec-free debug artifact sink.
#[derive(Debug)]
pub struct PngVideoDebugArtifactSink {
    output_root: PathBuf,
}

impl PngVideoDebugArtifactSink {
    pub fn new(output_root: impl Into<PathBuf>) -> Self {
        Self {
            output_root: output_root.into(),
        }
    }
}

impl VideoDebugArtifactSink for PngVideoDebugArtifactSink {
    fn write_binary_mask(
        &self,
        relative_path: &Path,
        width: usize,
        height: usize,
        pixels: &[u8],
    ) -> Result<()> {
        if pixels.len() != width.saturating_mul(height) {
            candle::bail!(
                "debug mask contains {} pixels for {width}x{height}",
                pixels.len()
            )
        }
        let image = image::GrayImage::from_raw(width as u32, height as u32, pixels.to_vec())
            .ok_or_else(|| candle::Error::Msg("invalid debug mask dimensions".into()))?;
        let path = self.output_root.join(relative_path);
        image.save(&path).map_err(candle::Error::wrap)
    }
}

fn load_image_frame_blob(
    path: &Path,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    if matches!(lowercase_extension(path).as_deref(), Some("jpg" | "jpeg")) {
        if let Ok(blob) = load_jpeg_frame_blob_via_pillow(
            path,
            image_size,
            image_mean,
            image_std,
            expected_video_size,
        ) {
            return Ok(blob);
        }
    }
    let image = ImageReader::open(path)?
        .decode()
        .map_err(candle::Error::wrap)?
        .to_rgb8();
    frame_blob_from_rgb_image(
        image,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
        &path.display().to_string(),
    )
}

fn load_jpeg_frame_blob_via_pillow(
    path: &Path,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    let python = std::env::var_os("SAM3_PILLOW_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"));
    let script = r#"
import struct
import sys
from PIL import Image

image = Image.open(sys.argv[1]).convert("RGB")
orig_w, orig_h = image.size
size = int(sys.argv[2])
if image.size != (size, size):
    image = image.resize((size, size), Image.Resampling.BICUBIC)
sys.stdout.buffer.write(struct.pack("<II", orig_w, orig_h))
sys.stdout.buffer.write(image.tobytes())
"#;
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(path)
        .arg(image_size.to_string())
        .output()
        .map_err(candle::Error::wrap)?;
    if !output.status.success() {
        candle::bail!(
            "Pillow frame load failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        )
    }
    frame_blob_from_pillow_bytes(
        &output.stdout,
        path,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
    )
}

fn frame_blob_from_pillow_bytes(
    bytes: &[u8],
    path: &Path,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    if bytes.len() < 8 {
        candle::bail!(
            "Pillow frame load returned truncated output for {}",
            path.display()
        )
    }
    let width = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    validate_source_size(
        ImageSize::new(height, width),
        expected_video_size,
        &path.display().to_string(),
    )?;
    let raw = &bytes[8..];
    let expected_bytes = image_size.saturating_mul(image_size).saturating_mul(3);
    if raw.len() != expected_bytes {
        candle::bail!(
            "Pillow returned {} resized bytes for {}, expected {expected_bytes}",
            raw.len(),
            path.display()
        )
    }
    normalized_blob_from_rgb_bytes(raw.to_vec(), image_size, image_mean, image_std)
}

fn frame_blob_from_rgb_image(
    image: image::RgbImage,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
    source_label: &str,
) -> Result<FrameBlob> {
    let current_size = ImageSize::new(image.height() as usize, image.width() as usize);
    validate_source_size(current_size, expected_video_size, source_label)?;
    let resized = if current_size == ImageSize::square(image_size) {
        image
    } else {
        image::imageops::resize(
            &image,
            image_size as u32,
            image_size as u32,
            image::imageops::FilterType::CatmullRom,
        )
    };
    normalized_blob_from_rgb_bytes(resized.into_raw(), image_size, image_mean, image_std)
}

fn normalized_blob_from_rgb_bytes(
    bytes: Vec<u8>,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
) -> Result<FrameBlob> {
    let image = Tensor::from_vec(bytes, (image_size, image_size, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?;
    let image = (image / 255.)?;
    let normalized = normalize_rgb_frame_for_sam3(&image, image_mean, image_std)?;
    Ok(FrameBlob {
        data: normalized.flatten_all()?.to_vec1::<f32>()?,
        frame_size: ImageSize::square(image_size),
    })
}

fn validate_source_size(current: ImageSize, expected: ImageSize, source_label: &str) -> Result<()> {
    if current != expected {
        candle::bail!(
            "frame {source_label} has size {}x{} but the session expects {}x{}",
            current.height,
            current.width,
            expected.height,
            expected.width
        )
    }
    Ok(())
}

fn decode_video_frame_blob(
    path: &Path,
    frame_idx: usize,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    let select_filter = format!("select=eq(n\\,{frame_idx})");
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &select_filter,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ])
        .output()
        .map_err(candle::Error::wrap)?;
    if !output.status.success() || output.stdout.is_empty() {
        candle::bail!(
            "ffmpeg failed for {} frame {frame_idx}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        )
    }
    let image = image::load(Cursor::new(output.stdout), image::ImageFormat::Png)
        .map_err(candle::Error::wrap)?
        .to_rgb8();
    frame_blob_from_rgb_image(
        image,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
        &format!("{}#{frame_idx}", path.display()),
    )
}

#[derive(Debug)]
struct VideoProbeMetadata {
    video_size: ImageSize,
    frame_count: usize,
}

#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    streams: Vec<FfprobeStream>,
}

#[derive(Debug, Deserialize)]
struct FfprobeStream {
    width: Option<usize>,
    height: Option<usize>,
    nb_frames: Option<String>,
    nb_read_frames: Option<String>,
    duration: Option<String>,
    r_frame_rate: Option<String>,
}

fn probe_video_file(path: &Path) -> Result<VideoProbeMetadata> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames,duration,r_frame_rate",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(candle::Error::wrap)?;
    if !output.status.success() {
        candle::bail!(
            "ffprobe failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        )
    }
    let parsed: FfprobeOutput = serde_json::from_slice(&output.stdout)
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
    let stream = parsed.streams.into_iter().next().ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe found no video stream in {}",
            path.display()
        ))
    })?;
    let width = stream.width.ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe did not report width for {}",
            path.display()
        ))
    })?;
    let height = stream.height.ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe did not report height for {}",
            path.display()
        ))
    })?;
    let frame_count = parse_frame_count(&stream).ok_or_else(|| {
        candle::Error::Msg(format!(
            "could not determine frame count for {}",
            path.display()
        ))
    })?;
    Ok(VideoProbeMetadata {
        video_size: ImageSize::new(height, width),
        frame_count,
    })
}

fn parse_frame_count(stream: &FfprobeStream) -> Option<usize> {
    parse_optional_usize(stream.nb_read_frames.as_deref())
        .or_else(|| parse_optional_usize(stream.nb_frames.as_deref()))
        .or_else(|| {
            let duration = stream.duration.as_deref()?.parse::<f64>().ok()?;
            let (numerator, denominator) = stream.r_frame_rate.as_deref()?.split_once('/')?;
            let fps = numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?;
            let count = (duration * fps).round();
            (count.is_finite() && count > 0.0).then_some(count as usize)
        })
}

fn parse_optional_usize(value: Option<&str>) -> Option<usize> {
    value?
        .parse::<usize>()
        .ok()
        .filter(|frame_count| *frame_count > 0)
}

fn sorted_image_paths(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            matches!(
                lowercase_extension(path).as_deref(),
                Some("jpg" | "jpeg" | "png")
            )
        })
        .collect::<Vec<_>>();
    if paths.is_empty() {
        candle::bail!("no supported image files found in {}", directory.display())
    }
    if paths.iter().all(|path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.parse::<usize>().ok())
            .is_some()
    }) {
        paths.sort_by_key(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| stem.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });
    } else {
        paths.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    }
    Ok(paths)
}

fn lowercase_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_constant_rgb_tensor(tensor: &Tensor, rgb: [u8; 3], size: usize) -> Result<()> {
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        let plane = size * size;
        assert_eq!(values.len(), 3 * plane);
        for (channel, expected) in rgb.into_iter().enumerate() {
            let expected = expected as f32 / 255.0;
            for value in &values[channel * plane..(channel + 1) * plane] {
                assert!((value - expected).abs() <= 1e-6, "{value} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn jpeg_and_png_golden_frames_match_the_tensor_preprocessing_contract() -> Result<()> {
        for (extension, rgb) in [("jpg", [60, 120, 180]), ("png", [64, 128, 192])] {
            let directory = std::env::temp_dir().join(format!(
                "sam3-example-{extension}-golden-{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&directory);
            fs::create_dir_all(&directory)?;
            let path = directory.join(format!("0.{extension}"));
            image::RgbImage::from_pixel(3, 2, image::Rgb(rgb))
                .save(&path)
                .map_err(candle::Error::wrap)?;
            let decoded_rgb = ImageReader::open(&path)?
                .decode()
                .map_err(candle::Error::wrap)?
                .to_rgb8()
                .get_pixel(0, 0)
                .0;
            let mut source =
                MediaFrameSource::from_path(&directory, 4, [0.0; 3], [1.0; 3])?;

            assert_eq!(source.loaded_frame_count(), 0);
            let tensor = source.get_frame(0, &Device::Cpu)?;
            assert_constant_rgb_tensor(&tensor, decoded_rgb, 4)?;
            assert_eq!(source.loaded_frame_count(), 1);
            source.close();
            fs::remove_dir_all(directory)?;
        }
        Ok(())
    }

    #[test]
    fn png_debug_sink_writes_a_readable_binary_mask() -> Result<()> {
        let output_root =
            std::env::temp_dir().join(format!("sam3-example-png-sink-{}", std::process::id()));
        let _ = fs::remove_dir_all(&output_root);
        fs::create_dir_all(&output_root)?;
        let sink = PngVideoDebugArtifactSink::new(&output_root);
        sink.write_binary_mask(Path::new("mask.png"), 2, 2, &[0, 255, 255, 0])?;

        let image = ImageReader::open(output_root.join("mask.png"))?
            .decode()
            .map_err(candle::Error::wrap)?
            .to_luma8();
        assert_eq!(image.dimensions(), (2, 2));
        assert_eq!(image.into_raw(), vec![0, 255, 255, 0]);
        fs::remove_dir_all(output_root)?;
        Ok(())
    }
}
