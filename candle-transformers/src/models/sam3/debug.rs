use candle::{Result, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

thread_local! {
    static DEBUG_EXPORTER: RefCell<Option<DebugExporter>> = RefCell::new(None);
}

#[derive(Clone)]
pub struct DebugExporter {
    output_dir: std::path::PathBuf,
    tensors: Vec<(String, Tensor)>,
}

impl DebugExporter {
    pub fn new(output_dir: impl AsRef<Path>) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_dir)?;
        Ok(Self {
            output_dir,
            tensors: Vec::new(),
        })
    }

    pub fn capture(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        if tensor.elem_count() == 0 {
            return Ok(());
        }
        self.tensors
            .push((name.to_string(), tensor.force_contiguous()?));
        Ok(())
    }

    pub fn finalize(&self) -> Result<()> {
        // Convert tensors to save format - need owned Strings for keys
        let mut pairs = HashMap::new();
        for (name, tensor) in &self.tensors {
            pairs.insert(name.clone(), tensor.clone());
        }

        let path = self.output_dir.join("debug_tensors.safetensors");
        candle::safetensors::save(&pairs, &path)
            .map_err(|e| candle::Error::Msg(format!("Save failed: {}", e)))?;

        // Save metadata
        let metadata: Vec<_> = self
            .tensors
            .iter()
            .map(|(name, tensor)| {
                serde_json::json!({
                    "name": name,
                    "shape": tensor.dims(),
                    "dtype": format!("{:?}", tensor.dtype()),
                })
            })
            .collect();

        let meta_path = self.output_dir.join("metadata.json");
        std::fs::write(
            meta_path,
            serde_json::to_string_pretty(&metadata)
                .map_err(|e| candle::Error::Msg(format!("JSON serialization failed: {}", e)))?,
        )?;

        Ok(())
    }
}

// Thread-local API
pub fn set_exporter(ctx: Option<DebugExporter>) {
    DEBUG_EXPORTER.with(|e| *e.borrow_mut() = ctx);
}

pub fn capture_tensor(name: &str, tensor: &Tensor) -> Result<()> {
    DEBUG_EXPORTER.with(|e| {
        if let Some(exp) = e.borrow_mut().as_mut() {
            return exp.capture(name, tensor);
        }
        Ok(())
    })
}

pub fn finish() -> Result<()> {
    DEBUG_EXPORTER.with(|e| {
        if let Some(exp) = e.take() {
            return exp.finalize();
        }
        Ok(())
    })
}
