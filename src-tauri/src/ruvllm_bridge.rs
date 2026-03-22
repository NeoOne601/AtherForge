// AetherForge v1.0 — src-tauri/src/ruvllm_bridge.rs
// ─────────────────────────────────────────────────────────────────
// ruvllm Rust bridge — Tauri command for native LLM inference.
// Replaces Ollama HTTP round-trip with direct GGUF runtime on Metal.
// Context: 16384 tokens (2× old 8192 limit).
// ─────────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;

// ── State ───────────────────────────────────────────────────────

/// Holds the ruvllm model loaded at startup.
/// Wrapped in Arc<Mutex<>> for safe concurrent Tauri command access.
pub struct LLMState {
    pub model_path: String,
    pub initialized: bool,
    // When ruvllm crate is available, this holds: Arc<Mutex<RuvLLM>>
    // For now, we forward to the Python FastAPI backend's existing inference.
}

// ── Request / Response DTOs ─────────────────────────────────────

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system_prompt: Option<String>,
}

#[derive(Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: u32,
    pub duration_ms: u64,
}

// ── Tauri Commands ──────────────────────────────────────────────

/// Generate text using the ruvllm model.
/// Falls back gracefully if model is not loaded.
#[tauri::command]
pub async fn llm_generate(
    state: State<'_, LLMState>,
    request: GenerateRequest,
) -> Result<GenerateResponse, String> {
    if !state.initialized {
        return Err(
            "ruvllm model not loaded. Using Python backend inference.".to_string()
        );
    }

    let start = std::time::Instant::now();

    // ── ruvllm native inference ─────────────────────────────
    // When the ruvllm crate is compiled and available:
    //
    // use ruvllm::{RuvLLM, GenerateOptions, Message, Role};
    //
    // let llm = state.model.lock().await;
    // let opts = GenerateOptions {
    //     max_tokens: request.max_tokens.unwrap_or(1024),
    //     temperature: request.temperature.unwrap_or(0.7),
    //     repetition_penalty: 1.15,
    //     n_ctx: 16384,
    // };
    // let messages = if let Some(sys) = request.system_prompt {
    //     vec![
    //         Message { role: Role::System, content: sys },
    //         Message { role: Role::User, content: request.prompt },
    //     ]
    // } else {
    //     vec![Message { role: Role::User, content: request.prompt }]
    // };
    // let result = llm.chat(messages, opts)
    //     .await
    //     .map_err(|e| format!("ruvllm error: {e}"))?;
    //
    // Ok(GenerateResponse {
    //     text: result.content,
    //     tokens_generated: result.tokens_generated,
    //     duration_ms: start.elapsed().as_millis() as u64,
    // })

    // Placeholder until ruvllm crate compiles:
    Ok(GenerateResponse {
        text: format!(
            "[ruvllm bridge] Model: {}. Prompt length: {} chars. \
             Use Python backend for inference until ruvllm crate is compiled.",
            state.model_path,
            request.prompt.len()
        ),
        tokens_generated: 0,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

/// Health check for the ruvllm model.
#[tauri::command]
pub async fn llm_health(state: State<'_, LLMState>) -> Result<bool, String> {
    Ok(state.initialized)
}

// ── Initialization ──────────────────────────────────────────────

/// Initialize the LLM state. Called at Tauri startup.
/// Returns LLMState — if model loading fails, returns uninitialized state
/// so the app can fall back to Python backend inference.
pub fn init_llm(model_path: &str) -> LLMState {
    let path = std::path::Path::new(model_path);
    if !path.exists() {
        eprintln!(
            "WARNING: Model file not found at '{}'. \
             ruvllm bridge will be inactive. Using Python backend inference.",
            model_path
        );
        return LLMState {
            model_path: model_path.to_string(),
            initialized: false,
        };
    }

    // When ruvllm crate is available:
    // match RuvLLM::new(model_path, "metal") {
    //     Ok(llm) => LLMState {
    //         model: Arc::new(Mutex::new(llm)),
    //         model_path: model_path.to_string(),
    //         initialized: true,
    //     },
    //     Err(e) => {
    //         eprintln!("WARNING: Failed to load ruvllm model: {e}");
    //         LLMState { model_path: model_path.to_string(), initialized: false }
    //     }
    // }

    eprintln!(
        "INFO: ruvllm bridge configured for model '{}'. \
         Native inference pending ruvllm crate compilation.",
        model_path
    );
    LLMState {
        model_path: model_path.to_string(),
        initialized: false, // Set to true when ruvllm crate compiles
    }
}
