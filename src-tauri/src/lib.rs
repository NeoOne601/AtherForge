pub mod ruvllm_bridge;

use ruvllm_bridge::{init_llm, llm_generate, llm_health};

pub fn run() {
    // Initialize LLM state (graceful fallback if model not found)
    let model_path = std::env::var("QWEN_MODEL_PATH")
        .unwrap_or_else(|_| "/Volumes/Apple/AI Model/qwen2.5-7b-instruct-q4_k_m.gguf".to_string());
    let llm_state = init_llm(&model_path);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .manage(llm_state)
        .invoke_handler(tauri::generate_handler![
            llm_generate,
            llm_health,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
