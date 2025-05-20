use crate::proxy::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, ModelInfo, ProxyStream,
};
use anyhow::Result;
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
};
use log::{debug, info};
use md5;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use super::{ApiError, ServerState};

// 根路由处理程序，返回 Ollama 运行状态
pub(crate) async fn root_handler() -> Response {
    Json(json!({
        "status": "running"
    }))
    .into_response()
}

// 列出所有可用模型
pub(crate) async fn list_models_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Response, ApiError> {
    let models = state.proxy.get_all_models().await.map_err(ApiError::from)?;

    // 将 OpenAI 模型响应转换为 Ollama 格式
    let ollama_models = models
        .data
        .iter()
        .map(|model| {
            json!({
                "name": model.name,
                "id": model.id,
                "modified_at": model.created.unwrap_or(0),
                "size": 0,  // Ollama 特有字段
                "digest": "",  // Ollama 特有字段
            })
        })
        .collect::<Vec<_>>();

    Ok(Json(json!({
        "models": ollama_models
    }))
    .into_response())
}

// 获取特定模型信息
pub(crate) async fn get_model_handler(
    State(state): State<Arc<ServerState>>,
    Path(model_id): Path<String>,
) -> Result<Response, ApiError> {
    let model = state
        .proxy
        .get_model(&model_id)
        .await
        .map_err(ApiError::from)?;

    // 转换为 Ollama 格式
    Ok(Json(json!({
        "name": model.name,
        "id": model.id,
        "modified_at": model.created.unwrap_or(0),
        "size": 0,
        "digest": "",
    }))
    .into_response())
}

// 处理聊天完成请求
pub(crate) async fn chat_completions_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    debug!("Chat completion request: {:?}", request);

    // 判断是否为流式请求
    if request.stream {
        // 获取流式响应
        let response = state
            .proxy
            .create_chat_completion(request)
            .await
            .map_err(ApiError::from)?;

        // 检查状态码
        if response.status().is_success() {
            // 转发流式数据
            let stream = response.bytes_stream();
            let proxy_stream = ProxyStream::new(stream);

            // 构建响应
            let response = axum::response::Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(proxy_stream))
                .map_err(|e| ApiError::internal_error(e.to_string()))?;

            Ok(response)
        } else {
            // 处理错误
            let status = StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ApiError::new(status, body))
        }
    } else {
        // 非流式请求
        let response = state
            .proxy
            .create_chat_completion(request)
            .await
            .map_err(ApiError::from)?;

        if response.status().is_success() {
            let json_value = response.json::<Value>().await.map_err(|e| {
                ApiError::internal_error(format!("Failed to parse JSON response: {}", e))
            })?;
            Ok(Json(json_value).into_response())
        } else {
            let status = StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ApiError::new(status, text))
        }
    }
}

// 处理文本补全请求
pub(crate) async fn completions_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    debug!("Completion request: {:?}", request);

    // 与聊天完成处理类似逻辑
    if request.stream {
        let response = state
            .proxy
            .create_completion(request)
            .await
            .map_err(ApiError::from)?;

        if response.status().is_success() {
            let stream = response.bytes_stream();
            let proxy_stream = ProxyStream::new(stream);

            let response = axum::response::Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(proxy_stream))
                .map_err(|e| ApiError::internal_error(e.to_string()))?;

            Ok(response)
        } else {
            let status = StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ApiError::new(status, body))
        }
    } else {
        let response = state
            .proxy
            .create_completion(request)
            .await
            .map_err(ApiError::from)?;

        if response.status().is_success() {
            let json_value = response.json::<Value>().await.map_err(|e| {
                ApiError::internal_error(format!("Failed to parse JSON response: {}", e))
            })?;
            Ok(Json(json_value).into_response())
        } else {
            let status = StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ApiError::new(status, text))
        }
    }
}

// 处理嵌入请求
pub(crate) async fn embeddings_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Response, ApiError> {
    debug!("Embedding request: {:?}", request);

    let response = state
        .proxy
        .create_embedding(request)
        .await
        .map_err(ApiError::from)?;

    // 处理响应
    if response.status().is_success() {
        let json_value = response.json::<Value>().await.map_err(|e| {
            ApiError::internal_error(format!("Failed to parse JSON response: {}", e))
        })?;
        Ok(Json(json_value).into_response())
    } else {
        let status = StatusCode::from_u16(response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        Err(ApiError::new(status, text))
    }
}

// 处理 /api/tags 请求，返回兼容 Ollama 格式的模型列表
pub(crate) async fn tags_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Response, ApiError> {
    let models = state.proxy.get_all_models().await.map_err(ApiError::from)?;
    // 将模型转换为 Ollama API 的格式
    let ollama_models = models
        .data
        .iter()
        .map(|model| openai_model_to_ollama_model(model))
        .collect::<Vec<_>>();

    Ok(Json(json!({
        "models": ollama_models
    }))
    .into_response())
}

// model to ollama model ，将 OpenAI 模型转换为 Ollama 模型
pub(crate) fn openai_model_to_ollama_model(model: &ModelInfo) -> Value {
    let model_name = model.name.clone();
    let model_id = model.id.clone();
    let default_time: &'static str = "2025-05-20T17:12:39.592418003+08:00";
    json!({
    "name": model_name,
    "model": model_id,
    "modified_at": default_time,
    "size": 274302450, // 使用默认大小
            "digest": format!("{:x}", md5::compute(model.id.as_bytes())), // 使用ID的MD5作为唯一标识
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": derive_model_family(&model_name),
                "families": [derive_model_family(&model_name)],
                "parameter_size": "7B", // 默认参数大小
                "quantization_level": "F16"
            }
        })
}

// 根据模型名称推导模型家族
fn derive_model_family(model_name: &str) -> String {
    let lower_name = model_name.to_lowercase();

    if lower_name.contains("gpt") {
        "gpt".to_string()
    } else if lower_name.contains("llama") {
        "llama".to_string()
    } else if lower_name.contains("bert") || lower_name.contains("embed") {
        "nomic-bert".to_string()
    } else if lower_name.contains("mistral") {
        "mistral".to_string()
    } else if lower_name.contains("qwen") {
        "qwen".to_string()
    } else if lower_name.contains("gemma") {
        "gemma".to_string()
    } else {
        // 如果无法识别，使用通用名称
        "unknown".to_string()
    }
}

#[derive(Deserialize)]
pub(crate) struct ShowModelParams {
    model: String,
}

// 处理 /api/show 请求，根据模型名称返回单个模型的详细信息
pub(crate) async fn show_model_handler(
    State(state): State<Arc<ServerState>>,
    Json(params): Json<ShowModelParams>,
) -> Result<Response, ApiError> {
    info!("Show model request for name: {}", params.model);

    // 获取所有模型
    let models = state.proxy.get_all_models().await.map_err(ApiError::from)?;

    // 查找匹配名称的模型
    let model = models
        .data
        .iter()
        .find(|m| m.id.to_lowercase() == params.model.to_lowercase())
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found", params.model),
            )
        })?;

    let default_time: &'static str = "2025-05-20T17:12:39.592418003+08:00";
    let family = derive_model_family(&model.name);

    // 构建 Ollama API 格式的响应
    let response_data = json!({
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": "751.63M", // 默认参数大小
            "quantization_level": "Q4_K_M"
        },
        "model_info": {
            "general.architecture": family,
            "general.basename": model.name,
            "general.file_type": 15,
            "general.license": "apache-2.0",
            "general.parameter_count": 751632384,
            "general.quantization_version": 2,
            "general.size_label": "0.6B",
            "general.type": "model",
            "tokenizer.ggml.add_bos_token": false,
            "tokenizer.ggml.bos_token_id": 151643,
            "tokenizer.ggml.eos_token_id": 151645,
            "tokenizer.ggml.padding_token_id": 151643,
            "tokenizer.ggml.pre": "gpt2",
        },
        "tensors": [{
            "name": "output.weight",
            "type": "Q6_K",
            "shape": [1024, 151936]
        }],
        "capabilities": ["completion", "tools"],
        "modified_at": default_time
    });

    Ok(Json(response_data).into_response())
}
