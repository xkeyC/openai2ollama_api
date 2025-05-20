use crate::config::{Config, OpenAIServerConfig};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::error;
use reqwest::{header, Client, Response as ReqwestResponse, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use once_cell::sync::Lazy;
use tokio::sync::RwLock;
use bytes::Bytes;
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
// 引入 futures::StreamExt，用于流处理
use futures::StreamExt;

// 模型缓存，避免重复请求模型列表
static MODEL_CACHE: Lazy<RwLock<HashMap<String, Vec<ModelInfo>>>> = Lazy::new(|| {
    RwLock::new(HashMap::new())
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default = "default_name_from_id")]
    pub name: String,
    pub created: Option<i64>,
}

// 当name字段不存在时，使用id作为默认值
fn default_name_from_id() -> String {
    String::new() // 这个实际上不会被调用，因为我们会在反序列化后手动处理
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelInfo>,
    #[serde(default = "default_object")]
    pub object: String,
}

fn default_object() -> String {
    "list".to_string() // 当object字段不存在时，默认使用"list"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Value,
}

#[async_trait]
pub trait OpenAIClient: Send + Sync {
    async fn get_models(&self) -> Result<ModelsResponse>;
    async fn get_model(&self, model_id: &str) -> Result<ModelInfo>;
    async fn create_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ReqwestResponse>;
    async fn create_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<ReqwestResponse>;
    async fn create_embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<ReqwestResponse>;
}

pub struct OpenAIProxy {
    #[allow(dead_code)]
    config: Arc<Config>,
    clients: Vec<(OpenAIClientImpl, OpenAIServerConfig)>,
}

struct OpenAIClientImpl {
    client: Client,
    base_url: String,
    api_key: String,
}

impl OpenAIProxy {
    pub fn new(config: Arc<Config>) -> Self {
        let mut clients = Vec::new();
        
        for server_config in &config.openai_servers {
            let client = Client::builder()
                .build()
                .unwrap();
            
            let base_url = server_config.url.clone();
            let api_key = server_config.api_key.clone();
            
            clients.push((
                OpenAIClientImpl {
                    client,
                    base_url,
                    api_key,
                },
                server_config.clone(),
            ));
        }
        
        Self {
            config,
            clients,
        }
    }
    
    pub async fn get_all_models(&self) -> Result<ModelsResponse> {
        let mut all_models = Vec::new();
        
        for (client, server_config) in &self.clients {
            // 如果配置了模型列表，直接使用
            if let Some(model_list) = &server_config.model_list {
                for model in model_list {
                    let prefixed_id = if let Some(prefix) = &server_config.prefix_id {
                        format!("{}/{}", prefix, model.id)
                    } else {
                        model.id.clone()
                    };
                    
                    all_models.push(ModelInfo {
                        id: prefixed_id,
                        name: model.name.clone(),
                        created: Some(0), // 默认创建时间
                    });
                }
            } else {
                // 否则，从服务器获取模型列表
                match client.get_models().await {
                    Ok(models) => {
                        for mut model in models.data {
                            // 如果指定了前缀，添加前缀到id而不是name
                            if let Some(prefix) = &server_config.prefix_id {
                                model.id = format!("{}/{}", prefix, model.id);
                            }
                            all_models.push(model);
                        }
                    }
                    Err(e) => {
                        error!("Failed to fetch models from {}: {}", client.base_url, e);
                    }
                }
            }
        }
        
        Ok(ModelsResponse {
            data: all_models,
            object: "list".to_string(), // 始终使用"list"作为默认值
        })
    }
    
    pub async fn get_model(&self, model_name: &str) -> Result<ModelInfo> {
        // 解析模型名称，查找匹配的客户端
        let (client_idx, actual_model_id) = self.resolve_model_name(model_name)?;
        let (client, _) = &self.clients[client_idx];
        
        client.get_model(&actual_model_id).await
    }
    
    pub async fn create_chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ReqwestResponse> {
        // 解析模型名称，查找匹配的客户端
        let (client_idx, actual_model_id) = self.resolve_model_name(&request.model)?;
        let (client, _) = &self.clients[client_idx];
        
        // 使用实际模型ID替换请求中的模型名称
        request.model = actual_model_id;
        
        client.create_chat_completion(request).await
    }
    
    pub async fn create_completion(
        &self,
        mut request: CompletionRequest,
    ) -> Result<ReqwestResponse> {
        // 解析模型名称，查找匹配的客户端
        let (client_idx, actual_model_id) = self.resolve_model_name(&request.model)?;
        let (client, _) = &self.clients[client_idx];
        
        // 使用实际模型ID替换请求中的模型名称
        request.model = actual_model_id;
        
        client.create_completion(request).await
    }
    
    pub async fn create_embedding(
        &self,
        mut request: EmbeddingRequest,
    ) -> Result<ReqwestResponse> {
        // 解析模型名称，查找匹配的客户端
        let (client_idx, actual_model_id) = self.resolve_model_name(&request.model)?;
        let (client, _) = &self.clients[client_idx];
        
        // 使用实际模型ID替换请求中的模型名称
        request.model = actual_model_id;
        
        client.create_embedding(request).await
    }
    
    // 解析模型名称，返回匹配的客户端索引和实际模型ID
    fn resolve_model_name(&self, model_name: &str) -> Result<(usize, String)> {
        // 首先检查是否包含前缀
        if let Some((prefix, name)) = model_name.split_once('/') {
            // 寻找匹配前缀的服务器
            for (idx, (_, server_config)) in self.clients.iter().enumerate() {
                if let Some(server_prefix) = &server_config.prefix_id {
                    if server_prefix == prefix {
                        // 如果配置了模型列表，检查模型是否存在
                        if let Some(model_list) = &server_config.model_list {
                            if let Some(model) = model_list.iter().find(|m| m.id == name || m.name == name) {
                                return Ok((idx, model.id.clone()));
                            }
                        }
                        // 否则直接返回模型名
                        return Ok((idx, name.to_string()));
                    }
                }
            }
        }
        
        // 如果没有前缀，尝试在所有没有前缀的服务器中寻找
        for (idx, (_, server_config)) in self.clients.iter().enumerate() {
            if server_config.prefix_id.is_none() {
                // 如果配置了模型列表，检查模型是否存在
                if let Some(model_list) = &server_config.model_list {
                    if let Some(model) = model_list.iter().find(|m| m.id == model_name || m.name == model_name) {
                        return Ok((idx, model.id.clone()));
                    }
                } else {
                    // 没有模型列表，直接使用模型名称
                    return Ok((idx, model_name.to_string()));
                }
            }
        }
        
        // 如果都没找到，返回错误
        Err(anyhow!("Model '{}' not found in any configured server", model_name))
    }
}

#[async_trait]
impl OpenAIClient for OpenAIClientImpl {
    async fn get_models(&self) -> Result<ModelsResponse> {
        // 检查缓存
        {
            let cache = MODEL_CACHE.read().await;
            if let Some(models) = cache.get(&self.base_url) {
                return Ok(ModelsResponse {
                    data: models.clone(),
                    object: "list".to_string(),
                });
            }
        }
        
        let url = format!("{}v1/models", self.base_url);
        
        let response = self
            .client
            .get(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .send()
            .await?;
        
        if response.status() != StatusCode::OK {
            let error_text = response.text().await?;
            return Err(anyhow!("Failed to get models: {}", error_text));
        }
        
        let mut models: ModelsResponse = response.json().await?;
        
        // 处理没有name的情况，将id赋值给name
        for model in &mut models.data {
            if model.name.is_empty() {
                model.name = model.id.clone();
            }
        }
        
        // 确保object字段有值
        if models.object.is_empty() {
            models.object = "list".to_string();
        }
        
        // 更新缓存
        {
            let mut cache = MODEL_CACHE.write().await;
            cache.insert(self.base_url.clone(), models.data.clone());
        }
        
        Ok(models)
    }
    
    async fn get_model(&self, model_id: &str) -> Result<ModelInfo> {
        let url = format!("{}v1/models/{}", self.base_url, model_id);
        
        let response = self
            .client
            .get(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .send()
            .await?;
        
        if response.status() != StatusCode::OK {
            let error_text = response.text().await?;
            return Err(anyhow!("Failed to get model {}: {}", model_id, error_text));
        }
        
        let mut model: ModelInfo = response.json().await?;
        
        // 处理没有name的情况，将id赋值给name
        if model.name.is_empty() {
            model.name = model.id.clone();
        }
        
        Ok(model)
    }
    
    async fn create_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ReqwestResponse> {
        let url = format!("{}v1/chat/completions", self.base_url);
        
        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&request)
            .send()
            .await?;
        
        Ok(response)
    }
    
    async fn create_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<ReqwestResponse> {
        let url = format!("{}v1/completions", self.base_url);
        
        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&request)
            .send()
            .await?;
        
        Ok(response)
    }
    
    async fn create_embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<ReqwestResponse> {
        let url = format!("{}v1/embeddings", self.base_url);
        
        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&request)
            .send()
            .await?;
        
        Ok(response)
    }
}

// 用于转发流式响应的结构
pub struct ProxyStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
}

impl ProxyStream {
    pub fn new(stream: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(stream),
        }
    }
}

impl Stream for ProxyStream {
    type Item = Result<Bytes, Box<dyn std::error::Error + Send + Sync + 'static>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(Box::new(err)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}
