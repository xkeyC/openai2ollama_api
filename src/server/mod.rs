use crate::proxy::OpenAIProxy;
use anyhow::Result;
use axum::{
    http::{Request, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use futures::future::BoxFuture;
use log::info;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tower::{Layer, Service};
use tower_http::cors::CorsLayer;

mod ollama;

// 服务器状态
#[derive(Clone)]
pub struct ServerState {
    pub proxy: Arc<OpenAIProxy>,
}

// 请求日志中间件
#[derive(Clone)]
struct RequestLogger<S> {
    inner: S,
}

impl<S> RequestLogger<S> {
    fn new(service: S) -> Self {
        Self { inner: service }
    }
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for RequestLogger<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<ReqBody>) -> Self::Future {
        // 克隆服务和请求信息，以便在异步闭包中使用
        let mut inner = self.inner.clone();
        let method = request.method().clone();
        let uri = request.uri().clone();
        let version = request.version();
        let headers = request.headers().clone();

        let start_time = Instant::now();

        // 在请求开始时记录信息
        info!("=> 请求开始: {} {} {:?} {:?}", method, uri, version, headers);

        // 调用内部服务处理请求
        let future = Box::pin(async move {
            let response = inner.call(request).await?;

            // 在请求结束时记录响应信息
            let status = response.status();
            let duration = start_time.elapsed();
            info!(
                "<= 请求完成: {} {} - {} ({:.2}ms)",
                method,
                uri,
                status,
                duration.as_secs_f64() * 1000.0
            );

            Ok(response)
        });

        future
    }
}

// 请求日志中间件层
#[derive(Clone)]
struct RequestLoggerLayer;

impl<S> Layer<S> for RequestLoggerLayer {
    type Service = RequestLogger<S>;

    fn layer(&self, service: S) -> Self::Service {
        RequestLogger::new(service)
    }
}

// 实现 Ollama API
pub async fn run_server(state: ServerState, addr: &str) -> Result<()> {
    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/", get(ollama::root_handler))
        .route("/v1/models", get(ollama::list_models_handler))
        .route("/v1/models/:model_id", get(ollama::get_model_handler))
        .route("/v1/chat/completions", post(ollama::chat_completions_handler))
        .route("/v1/completions", post(ollama::completions_handler))
        .route("/v1/embeddings", post(ollama::embeddings_handler))
        .route("/api/tags", get(ollama::tags_handler))
        .route("/api/show", post(ollama::show_model_handler))
        .layer(cors)
        .layer(RequestLoggerLayer)
        .with_state(Arc::new(state));

    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// API 错误处理
pub struct ApiError {
    status_code: StatusCode,
    message: String,
}

impl ApiError {
    pub fn new(status_code: StatusCode, message: String) -> Self {
        Self {
            status_code,
            message,
        }
    }

    pub fn internal_error(message: String) -> Self {
        Self {
            status_code: StatusCode::INTERNAL_SERVER_ERROR,
            message,
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            status_code: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
                "code": self.status_code.as_u16(),
            }
        }));
        (self.status_code, body).into_response()
    }
}
