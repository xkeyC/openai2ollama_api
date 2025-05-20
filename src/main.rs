mod config;
mod proxy;
mod server;

use anyhow::Result;
use clap::Parser;
use config::Config;
use env_logger::Env;
use log::{error, info};
use std::path::PathBuf;
use std::sync::Arc;
use proxy::OpenAIProxy;
use server::ServerState;

#[derive(Parser)]
#[clap(name = "openai2ollama", version = env!("CARGO_PKG_VERSION"), about = "OpenAI API to Ollama API proxy")]
struct Cli {
    #[clap(short, long, help = "Path to config file", default_value = "config.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    
    // 解析命令行参数
    let cli = Cli::parse();
    
    // 加载配置
    info!("Loading config from {:?}", cli.config);
    let config = match Config::from_file(&cli.config) {
        Ok(config) => {
            info!("Config loaded successfully");
            Arc::new(config)
        },
        Err(e) => {
            error!("Failed to load config: {}", e);
            return Err(e);
        }
    };
    
    // 创建 OpenAI 代理
    let proxy = Arc::new(OpenAIProxy::new(config.clone()));
    
    // 创建服务器状态
    let state = ServerState { proxy };
    
    // 获取服务器地址
    let addr = format!("{}:{}", config.server.host, config.server.port);
    
    // 运行服务器
    info!("Starting server on {}", addr);
    if let Err(e) = server::run_server(state, &addr).await {
        error!("Server error: {}", e);
        return Err(e);
    }
    
    Ok(())
}
