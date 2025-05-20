# OpenAI to Ollama API 代理
将多个 openai 服务器的 API 代理到 Ollama API，支持多种模型。

优势：
1. localhost 可匿名调用，减少 API 密钥泄露风险。
2. 某些软件仅支持 ollama 形式的自定义 api。
3. 可单独设置代理分流服务器，无需全局代理 （WIP）。

## 配置说明

创建 `config.yaml` 配置文件，格式如下:

```yaml
server:
  host: "127.0.0.1"  # 服务监听地址
  port: 11434        # 服务监听端口(默认与Ollama相同)

openai_servers:
  - url: "https://example.com/v1/"  # OpenAI兼容API的基础URL
    api_key: "sk-your-api-key"      # API密钥
    prefix_id: "服务前缀."           # 可选，为该服务的模型添加前缀
    model_list:              # 可选，指定该服务的模型列表
        - name: "gpt-3.5-turbo"  # 模型名称
          id: "gpt-3.5-turbo"    # 模型ID
    # 可以添加更多服务器...
```    