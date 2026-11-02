# 一页怎么跑

在项目根目录执行。默认使用本机 Ollama 和本地知识库，无需 API key。

## 前置条件

1. 安装 [Ollama](https://ollama.com) 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
2. 启动 Ollama（已在后台运行则跳过），并在另一个终端拉取模型：

   ```bash
   ollama serve
   # 在另一个终端执行：
   ollama pull qwen3.5:9b
   ollama pull bge-m3
   ```

   模型约占 7.8 GB，首次下载和推理需要一定时间。

## 最短路径

```bash
uv sync
uv run python -m agentic_search.ingest sample_docs
uv run python -m agentic_search.chat
```

在 `你:` 提示后输入：

```text
星尘咖啡馆的招牌饮品是什么？
```

等待检索过程和回答，末尾的「本轮引用」列出命中的来源文件；输入 `exit` 退出。
首次索引音视频样本时还会下载 faster-whisper small 语音模型，可能需要较长时间；重跑索引会增量更新。更多媒体依赖及下载方式见 [README](../README.md)。

## 可选功能

- **Web 搜索**：在项目根目录的 `.env` 写入 `TAVILY_API_KEY=你的密钥`，或设置同名环境变量；不要提交密钥。不配置时仅搜索本地知识库。
- **FastAPI + SSE**：先运行 `uv sync --extra server`，再运行 `uv run python -m agentic_search.server`，然后在另一个 Bash 终端执行：

  ```bash
  curl -N -X POST http://127.0.0.1:8080/chat -H 'Content-Type: application/json' -d '{"question":"星尘咖啡馆的招牌饮品是什么？","thread_id":"demo","role":"manager"}'
  ```

  SSE 的 `done` 事件包含来源。Windows PowerShell 请使用 `curl.exe` 并按该 shell 的 JSON 引号规则转义。
- **Docker Compose**：运行 `docker compose up --build`（本机 Ollama 继续运行；索引目录 `chroma_db` 挂载到容器）。容器运行方式尚未在本次演示中验证。

## 常见问题

- **无法连接 Ollama**：确认 `ollama serve` 已启动，默认地址为 `http://127.0.0.1:11434`；如果端口被占用，先检查现有服务。
- **模型不存在**：用 `ollama list` 检查，并重新执行上面的两个 `ollama pull`。
- **没有联网搜索**：未配置 `TAVILY_API_KEY` 时是正常的纯本地模式；仍可查询已索引的本地文档。
- **测试缺少 FastAPI 等依赖**：完整测试前先运行 `uv sync --extra server`，再运行 `uv run pytest -v`。
- **知识库为空**：先在项目根目录运行上面的 ingest 命令，再启动 chat。
