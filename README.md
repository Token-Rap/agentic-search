# agentic-search

## Demo

![本地 RAG 问答演示](docs/assets/demo.gif)

本地检索样本文档，回答问题并展示引用来源。

**[一页怎么跑](docs/QUICKSTART.md)**

LangChain + LangGraph 的 agentic search 引擎:**本地知识库 RAG + Web 搜索双通道,agent 自主路由**。模型自主决定每个问题走本地检索(`retrieve_docs`)、联网搜索(`web_search`)还是直接回答。本地通道全链路离线:Ollama(qwen3.5:9b + bge-m3)+ Chroma,零 API key;Web 通道(Tavily)可选,不配置则自动降级为纯本地 RAG。支持 Markdown / PDF / 音频 / 视频多模态语料、增量索引、BM25+向量混合检索、ACL 权限过滤、语义缓存、检索质量评估。

## 前置条件

1. 安装 [Ollama](https://ollama.com) 并启动:`ollama serve`
2. 拉取模型:`ollama pull qwen3.5:9b && ollama pull bge-m3`(约 6.6GB + 1.2GB;8GB 显存可完整驻留,换更大模型需相应显存)
3. 安装 [uv](https://docs.astral.sh/uv/)
4. (可选)Web 搜索:[tavily.com](https://tavily.com) 免费注册拿 API key(1000 次/月,国内可直连),写入 `.env` 的 `TAVILY_API_KEY`;不配则纯本地模式

## 使用

```bash
uv sync                                           # 首次:建环境(自动下载 Python 3.12)
uv run python -m agentic_search.ingest            # 增量索引 sample_docs(或传任意文档目录)
uv run python -m agentic_search.ingest --rebuild  # 全量重建向量库
uv run python -m agentic_search.chat              # 开始问答,exit 退出
```

- 语料格式:`.md` 直读;`.pdf` 经 pymupdf4llm 归一化;`.wav/.mp3/.m4a` 经 faster-whisper 转写(时间窗分段,引用带时间戳);`.mp4/.mov` 音轨转写 + 关键帧经 qwen3.5 vision 描述
- 双通道路由:配置了 `TAVILY_API_KEY` 后,内部事实类问题模型先查本地库;时效性/公共知识问题(天气、新闻、版本发布)自动走 `web_search`;本地库确认没有时可回退联网补充。回答分别汇总本地引用(文件名)与 Web 引用(URL)

### 服务化(FastAPI + SSE)

```bash
uv sync --extra server                        # 装服务依赖
uv run python -m agentic_search.server        # 起服务(默认 localhost:8080;AGENTIC_SEARCH_HOST/AGENTIC_SEARCH_PORT 配端口)
# SSE 问答(role/collection/thread_id 透传给图;done 事件带 sources 与 web_sources):
curl -N -X POST localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"星尘咖啡馆的招牌饮品是什么?","thread_id":"s1","role":"manager"}'
# 其它端点:GET /health(含 web_search 通道状态)、GET /roles、GET /metrics、POST /ingest
```

全栈(服务 + Prometheus + Grafana):`cp .env.example .env && docker compose up`。
Ollama 留宿主(compose 经 `host.docker.internal` 连);Grafana 面板见 `localhost:3000`,Prometheus 见 `localhost:9090`。

**注**: 容器栈(Dockerfile / docker-compose)已编写并静态验证,但在开发环境中未实际运行过(无 Docker)——建议在支持 Docker 的机器上验证 `docker build` 与 `docker compose up` 可用性。

媒体依赖已归入默认安装的 `media` group(CLI `uv sync` 仍自带);精简 server 镜像以 `--no-group media` 排除。
- 增量索引:块级内容哈希做稳定 ID,只嵌入新增/变更块、清理已消失块;重跑输出"新增 0, 删除 0"
- 权限过滤:语料目录放 `acl.json`(glob → access 级别),`chat --role public|staff|manager` 决定检索可见范围(向量与 BM25 双通道过滤)
- 试试问:"星尘咖啡馆的招牌饮品是什么?"(本地库)、"供应商 NX-42 供应什么?"(PDF,编号词面命中靠 BM25)、"例会决定新店选址在哪?"(音频)、"SP-2026 评审的是什么新品?"(视频画面)、"LangGraph 最新版本有什么变化?"(时效性 → web_search)
- 首次带媒体语料 ingest 需要 faster-whisper small 模型(约 460MB)。国内网络推荐从 ModelScope 直接下载到本地目录(实测比 hf-mirror 快若干量级),`media` 模块会优先使用:

  ```bash
  mkdir -p models/faster-whisper-small && cd models/faster-whisper-small
  for f in config.json tokenizer.json vocabulary.txt model.bin; do
    curl -sLO "https://modelscope.cn/models/gpustack/faster-whisper-small/resolve/master/$f"
  done
  ```

## 架构

- `parsers`:类型路由把多格式文档归一化为 markdown(文本是索引,原始文件是档案)
- `media`:音频 ASR 时间窗分段、视频关键帧 + VLM 描述——时间轴 locator 复用 headers 字段,引用格式零改动
- `acl`:acl.json glob 规则给块打 access 标,检索按角色可见范围过滤
- `llm`:后端工厂(`make_chat_llm` / `make_embeddings`),生成/嵌入客户端的唯一构造点;下游只依赖 LangChain 抽象(`BaseChatModel` / `Embeddings`),`AGENTIC_SEARCH_BACKEND` 环境变量一键切换 Ollama ↔ vLLM
- `search`:Web 搜索后端工厂(`make_search_backend`),对齐 `llm` 工厂模式;`SearchBackend` 协议 + Tavily 实现(httpx 直调 REST),换供应商只加一个实现类;`TAVILY_API_KEY` 缺失返回 None → 装配点不绑定 web_search,自动降级纯 RAG
- `ingest`:解析/转写 → 切块 → bge-m3 向量化 → Chroma 增量同步
- `retrieval`:jieba 分词 BM25 + 向量检索,RRF 融合(编号/专名词面命中兜底向量盲区)+ ACL 双通道过滤
- `graph`:手写 LangGraph StateGraph,agent 节点(qwen3.5:9b)⇄ ToolNode 循环;双工具自主路由——模型自主决定调 `retrieve_docs` 还是 `web_search`、调几次、何时直接回答;system prompt 随工具集切换(纯 RAG 版 / 双通道版)
- `cache`:QA 语义缓存(独立 collection,cosine 阈值命中)——**不进检索池**(模型生成内容不能无审核地成为检索来源);失效复用块内容哈希(源文档一变即作废);带 ACL 检查防跨角色泄漏;**用过 web_search 的回答不写缓存**(时效性内容无法靠块哈希失效);命中计数供 FAQ 沉淀筛选
- `faq`:`python -m agentic_search.faq` 导出高频问答候选 → 人工审核 → 放入 `sample_docs/faq.md` 重跑 ingest 入库(切块/哈希/ACL 全复用)——这是"模型内容不得无审核入库"边界的合规出口
- `evals`:golden set 检索评估,`uv run python -m agentic_search.evals` 输出纯向量 vs 混合的 hit@k / MRR;**每次 ingest 默认语料目录后自动跑一次**
- `chat`:CLI 流式问答,实时展示检索(🔍)与联网搜索(🌐)过程,回答后分别汇总本地/Web 引用;缓存命中显示 ⚡ 并秒回;对话历史在 LLM 视图层裁剪(最近 `HISTORY_KEEP_TURNS` 轮完整,更早轮次仅留问答对),`num_ctx` 提升至 16K

## 推理后端(可切换:Ollama / vLLM)

所有 LLM 与嵌入客户端统一经 `agentic_search.llm` 工厂构造,下游只依赖 LangChain 抽象类型。切换后端是**改一个环境变量**,不动业务代码:

```bash
export AGENTIC_SEARCH_BACKEND=vllm   # 默认 ollama
```

默认 Ollama 单进程同时服务生成 + 嵌入,零配置、低显存友好(可溢出到 CPU),适合单机 demo。切到 vLLM 面向**高并发吞吐**(PagedAttention + continuous batching)——但其价值只在并发负载下兑现,单用户场景 vLLM ≈ Ollama。

### 切到 vLLM 的步骤

1. 装可选依赖:`uv sync --extra vllm`(拉 `langchain-openai`)
2. 起两个 vLLM 实例(vLLM 一个服务只服务一个模型):

   ```bash
   # 生成:后两个 flag 是承重点,整个 agent 图靠模型自主发 tool_call
   vllm serve Qwen2.5-7B-Instruct-AWQ \
     --port 8000 --max-model-len 8192 \
     --enable-auto-tool-choice --tool-call-parser hermes
   vllm serve BAAI/bge-m3 --port 8001 --task embed          # 嵌入:独立实例
   ```

3. 指端点并切后端:

   ```bash
   export AGENTIC_SEARCH_BACKEND=vllm
   export VLLM_BASE_URL=http://localhost:8000/v1
   export VLLM_EMBED_BASE_URL=http://localhost:8001/v1
   export GENERATION_MODEL=Qwen2.5-7B-Instruct-AWQ
   ```

### 参数映射(Ollama → vLLM 非 1:1,工厂已内置适配)

| Ollama 客户端参数 | vLLM 侧对应 |
|---|---|
| `num_ctx=8192` | 无客户端参数,启动 `--max-model-len 8192` |
| `reasoning=False`(关思考段) | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` |
| `bind_tools`(工具调用) | 启动 `--enable-auto-tool-choice --tool-call-parser hermes`;**Qwen 系 parser 稳定性需实测**,不发 tool_call 则整条 agentic 链失效 |
| 单进程双模型 | 生成、嵌入各起一个 vLLM 实例 |

### 显存与硬件

vLLM 激进预分配 KV cache、要求模型整个进显存、不像 Ollama 优雅溢出到 CPU。**8GB 显存(如 RTX 4060)跑 9B + bge-m3 双实例大概率 OOM**;要演示并发吞吐优势需 16–24GB+(L4 / A10 / A100)或 4-bit 量化模型(如上 `-AWQ`)。

## 规模实验(21,838 块实测)

MDN 中文文档(1,921 文件)实测:嵌入 69 块/s;检索 hit@5 双通道 100%,p50 213ms;BM25 持久化索引启动加载 0.20s(vs 现场重建 8.9s)。完整数字与结论见 `docs/benchmarks/2026-07-11-scale-test.md`。多语料共存:`ingest <目录> --collection <名>`,chat/evals 同参。

## 工程决策记录

- **Web 搜索选 Tavily 而非 SearXNG/DDG**:Tavily 返回已清洗正文片段(专为 RAG 设计)、国内可直连、免费档够 demo;SearXNG 需 Docker 且上游引擎国内连通性差,ddgs 无正文且非官方接口易限流。代价是引入一个可选 API key——经 `search` 工厂抽象隔离,本地通道零 key 卖点不受影响,换供应商只加一个 `SearchBackend` 实现。
- **web 内容不进缓存、不进检索池**:语义缓存的失效机制靠本地块哈希(源文档一变即作废),管不了 web 内容的时效性——"今天天气"缓存了就是错的;检索池同理(与"模型内容不得无审核入库"同一条边界)。实现为写侧 gate:本轮用过 `web_search` 即跳过缓存写入,读侧不变。
- **路由不设显式分类节点**:双工具直接交给模型自主选择(agent ⇄ ToolNode 循环零改动),而非前置一个 LLM 分类器——少一次调用延迟,且模型可在循环中途纠偏(本地没查到再转 web);代价是路由质量依赖模型能力,由 system prompt 规则约束。
- **暂不上重排序(reranker)**:当前语料规模下评估显示纯向量与混合检索均 hit@5=100%、MRR=1.0——排序不是瓶颈,跨编码器还需 ~2.5GB torch 依赖。待语料上规模、评估分数出现分化时再引入(评估集已就位,决策有数据依据)。
- **语义缓存与知识库严格分离**:自动回写模型答案到检索池会造成幻觉自我固化(错误答案被后续检索引用,越滚越"可信")。缓存只做同问复用,三重防线:独立 collection、哈希失效、ACL 检查;沉淀入库的唯一通道是 FAQ 人工审核工作流。
- **媒体解析缓存**:ASR/VLM 属采样式解析,同一文件重复解析文本会微变,导致块哈希变化、增量索引被无意义搅动且重复付模型开销——解析结果按文件内容哈希缓存(`.media_cache/`),文件不变不重解析。
- **历史裁剪在 LLM 视图层而非状态层**:checkpointer 保留完整历史(可审计、可回放),只在拼 prompt 时裁剪;旧轮次的检索块已被回答蒸馏,成对裁掉 tool_calls/ToolMessage 损失最小且保证消息序列合法。
- **多租户延后**:collection-per-tenant 是纯配置管道,单机 demo 无第二租户场景;需要时按"每租户独立 collection + 独立缓存 + 租户级 ACL 根规则"落地即可。

- **混合检索按查询特征路由(已实现)**:规模实验发现 BM25 的价值依语料反转——稀有实体词上是召回救星,术语密集语料的常见词查询上反而注噪(MRR -0.011)。现按 df 自适应判据路由(查询含 `0 < df ≤ max(3, ⌈0.5%·N⌉)` 的词元才开 BM25 通道),复测两套语料混合检索**处处不劣于纯向量**;判据随语料规模自适应,零硬编码模式。

- **服务层加固延后(带标签的决策记录)**:当前为作品级单进程单 worker,内存态 `MemorySaver` 对单进程正确。以下按需补上,非静默省略:多 worker + 共享/持久 checkpointer(SqliteSaver/Redis);ingest 任务队列 + 状态轮询(大语料同步 ingest 会阻塞,demo 语料无需);Prometheus 多进程模式(`PROMETHEUS_MULTIPROC_DIR`,多 worker 才需);鉴权/限流;k8s manifests(本地 kind/minikube 跑通即可)。
- **可观测性边界埋点**:指标与审计日志只在 server 边界采集(缓存命中在调用点计量、检索经 `MeteredRetriever`、Web 搜索经 `MeteredSearchBackend` 记录成功/失败计数),核心检索/搜索/缓存模块零改动——延续「不动业务代码」边界。

## 企业演进路线

**已落地(Phase 1-4)**: FastAPI + SSE 流式服务化 ✅、多阶段容器化 + docker-compose ✅、Prometheus/Grafana 检索观测面板(QPS/p95/缓存命中率/路由分布) + structlog 审计日志 ✅、**本地 RAG + Web 搜索双通道 agentic 路由** ✅。设计与实现记录:`docs/superpowers/specs/` 与 `docs/superpowers/plans/`。

后续候选:多租户、答案级评估(LLM judge)、路由质量评估(golden set 标注期望通道,统计模型路由准确率)、受限角色的每角色 BM25 索引、**并发压测**(locust QPS/p95 曲线,对比 Ollama 串行 vs vLLM continuous batching;后端切换点已就位于 `agentic_search.llm`)。

## 测试

```bash
uv run pytest -v   # 全部单测不依赖 Ollama 与网络(Tavily 经注入 fake 测试)
```
