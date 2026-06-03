# 🤖 智扫通 — 扫地机器人智能客服系统

> 基于 ReAct Agent + RAG 的智能问答系统，为扫地机器人用户提供 7×24 小时智能客服服务。

---

## 📋 项目描述

**智扫通**是一个面向扫地机器人/扫拖一体机用户的智能客服系统，采用 **ReAct（Reasoning + Acting）Agent 架构** 与 **RAG（检索增强生成）技术** 构建。

系统以通义千问（Qwen3-max）大语言模型为核心引擎，配合 LangChain Agent 框架的 ReAct 推理能力与 Chroma 向量数据库的知识检索能力，实现了：

- 基于专业知识库的智能问答（选购指南、使用技巧、维护保养、故障排除）
- 个性化的使用报告自动生成
- 结合环境和用户信息的智能建议
- 动态提示词切换（通用问答 ↔ 报告生成）
- 完整的工具调用链与中间件监控体系

> **技术栈**：Python · LangChain · Streamlit · Chroma DB · Qwen3-max · DashScope Embeddings · LangGraph

---

## 🎯 功能模块

### 1. 智能问答（RAG 检索 + 生成）
| 功能 | 说明 |
|------|------|
| 🔍 知识库检索 | 基于 Chroma 向量数据库，从「扫地机器人100问」「选购指南」「维护保养」「故障排除」等文档中精准检索 |
| 📝 摘要生成 | 将检索到的参考资料结合用户提问，由 LLM 生成简洁准确的回答 |
| 💡 覆盖场景 | 产品选购、使用技巧、故障处理、环境适配、维护保养等全场景 |

### 2. ReAct 智能 Agent
| 功能 | 说明 |
|------|------|
| 🧠 自主推理 | 严格遵循「思考→行动→观察→再思考」的 ReAct 推理闭环 |
| 🔧 工具调用 | 支持 7 种工具，按需自主决策调用 |
| 🔄 多步推理 | 最多 5 轮工具调用，信息不足时自主发起二次检索 |

### 3. 报告生成
| 功能 | 说明 |
|------|------|
| 📊 使用报告 | 根据用户 ID 和月份，自动生成个性化扫地机器人使用报告 |
| 📋 结构化输出 | 包含清洁效率、耗材状态、使用对比等核心数据维度 |
| 💡 建议生成 | 基于使用数据，给出针对性的保养和使用建议 |

### 4. 环境感知
| 功能 | 说明 |
|------|------|
| 🌤️ 天气查询 | 获取城市实时天气、湿度、降雨概率，辅助使用建议 |
| 📍 位置定位 | 自动获取用户所在城市，提供本地化建议 |

### 5. 系统管理
| 功能 | 说明 |
|------|------|
| 🖥️ 交互界面 | Streamlit 构建的现代化 Web UI，支持流式输出 |
| 📜 对话历史 | 完整记录用户与 AI 的对话上下文 |
| 🗂️ 知识库管理 | 支持 TXT/PDF 文档导入，MD5 去重自动加载 |
| 📝 日志监控 | 完整的工具调用日志与模型请求日志 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│           (聊天界面 · 侧边栏 · 状态显示)              │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                  ReactAgent                          │
│     ┌──────────────────────────────────────────┐     │
│     │            LangChain Agent                │     │
│     │     (ReAct 推理引擎 · LangGraph)          │     │
│     └──────────────────────────────────────────┘     │
│                         │                            │
│    ┌────────────────────┼────────────────────┐      │
│    │        工具层      │    中间件层         │      │
│    ├────────────────────┼────────────────────┤      │
│    │ rag_summarize      │ monitor_tool       │      │
│    │ get_weather        │ log_before_model   │      │
│    │ get_user_location  │ report_prompt_     │      │
│    │ get_user_id        │ switch             │      │
│    │ get_current_month  │                    │      │
│    │ fetch_external_data│                    │      │
│    │ fill_context_for_  │                    │      │
│    │ report             │                    │      │
│    └────────────────────┴────────────────────┘      │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                    RAG 服务层                         │
│  ┌─────────────────┐  ┌────────────────────────┐   │
│  │  VectorStore     │  │  RagSummarizeService   │   │
│  │  (Chroma)        │  │  (检索+生成链)         │   │
│  └────────┬────────┘  └───────────┬────────────┘   │
└───────────┼───────────────────────┼────────────────┘
            │                       │
┌───────────▼───────────────────────▼────────────────┐
│              基础设施层                              │
│  ┌────────┐  ┌─────────┐  ┌────────┐  ┌────────┐  │
│  │ Qwen3  │  │DashScope│  │ Config │  │Logger  │  │
│  │ -max   │  │Embedding│  │ (YAML) │  │ Handler│  │
│  │(LLM)   │  │(向量化)  │  │        │  │        │  │
│  └────────┘  └─────────┘  └────────┘  └────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 关键技术实现

### 1. ReAct Agent 推理引擎

基于 LangChain Agent 框架 + LangGraph 运行时，实现了完整的 ReAct 推理闭环。

```
核心流程：
  用户输入 → 思考（分析需求） → 判断是否需要工具
    → 否：直接回答 → 输出
    → 是：调用工具 → 观察结果 → 再次思考
      → 信息足够：整合回答 → 输出
      → 信息不足：再调用工具（最多 5 轮）→ 输出
```

**代码位置**：[agent/react_agent.py](agent/react_agent.py)

```python
# 核心 Agent 构建（简化）
self.agent = create_agent(
    model=chat_model,
    system_prompt=load_system_prompts(),
    tools=[rag_summarize, get_weather, ...],
    middleware=[monitor_tool, log_before_model, report_prompt_switch],
)
```

### 2. RAG 检索增强生成

采用 **Chroma 向量数据库** + **DashScope Embeddings (text-embedding-v4)** 实现知识库的向量化存储与语义检索。

| 组件 | 配置 |
|------|------|
| 向量数据库 | Chroma（collection: `agent`） |
| 嵌入模型 | `text-embedding-v4`（通义千问） |
| 文档分片 | RecursiveCharacterTextSplitter（chunk_size=200, overlap=20） |
| 检索策略 | 相似度检索（top-k=3） |
| 支持格式 | TXT、PDF |

**关键特性**：
- MD5 去重机制：避免重复加载同一文档
- 多分隔符分片：支持中文（。！？）和英文（.!?）标点
- 检索+生成链：`PromptTemplate → LLM → StrOutputParser`

**代码位置**：[rag/vector_store.py](rag/vector_store.py) · [rag/rag_service.py](rag/rag_service.py)

### 3. 动态提示词切换（Dynamic Prompt Switching）

通过 **LangChain Middleware** 机制，根据上下文场景动态切换系统提示词。

```python
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()    # 切换到报告生成提示词
    return load_system_prompts()        # 使用通用问答提示词
```

**触发流程**：
1. Agent 识别用户意图为「报告生成」
2. 调用 `fill_context_for_report` 工具
3. 中间件 `monitor_tool` 捕捉该调用，设置 `context["report"] = True`
4. `report_prompt_switch` 中间件检测到标记，切换提示词
5. 后续所有模型调用均使用报告生成提示词

**代码位置**：[agent/tools/middleware.py](agent/tools/middleware.py)

### 4. 工具调用中间件体系

三层中间件贯穿 Agent 执行全流程：

| 中间件 | 类型 | 职责 |
|--------|------|------|
| `monitor_tool` | 工具包装器 | 记录工具调用日志、注入报告上下文标记 |
| `log_before_model` | 模型前置钩子 | 每次模型调用前记录日志 |
| `report_prompt_switch` | 动态提示词 | 根据上下文切换系统提示词 |

### 5. 模型工厂模式

采用 **抽象工厂模式（Factory Pattern）** 统一管理模型实例：

```python
class ChatModelFactory(BaseModelFactory):
    def generator(self):
        return ChatTongyi(model="qwen3-max")        # 大语言模型

class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        return DashScopeEmbeddings(model="text-embedding-v4")  # 嵌入模型
```

**代码位置**：[model/factory.py](model/factory.py)

### 6. YAML 配置驱动

所有配置项通过 YAML 文件集中管理，模块化加载：

| 配置文件 | 管理内容 |
|----------|----------|
| `config/rag.yml` | 模型名称（qwen3-max, text-embedding-v4） |
| `config/chroma.yml` | 向量库参数（chunk_size, k, 分片策略） |
| `config/prompts.yml` | 提示词文件路径 |
| `config/agent.yml` | Agent 参数（外部数据路径） |

### 7. 外部数据集成

支持从 CSV 文件中读取用户的扫地机器人使用记录（清洁效率、耗材状态、使用对比等），为报告生成提供数据支撑。数据按 `{user_id → {month → {特征, 效率, 耗材, 对比}}}` 结构组织。

### 8. 流式交互 UI

基于 **Streamlit** 构建的现代化聊天界面：
- 流式输出（`st.write_stream`），实时展示 AI 回复
- 快速提问按钮（6 个预置问题） 
- 对话历史记录
- 系统状态指示（在线/就绪）
- 清空对话功能

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- DashScope（通义千问）API 密钥

### 安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
echo "DASHSCOPE_API_KEY=your-api-key" > .env

# 3. 加载知识库（首次运行）
python -m rag.vector_store

# 4. 启动应用
streamlit run app.py
```

### 知识库文件

系统支持 Txt/PDF 格式的知识文档，默认加载目录 `data/`：
| 文件 | 内容 |
|------|------|
| `data/扫地机器人100问2.txt` | 常见问题 FAQ |
| `data/扫拖一体机器人100问.txt` | 扫拖一体机 FAQ |
| `data/选购指南.txt` | 产品选购建议 |
| `data/维护保养.txt` | 保养维护知识 |
| `data/故障排除.txt` | 故障处理方法 |

---

## 📁 项目结构

```
├── app.py                    # Streamlit 应用主入口
├── agent/
│   ├── react_agent.py        # ReAct Agent 定义
│   └── tools/
│       ├── agent_tools.py    # 工具函数定义
│       └── middleware.py     # 中间件（监控、日志、提示词切换）
├── rag/
│   ├── rag_service.py        # RAG 检索+生成服务
│   └── vector_store.py       # Chroma 向量库管理
├── model/
│   └── factory.py            # 模型工厂（LLM + Embedding）
├── config/
│   ├── rag.yml               # RAG 模型配置
│   ├── chroma.yml            # 向量库参数
│   ├── prompts.yml           # 提示词路径配置
│   └── agent.yml             # Agent 配置
├── prompts/
│   ├── main_prompt.txt       # 通用问答系统提示词
│   ├── rag_summarize.txt     # RAG 摘要提示词
│   └── report_prompt.txt     # 报告生成提示词
├── data/                     # 知识库文件
├── utils/
│   ├── config_handler.py     # YAML 配置加载
│   ├── prompt_loader.py      # 提示词文件加载
│   ├── file_handler.py       # 文件处理（TXT/PDF 读取）
│   ├── path_tool.py          # 路径工具
│   └── logger_handler.py     # 日志处理
└── chroma_db/                # Chroma 持久化存储目录
```

---

## ⚙️ 配置详解

### RAG 配置（`config/rag.yml`）
```yaml
chat_model_name: qwen3-max          # 大语言模型
embedding_model_name: text-embedding-v4  # 向量嵌入模型
```

### Chroma 配置（`config/chroma.yml`）
```yaml
collection_name: agent              # 集合名称
persist_directory: chroma_db        # 持久化路径
k: 3                                # 检索返回文档数
chunk_size: 200                     # 文档分片大小
chunk_overlap: 20                   # 分片重叠数
```

---

## 🧪 核心业务流程示例

### 场景 1：普通问答
```
用户：小户型适合哪些扫地机器人？
Agent 思考 → 调用 rag_summarize("小户型 扫地机器人 推荐")
    → 返回参考资料 → 整合回答
    → "对于小户型（60-90㎡），推荐以下扫地机器人..."
```

### 场景 2：报告生成
```
用户：生成我的使用报告
Agent 思考 → 调用 get_user_id() → "1001"
    → 调用 get_current_month() → "2025-06"
    → 调用 fill_context_for_report() → 触发提示词切换
    → 调用 fetch_external_data("1001", "2025-06")
    → 使用报告提示词生成完整报告
```

### 场景 3：天气+环境建议
```
用户：今天深圳适合用扫地机器人吗？
Agent 思考 → 调用 get_weather("深圳")
    → 返回天气信息
    → 结合 rag_summarize 获取湿度使用建议
    → 整合回答："今天深圳天气晴朗，湿度50%，非常适合使用扫地机器人..."
```

---

## 📄 许可证

MIT License © 2026 智扫通
