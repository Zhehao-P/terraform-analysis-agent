# MCP + Qdrant 架构与可行性方案

## 🧱️ 系统架构概览

```mermaid
graph TD
    A[GitHub Repositories] --> B[Processing Script]
    B --> C[Qdrant]
    D[LLM (e.g., Claude, ChatGPT)] --> E[MCP Server]
    E --> C
    C --> E
    E --> D
```

---

## 🧺 组件详解

### 1. 📦 Qdrant 数据库

使用 Qdrant 作为向量数据库，提供高效的向量存储和检索能力。

- **REST API**：`http://localhost:6333`
- **Web UI**：`http://localhost:6333/dashboard`
- **gRPC API**：`localhost:6334`

---

### 2. 🛠️ Repo 处理脚本

编写脚本来处理 GitHub 仓库的文件，并将其存储到 Qdrant 中。每个文档应包含以下元数据：

- `project_name`：项目名称
- `file_url`：文件的原始链接
- `timestamp`：处理时间戳（用于后续更新）
- `type`：文件类型（如 `test`、`src`、`doc`）

示例代码：

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://localhost:6333")

points = [
    PointStruct(
        id=1,
        vector=[...],  # 替换为实际的嵌入向量
        payload={
            "project_name": "example_project",
            "file_url": "https://github.com/example_project/file.py",
            "timestamp": "2025-04-18T12:00:00Z",
            "type": "src"
        }
    ),
    # 添加更多 PointStruct 实例
]

client.upsert(collection_name="example_collection", points=points)
```

请确保在创建集合时，设置的向量维度与嵌入模型生成的向量维度一致（如使用 OpenAI 的 `text-embedding-ada-002` 模型时，维度为 1536）。

---

### 3. 🔌 MCP 服务器配置

MCP（Model Context Protocol）可以作为中间层，处理 LLM 的查询请求，并从 Qdrant 中检索相关上下文。

功能包括：

- 接收来自 LLM 的查询请求
- 根据元数据（如 `type`）进行预过滤
- 执行向量相似度搜索，获取相关文档
- 将检索结果返回给 LLM

---

### 4. 🧠 LLM 查询流程

LLM 查询步骤如下：

1. 发起查询请求，包含查询内容和预过滤条件
2. MCP 进行元数据预过滤（例如仅搜索 `src` 或 `doc`）
3. MCP 对过滤后的集合执行向量相似度搜索
4. 返回结果给 LLM，作为上下文
5. LLM 使用上下文生成高质量回答

---

## ✅ 总结

该方案支持以下功能：

- 使用 Qdrant 存储 GitHub Repo 中的结构化文档
- MCP 提供中间层抽象，供 LLM 动态获取上下文
- 支持元数据过滤和向量查询组合
- 可以扩展到多项目、多类型、增量更新等复杂需求

如需脚本模板或部署示例，可在此基础上打通实现。
