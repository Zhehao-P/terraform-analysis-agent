# MCP + ChromaDB 架构与可行性方案

## 🧱 系统架构概览

```mermaid
graph TD
    A[GitHub Repositories] --> B[Processing Script]
    B --> C[ChromaDB via Docker]
    D[LLM (e.g., Claude, ChatGPT)] --> E[MCP Server]
    E --> C
    C --> E
    E --> D
```

---

## 🧩 组件详解

### 1. 📦 ChromaDB 部署（Docker）

使用 Docker 部署 ChromaDB，确保数据持久化和服务的稳定运行。

```bash
docker run -d --rm --name chromadb \
  -p 8000:8000 \
  -v ./chroma_data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=TRUE \
  chromadb/chroma
```

上述命令将 ChromaDB 部署在本地的 8000 端口，并将数据持久化到 `./chroma_data` 目录中。

---

### 2. 🛠️ Repo 处理脚本

编写脚本来处理 GitHub 仓库的文件，并将其存储到 ChromaDB 中。每个文档应包含以下元数据：

- `project_name`：项目名称
- `file_url`：文件的原始链接
- `timestamp`：处理时间戳（用于后续更新）
- `type`：文件类型（如 `test`、`src`、`doc`）

示例代码：

```python
collection.add(
    documents=[...],
    metadatas=[
        {
            "project_name": "example_project",
            "file_url": "https://github.com/example_project/file.py",
            "timestamp": "2025-04-18T12:00:00Z",
            "type": "src"
        },
        ...
    ],
    ids=[...]
)
```

---

### 3. 🔌 MCP 服务器配置

MCP（Model Context Protocol）可以作为中间层，处理 LLM 的查询请求，并从 ChromaDB 中检索相关上下文。

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

- 使用 ChromaDB 存储 GitHub Repo 中的结构化文档
- MCP 提供中间层抽象，供 LLM 动态获取上下文
- 支持元数据过滤和向量查询组合
- 可以扩展到多项目、多类型、增量更新等复杂需求

如需脚本模板或部署示例，可在此基础上拓展实现。

