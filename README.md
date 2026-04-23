# 📄 低资源场景IE

> 此项目旨在使用 LLM 取代传统深度学习模型，实现非监督矿床知识图谱构建。使用由 LLM 与向量数据库组合的智能体，实现命名实体识别（NER）、关系抽取（RE）和实体对齐（EA）。适用于受 Schema 约束的矿床知识图谱构建。
> 此版本为尚未公开的调试版本，注意传播范围。

## 📌 简介（Overview）

在低资源、少标注数据的地质领域（特别是矿床学），传统基于监督学习的信息抽取方法难以应用。本项目提出一种基于大语言模型（LLM）与向量数据库（ChromaDB）协同工作的智能体框架，无需训练即可完成端到端的知识图谱构建流程：

- **命名实体识别（NER）**：从矿床相关论文中识别地质实体（如矿物、岩石、构造等）。
- **关系抽取（RE）**：抽取实体间的语义关系，形成三元组。
- **实体对齐（EA）**：跨文档/跨矿床合并指代相同现实对象的实体。

最终输出符合预定义 Schema 的结构化知识图谱，并支持导入 Neo4j 图数据库。

## 🚀 主要功能（Features）

- ✅ 基于 LLM 的零样本信息抽取（无需标注数据）
- ✅ 利用 ChromaDB与LLM结合 实现跨文档实体消歧与对齐
- ✅ 支持用户自定义知识图谱 Schema
- ✅ 分阶段处理：抽取 → 矿床内合并 → 全局合并 → 导入图数据库
- ✅ 适用于低资源、专业性强的地质文本场景

## 🛠️ 技术栈（Tech Stack）

- **语言**：Python 3.9+
- **核心库**：
  - `py2neo`：操作 Neo4j 图数据库
  - `openai`：调用 OpenAI 大语言模型 API
  - `chromadb`：本地向量数据库，用于实体嵌入与相似性匹配
  - `Neo4j`：图数据库
- **其他**：标准库 `os`, `json`, `logging` 等

## 📦 安装与运行（Installation & Usage）

### 前置条件
- Python 3.9 或更高版本
- 已安装并运行 Neo4j 数据库（本地或远程）
- 拥有 OpenAI API Key（或其他兼容 OpenAI 接口的 LLM 服务）

### 步骤

1. 克隆项目：
   ```bash
   git clone https://github.com/your-username/low-resource-ie.git 
   cd low-resource-ie
   #虚构的，尚未上传
2. 创建并激活虚拟环境（推荐）：
    conda create -n env
3. 安装依赖：
    pip install -r requirements.txt
4. 配置环境变量与图谱 Schema
    编辑 config.py 文件，设置以下内容：
    OpenAI API Key（或兼容接口的 base_url / api_key）以及所用模型
    Neo4j 连接参数（URI、用户名、密码）
    知识图谱 Schema（实体类型、关系类型等）
5. 运行三阶段构建流程：
    第一步：逐篇抽取各个矿床下论文内的实体与三元组
    python step1_extraction.py
    第二步：逐矿床进行第一轮实体合并（矿床内部对齐）
    python step2_merge_deposit_entities.py
    第三步：合并所有矿床下的实体，并导入 Neo4j 或生成备份文件。需先启动neo4j
    neo4j console
    python step3_merge_all_entities.py
### 项目结构
    low-resource-ie/
    ├── config.py
    ├── step1_extraction.py
    ├── step2_merge_deposit_entities.py
    ├── step3_merge_all_entities.py
    ├── utils/
    │   ├── KG_function.py
    │   ├── LLM_function.py
    │   └── Vector_Database_function.py
    ├── data/{date}
    │   ├── origin
    │   ├── step1_result
    │   ├── step2_result
    │   └── step3_result
    ├── requirements.txt
    └── README.md
### 许可证
    本项目采用 MIT 许可证 — 详情见 LICENSE 文件。


