# MyKnowledgeGraph
Intelligent Data Engineer 1 - Build Knowledge Graph and Q&A System
## QuickStart
```
cd MyKnowledgeGraph-main
pip install -r requirements.txt
mkdir data
```

### Method 1: Build graph by command
Put your data files in `data` and run:
```python build.py```

### Method 2: Build graph by WebUI
Run:
```streamlit run \qa_web.py```
Then drag files to upload.

```
graph TD
A[文档读取] --> B[文本预处理]
B --> C[实体识别]
C --> D[关系抽取]
D --> E[实体定义生成]
E --> F[Neo4j图谱构建]

```
