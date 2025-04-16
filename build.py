import os
import json
import pandas as pd
from knowledge_graph_backend import KnowledgeGraphBuilder


def main():
    builder = KnowledgeGraphBuilder()

    # 构建 PDF / TXT 图谱
    doc_files = [f"data/" + f for f in os.listdir("data") if f.endswith((".pdf", ".txt"))]
    builder.build_graph(doc_files)

    # 导入 JSON 数据
    json_files = [f for f in os.listdir("data") if f.endswith(".json")]
    for jf in json_files:
        builder.import_from_json(os.path.join("data", jf))

    # 处理 CSV 文件（三元组格式）
    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]
    for csvf in csv_files:
        df = pd.read_csv(os.path.join("data", csvf))
        with builder.driver.session() as session:
            for _, row in df.iterrows():
                session.run(
                    "MERGE (a:Entity {name: $src}) "
                    "MERGE (b:Entity {name: $dst}) "
                    f"MERGE (a)-[:{row['relation']}]->(b)",
                    src=row['source'], dst=row['target']
                )

    print("✅ 所有图谱数据导入完成！")


if __name__ == "__main__":
    main()
