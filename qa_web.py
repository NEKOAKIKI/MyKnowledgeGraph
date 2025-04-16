import streamlit as st
import tempfile
import os
import pandas as pd
from knowledge_graph_backend import KnowledgeGraphBuilder, CourseQA
from pyvis.network import Network
from streamlit.components.v1 import html

st.set_page_config(page_title="课程知识图谱问答系统", layout="wide")


@st.cache_resource
def load_qa():
    builder = KnowledgeGraphBuilder()
    return CourseQA(builder), builder


qa, builder = load_qa()

st.title("🎓 课程知识图谱问答系统")

tab1, tab2, tab3 = st.tabs(["📖 问答系统", "📂 上传构建图谱", "🌐 图谱可视化"])

# -------------------------
# 📖 问答系统
# -------------------------
with tab1:
    st.subheader("🧠 问答界面")

    # 历史记录持久化
    if "history" not in st.session_state:
        st.session_state.history = []

    # 常见问题样例
    example_questions = [
        "什么是主成分分析？",
        "什么是数据降维？",
        "PCA和LDA有什么关系？",
        "K均值和层次聚类的关系？",
        "支持向量机是什么？"
    ]

    st.markdown("📌 **示例问题**")
    cols = st.columns(len(example_questions))
    for i, ex in enumerate(example_questions):
        if cols[i].button(ex):
            st.session_state.question_input = ex

    question = st.text_input("请输入问题（如：什么是PCA？）", key="question_input")

    if st.button("🔍 查询") and question.strip():
        with st.spinner("查询中..."):
            answer = qa.ask(question)
        st.success("📘 回答：")
        st.write(answer)
        st.session_state.history.append((question, answer))

    if st.session_state.history:
        with st.expander("🕘 历史提问记录"):
            for q, a in reversed(st.session_state.history[-10:]):
                st.markdown(f"**Q：{q}**")
                st.markdown(f"🧾 {a}")
                st.markdown("---")

# -------------------------
# 📂 上传构建图谱
# -------------------------
with tab2:
    st.subheader("📤 知识图谱构建")

    uploaded_files = st.file_uploader(
        "选择文件上传（支持 PDF / JSON / TXT / CSV）：",
        type=["pdf", "json", "txt", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("⚙️ 构建知识图谱"):
        with st.spinner("⏳ 正在处理文件..."):
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                if suffix in [".pdf", ".txt"]:
                    builder.build_graph([tmp_path])
                elif suffix == ".json":
                    builder.import_from_json(tmp_path)
                elif suffix == ".csv":
                    df = pd.read_csv(tmp_path)
                    with builder.driver.session() as session:
                        for _, row in df.iterrows():
                            session.run(
                                "MERGE (a:Entity {name: $src}) "
                                "MERGE (b:Entity {name: $dst}) "
                                "MERGE (a)-[r:" + row['relation'] + "]->(b)",
                                src=row['source'], dst=row['target']
                            )
        st.success("✅ 构建完成，快去“问答系统”试试吧！")

# -------------------------
# 🌐 图谱可视化
# -------------------------
with tab3:
    st.subheader("🌐 图谱可视化")
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    with builder.driver.session() as session:
        result = session.run("MATCH (a)-[r]->(b) RETURN a.name AS source, type(r) AS rel, b.name AS target")
        added_nodes = set()
        for record in result:
            src, rel, dst = record["source"], record["rel"], record["target"]
            if src not in added_nodes:
                net.add_node(src, label=src, title=src)
                added_nodes.add(src)
            if dst not in added_nodes:
                net.add_node(dst, label=dst, title=dst)
                added_nodes.add(dst)
            net.add_edge(src, dst, label=rel)

    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "graph.html")
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as f:
        graph_html = f.read()
    html(graph_html, height=620, scrolling=True)
