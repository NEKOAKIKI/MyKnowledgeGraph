import os
import re
import json
import pdfplumber
from dotenv import load_dotenv
from neo4j import GraphDatabase
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

load_dotenv()


class KnowledgeGraphBuilder:
    def __init__(self):
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_en.max_length = 9_000_000_000
        self.ner_en = pipeline("ner", model="dslim/bert-base-NER")

        self.nlp_zh = spacy.load("zh_core_web_sm")
        self.nlp_zh.max_length = 9_000_000_000
        zh_model = AutoModelForTokenClassification.from_pretrained("hfl/chinese-bert-wwm")
        zh_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
        self.ner_zh = pipeline("ner", model=zh_model, tokenizer=zh_tokenizer, aggregation_strategy="simple")

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def extract_text_from_pdf(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    def extract_text_from_txt(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _split_long_text(self, texts, max_length):
        chunks = []
        current = ""
        for t in texts:
            if len(current) + len(t) < max_length:
                current += t + "\n"
            else:
                chunks.append(current.strip())
                current = t
        if current:
            chunks.append(current.strip())
        return chunks

    def _format_entities(self, ner_results):
        entities = set()
        for item in ner_results:
            word = item.get('word', item.get('entity_group', '')).strip()
            label = item.get('entity', item.get('entity_group', '')).strip()
            if word and label:
                entities.add((word, label))
        return entities

    def extract_entities(self, text):
        en_texts, zh_texts = [], []
        for paragraph in text.split("\n"):
            if re.search(r'[a-zA-Z]', paragraph):
                en_texts.append(paragraph)
            if re.search(r'[\u4e00-\u9fff]', paragraph):
                zh_texts.append(paragraph)

        entities = set()
        for chunk in self._split_long_text(en_texts, 512):
            ner_results = self.ner_en(chunk)
            entities.update(self._format_entities(ner_results))

        for chunk in self._split_long_text(zh_texts, 128):
            ner_results = self.ner_zh(chunk)
            entities.update(self._format_entities(ner_results))

        return list(entities)

    def extract_relations(self, text):
        relations = []

        EN_PATTERNS = [
            (r"(.+?) is a type of (.+)", "is_a"),
            (r"(.+?) is a kind of (.+)", "is_a"),
            (r"(.+?) is a form of (.+)", "is_a"),
            (r"(.+?) is a (.+)", "is_a"),
            (r"(.+?) refers to (.+)", "refers_to"),
            (r"(.+?) is defined as (.+)", "defined_as"),
            (r"(.+?) is described as (.+)", "defined_as"),
            (r"(.+?) consists of (.+)", "consists_of"),
            (r"(.+?) contains (.+)", "contains"),
            (r"(.+?) includes (.+)", "includes"),
            (r"(.+?) depends on (.+)", "depends_on"),
            (r"(.+?) supports (.+)", "supports"),
            (r"(.+?) is related to (.+)", "related_to"),
            (r"(.+?) relates to (.+)", "related_to"),
            (r"(.+?) has (.+)", "has"),
            (r"(.+?) uses (.+)", "used_in"),
            (r"(.+?) applies (.+)", "applies"),
            (r"(.+?) requires (.+)", "requires"),
            (r"(.+?) manages (.+)", "manages"),
            (r"(.+?) enables (.+)", "enables"),
            (r"(.+?) represents (.+)", "represents"),
            (r"(.+?) stores (.+)", "stores"),
            (r"(.+?) processes (.+)", "processes"),
            (r"(.+?) handles (.+)", "handles"),
        ]

        for chunk in self._split_long_text([text], 5000):
            doc_en = self.nlp_en(chunk)
            for sent in doc_en.sents:
                sentence = sent.text.strip()
                s_lower = sentence.lower()
                for pattern, rel in EN_PATTERNS:
                    match = re.match(pattern, s_lower)
                    if match:
                        subj, obj = match.groups()
                        subj = subj.strip(" .").capitalize()
                        obj = obj.strip(" .").capitalize()
                        if len(subj) > 1 and len(obj) > 1:
                            relations.append((subj, rel, obj))

                for token in sent:
                    if token.dep_ == "nsubj" and token.head.lemma_ in {"use", "apply", "enforce"}:
                        subject = token.text
                        obj = [t for t in token.head.rights if t.dep_ in {"dobj", "attr"}]
                        if obj:
                            rel = "used_in" if token.head.lemma_ == "use" else "applies"
                            relations.append((subject, rel, obj[0].text))

        zh_patterns = [
            (r"(.+?)是(.+?)的一种", "是"),
            (r"(.+?)称为(.+?)", "称为"),
            (r"(.+?)的定义是(.+?)", "定义为"),
            (r"(.+?)叫做(.+?)", "叫做"),
            (r"(.+?)依赖于(.+?)", "依赖"),
            (r"(.+?)依赖(.+?)", "依赖"),
            (r"(.+?)包括(.+?)", "包括"),
            (r"(.+?)包含(.+?)", "包含"),
            (r"(.+?)属于(.+?)", "属于"),
            (r"(.+?)使用(.+?)", "使用"),
            (r"(.+?)由(.+?)构成", "构成"),
            (r"(.+?)提供(.+?)", "提供"),
            (r"(.+?)管理(.+?)", "管理"),
            (r"(.+?)存储(.+?)", "存储"),
            (r"(.+?)实现(.+?)", "实现"),
            (r"(.+?)与(.+?)通信", "通信"),
            (r"(.+?)与(.+?)连接", "连接"),
            (r"(.+?)主要解决(.+?)", "解决"),
            (r"(.+?)提高(.+?)", "提高"),
            (r"(.+?)适用于(.+?)", "适用于"),
            (r"(.+?)用于(.+?)", "用于"),
            (r"(.+?)能够发现(.+?)", "发现"),
            (r"(.+?)是一种(.+?)", "是"),
            (r"(.+?)是一类(.+?)", "是"),
            (r"(.+?)通过(.+?)实现", "实现"),
            (r"(.+?)基于(.+?)", "基于"),
            (r"(.+?)利用(.+?)", "利用"),
        ]

        for chunk in self._split_long_text([text], 100000):
            doc_zh = self.nlp_zh(chunk)
            for sent in doc_zh.sents:
                s = sent.text.strip()
                for pattern, rel in zh_patterns:
                    match = re.match(pattern, s)
                    if match:
                        subj, obj = match.groups()
                        subj = subj.strip()
                        obj = obj.strip("。")
                        if len(subj) > 1 and len(obj) > 1:
                            relations.append((subj, rel, obj))

        return list(set(relations))

    def build_graph(self, documents):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # 清空图谱

            for doc in documents:
                text = self.extract_text_from_pdf(doc)
                entities = self.extract_entities(text)
                relations = self.extract_relations(text)

                for entity, type_ in entities:
                    session.run(
                        "MERGE (e:Entity {name: $name}) SET e.type = $type",
                        name=entity, type=type_
                    )

                for src, rel, dst in relations:
                    session.run(
                        f"MATCH (a:Entity {{name: $src}}), (b:Entity {{name: $dst}}) "
                        f"MERGE (a)-[:{rel}]->(b)",
                        src=src, dst=dst
                    )

    def import_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with self.driver.session() as session:
            for item in data:
                session.run(
                    "MERGE (e:Entity {name: $name}) SET e.type = $type",
                    name=item["name"], type=item["type"]
                )
                for rel in item.get("relations", []):
                    session.run(
                        "MERGE (a:Entity {name: $src}) "
                        "MERGE (b:Entity {name: $dst}) "
                        f"MERGE (a)-[:{rel['type']}]->(b)",
                        src=item["name"],
                        dst=rel["target"]
                    )

    def query(self, question):
        with self.driver.session() as session:
            if question.startswith("什么是") or question.endswith("是什么"):
                entity = question.replace("什么是", "").replace("是什么", "").strip()
                result = session.run(
                    """
                    MATCH (n:Entity)-[r:是|定义为|refers_to|defined_as|叫做|称为]->(m)
                    WHERE n.name CONTAINS $entity
                    RETURN n.name AS subject, type(r) AS relation, m.name AS object
                    """,
                    entity=entity
                )
                return [(record["subject"], record["relation"], record["object"]) for record in result]

            elif "和" in question and "关系" in question:
                entities = re.findall(r"(.+?)和(.+?)", question)
                if entities:
                    e1, e2 = entities[0]
                    result = session.run(
                        "MATCH (a)-[r]->(b) WHERE a.name CONTAINS $e1 AND b.name CONTAINS $e2 "
                        "RETURN type(r) AS relation",
                        e1=e1, e2=e2
                    )
                    return [record["relation"] for record in result]

            return ["暂时无法回答这个问题"]


class CourseQA:
    def __init__(self, graph_builder):
        self.graph = graph_builder

    def ask(self, question):
        with self.graph.driver.session() as session:
            if question.startswith("什么是") or question.endswith("是什么"):
                entity = question.replace("什么是", "").replace("是什么", "").strip()
                result = session.run(
                    "MATCH (n:Entity) WHERE n.name CONTAINS $entity "
                    "RETURN n.name AS name, n.description AS desc"
                )
                records = result.data()
                if not records:
                    return "未找到相关信息"
                return "\n".join(
                    f"{r['name']}：{r['desc']}" if r.get("desc") else f"{r['name']}（暂无定义）"
                    for r in records
                )
            elif "和" in question and "关系" in question:
                m = re.findall(r"(.+?)和(.+?)", question)
                if m:
                    e1, e2 = m[0]
                    result = session.run(
                        "MATCH (a)-[r]->(b) WHERE a.name CONTAINS $e1 AND b.name CONTAINS $e2 "
                        "RETURN type(r) AS rel", e1=e1, e2=e2
                    )
                    return "两者之间的关系是：" + "、".join([record["rel"] for record in result])
            return "暂时无法回答这个问题"
