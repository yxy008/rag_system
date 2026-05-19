"""
knowledge_graph.py - 知识图谱与知识探索模块

功能：
  1. 交互式知识图谱（实体识别 + 关系抽取）
  2. 对比式问答（多文档对比分析）
  3. 场景模拟问答（假设性推理）
  4. 智能文档摘要生成器（多层级摘要）
"""
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations.db")


class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        category TEXT DEFAULT 'concept',
                        doc_source TEXT DEFAULT '',
                        metadata_json TEXT DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_node_id INTEGER NOT NULL,
                        target_node_id INTEGER NOT NULL,
                        relation TEXT NOT NULL,
                        weight REAL DEFAULT 1.0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                        FOREIGN KEY (source_node_id) REFERENCES knowledge_graph_nodes(id),
                        FOREIGN KEY (target_node_id) REFERENCES knowledge_graph_nodes(id)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_kg_nodes_name
                    ON knowledge_graph_nodes(name)
                """)
                conn.commit()
            finally:
                conn.close()

    def extract_entities(self, documents: List[Dict]) -> Dict:
        """
        从文档中提取实体和关系，构建知识图谱数据

        使用基于规则的方法提取关键实体（生产环境可替换为 LLM 提取）
        """
        nodes = []
        edges = []
        node_names = set()

        # 基于规则的实体提取
        entity_patterns = [
            ("制度", "policy"),
            ("办法", "policy"),
            ("规定", "policy"),
            ("流程", "process"),
            ("申请", "action"),
            ("审批", "action"),
            ("年假", "concept"),
            ("事假", "concept"),
            ("病假", "concept"),
            ("加班", "concept"),
            ("报销", "concept"),
            ("考勤", "concept"),
            ("薪资", "concept"),
            ("福利", "concept"),
            ("培训", "concept"),
            ("考核", "concept"),
            ("入职", "process"),
            ("离职", "process"),
            ("转正", "process"),
            ("调岗", "process"),
        ]

        for doc in documents:
            source = doc.get("source", "")
            content = doc.get("full_content") or doc.get("preview") or ""

            for pattern, category in entity_patterns:
                if pattern in content or pattern in source:
                    if pattern not in node_names:
                        node_names.add(pattern)
                        nodes.append({
                            "id": len(nodes) + 1,
                            "name": pattern,
                            "category": category,
                            "doc_source": source,
                        })

        # 构建关系（基于共现）
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if i >= j:
                    continue
                # 检查是否在同一文档中共现
                if node_a["doc_source"] == node_b["doc_source"]:
                    edges.append({
                        "source": node_a["id"],
                        "target": node_b["id"],
                        "relation": "关联",
                        "weight": 0.8,
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def get_graph_data(self) -> Dict:
        """获取知识图谱数据"""
        with self._lock:
            conn = self._get_conn()
            try:
                nodes = conn.execute(
                    "SELECT * FROM knowledge_graph_nodes ORDER BY id"
                ).fetchall()
                edges = conn.execute(
                    "SELECT * FROM knowledge_graph_edges ORDER BY id"
                ).fetchall()

                return {
                    "nodes": [dict(n) for n in nodes],
                    "edges": [dict(e) for e in edges],
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                }
            finally:
                conn.close()

    def build_from_documents(self, documents: List[Dict]):
        """从文档列表构建知识图谱并持久化"""
        graph_data = self.extract_entities(documents)

        with self._lock:
            conn = self._get_conn()
            try:
                # 清空旧数据
                conn.execute("DELETE FROM knowledge_graph_edges")
                conn.execute("DELETE FROM knowledge_graph_nodes")

                # 插入节点
                node_id_map = {}
                for node in graph_data["nodes"]:
                    cursor = conn.execute(
                        """INSERT INTO knowledge_graph_nodes (name, category, doc_source, metadata_json)
                           VALUES (?, ?, ?, ?)""",
                        (node["name"], node["category"], node.get("doc_source", ""),
                         json.dumps(node.get("metadata", {}), ensure_ascii=False))
                    )
                    node_id_map[node["id"]] = cursor.lastrowid

                # 插入边
                for edge in graph_data["edges"]:
                    src_db_id = node_id_map.get(edge["source"])
                    tgt_db_id = node_id_map.get(edge["target"])
                    if src_db_id and tgt_db_id:
                        conn.execute(
                            """INSERT INTO knowledge_graph_edges
                               (source_node_id, target_node_id, relation, weight)
                               VALUES (?, ?, ?, ?)""",
                            (src_db_id, tgt_db_id, edge["relation"], edge.get("weight", 1.0))
                        )

                conn.commit()
                logger.info("知识图谱构建完成：%d 个节点，%d 条边",
                            graph_data["node_count"], graph_data["edge_count"])
            finally:
                conn.close()

        return graph_data


class ComparativeQA:
    """对比式问答处理器"""

    @staticmethod
    def build_comparison_prompt(topic_a: str, topic_b: str, context_a: str, context_b: str) -> str:
        """构建对比问答的 Prompt"""
        return f"""请对比分析以下两个主题，以表格形式呈现差异：

【主题 A】{topic_a}
参考内容：
{context_a}

【主题 B】{topic_b}
参考内容：
{context_b}

请按以下格式输出：

## 对比分析：{topic_a} vs {topic_b}

| 对比维度 | {topic_a} | {topic_b} |
|----------|-----------|-----------|
| （自动提取关键维度） | | |

## 关键差异总结
1. 
2. 

## 互补关系
- 

## 建议
- 
"""


class ScenarioSimulator:
    """场景模拟问答处理器"""

    @staticmethod
    def build_simulation_prompt(scenario: str, context: str) -> str:
        """构建场景模拟的 Prompt"""
        return f"""请基于以下参考文档，对假设场景进行推理分析：

【假设场景】
{scenario}

【参考文档】
{context}

请按以下格式回答：

## 场景分析
（分析场景涉及的关键要素）

## 适用规则
（列出适用的具体规定，标注来源）

## 可能结果
1. 最可能的结果：...
2. 其他可能：...

## 风险提示
- 

## 建议
- 
"""


class DocumentSummarizer:
    """智能文档摘要生成器"""

    @staticmethod
    def build_summary_prompt(content: str, level: str = "structured") -> str:
        """构建摘要生成的 Prompt"""
        level_prompts = {
            "one_line": "请用一句话（30字以内）总结以下文档的核心内容。",
            "paragraph": "请用一段话（100字以内）总结以下文档的主要内容。",
            "structured": """请对以下文档进行结构化摘要，按以下格式输出：

## 文档概述
（一句话总结）

## 核心要点
1. 
2. 
3. 

## 关键数据/规定
- 

## 适用场景
- """,
            "bullets": "请提取以下文档的关键要点，以 bullet points 列表形式输出。",
            "actions": "请从以下文档中提取所有需要执行的行动项和待办事项。",
        }

        instruction = level_prompts.get(level, level_prompts["structured"])
        return f"""{instruction}

【文档内容】
{content}
"""


knowledge_graph = KnowledgeGraphBuilder()