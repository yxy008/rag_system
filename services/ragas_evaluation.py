"""
ragas_evaluation.py - RAGAS 质量评估模块（自实现版）

基于 RAGAS 方法论，使用项目自身的 LLM 客户端实现核心评估指标，
无需依赖 ragas Python 库（避免 langchain-core 版本冲突）。

Phase 1（无需 ground truth）—— 4 个指标:
  - Faithfulness: 答案中的声明是否能从检索上下文推导
  - Answer Relevancy: 答案是否与问题相关
  - Context Precision: 检索到的文档信噪比（逐文档 + 排序位置加权）
  - Context Relevancy: 检索上下文中与问题相关的句子占比（逐句粒度）

Phase 2（需 ground truth）—— 6 个指标（含 Phase 1 的 4 个）:
  - Context Recall: ground truth 中的信息在检索上下文中的覆盖率
  - Context Entity Recall: ground truth 关键实体在检索上下文中的出现率
  - Answer Correctness: 答案与 ground truth 的事实准确性
  - Answer Semantic Similarity: 答案与 ground truth 的语义相似度

评估数据来源: answer_provenance 表（question + answer + sources_json）
评估模式: 离线批处理，不嵌入在线请求链路
"""
import json
import logging
import math
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional

import openai
import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import (
    EVAL_API_KEY,
    EVAL_BASE_URL,
    EVAL_LLM_MODEL,
    BASE_DIR,
)

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(str(BASE_DIR), "data", "conversations.db")

# ======================================================================
#  RAGAS Prompt 模板（中文优化版）
# ======================================================================

# Faithfulness: 逐声明验证
FAITHFULNESS_PROMPT = """你的任务是从给定答案中提取所有事实声明，并判断每条声明是否能从提供的上下文中推导出来。

## 上下文
{context}

## 答案
{answer}

## 指令
1. 将答案分解为独立的事实声明（statement）
2. 对每条声明，判断它是否能完全从上下文中推导出来
3. 输出 JSON 格式

## 输出格式
```json
{{
  "statements": [
    {{
      "statement": "声明内容",
      "verdict": 1,
      "reason": "判断理由"
    }}
  ]
}}
```

- verdict: 1 表示能从上下文推导，0 表示不能
- 如果答案中没有任何事实声明，返回空数组
"""

# Answer Relevancy: 基于答案反向生成问题
ANSWER_RELEVANCY_PROMPT = """你的任务是根据给定的答案，生成这个答案可能回答的问题。

## 答案
{answer}

## 指令
基于答案的内容，生成 3 个不同表述但语义相同的问题，这些问题应当都能被上述答案回答。

## 输出格式
```json
{{
  "questions": ["问题1", "问题2", "问题3"]
}}
```
"""

# Context Precision: 逐文档判断相关性
CONTEXT_PRECISION_PROMPT = """你的任务是判断给定的文档块是否与用户问题相关。

## 用户问题
{question}

## 文档内容
{context}

## 指令
判断上述文档内容是否与用户问题相关（即文档是否包含对回答该问题有用的信息）。

## 输出格式
```json
{{
  "relevant": true,
  "reason": "判断理由"
}}
```

- relevant: true 表示相关，false 表示不相关
"""

# Context Relevancy: 逐句判断相关性，计算相关句子占比
CONTEXT_RELEVANCY_PROMPT = """你的任务是从检索到的文档内容中提取与用户问题相关的句子，并判断相关句子占比。

## 用户问题
{question}

## 检索到的文档内容
{context}

## 指令
1. 将检索到的文档内容按句子拆分
2. 逐句判断是否与用户问题相关（即该句子是否包含对回答问题有用的信息）
3. 统计相关句子数和总句子数
4. 如果没有任何相关句子，返回空数组

## 输出格式
```json
{{
  "total_sentences": 10,
  "relevant_sentences": 6,
  "relevant_sentence_list": ["相关句子1", "相关句子2", "..."],
  "reason": "整体判断理由"
}}
```

- total_sentences: 文档内容的总句子数
- relevant_sentences: 与问题相关的句子数
"""

# Context Entity Recall: 实体级召回评估
CONTEXT_ENTITY_RECALL_PROMPT = """你的任务是从标准答案中提取关键实体，并判断这些实体是否在检索上下文中出现。

## 检索上下文
{context}

## 标准答案（ground truth）
{ground_truth}

## 指令
1. 从标准答案中提取所有关键实体（包括：人名、地名、数字/日期、专有名词、重要概念等）
2. 对每个实体，判断它是否在检索上下文中出现（允许同义表达，不要求完全一致）
3. 输出 JSON 格式

## 输出格式
```json
{{
  "entities": [
    {{
      "entity": "实体名称",
      "type": "实体类型（person/location/number/organization/concept等）",
      "found": true,
      "matched_expression": "在上下文中匹配到的表达"
    }}
  ]
}}
```

- found: true 表示实体在检索上下文中出现，false 表示未出现
"""
CONTEXT_RECALL_PROMPT = """你的任务是将标准答案分解为独立的事实声明，并判断每条声明是否能从提供的检索上下文中推导出来。

## 检索上下文
{context}

## 标准答案（ground truth）
{ground_truth}

## 指令
1. 将标准答案分解为独立的事实声明
2. 对每条声明，判断它是否能从检索上下文中推导出来
3. 输出 JSON 格式

## 输出格式
```json
{{
  "statements": [
    {{
      "statement": "声明内容",
      "verdict": 1,
      "reason": "判断理由"
    }}
  ]
}}
```

- verdict: 1 表示能从上下文推导，0 表示不能
"""

# Answer Correctness: 事实准确性判断
ANSWER_CORRECTNESS_PROMPT = """你的任务是判断 AI 生成的答案与标准答案（ground truth）的事实一致性。

## 用户问题
{question}

## AI 生成的答案
{answer}

## 标准答案（ground truth）
{ground_truth}

## 指令
1. 判断 AI 答案中的事实陈述是否与标准答案一致
2. 从 0 到 1 打分，其中 1 表示完全一致，0 表示完全不一致
3. 考虑以下维度：
   - 事实正确性 (Factual Correctness): AI 答案中的事实是否与标准答案一致
   - 语义相似度 (Semantic Similarity): AI 答案与标准答案的语义是否相似

## 输出格式
```json
{{
  "score": 0.85,
  "factual_correctness": 0.9,
  "semantic_similarity": 0.8,
  "reason": "判断理由"
}}
```
"""


class RAGASEvaluator:
    """
    RAGAS 质量评估器（自实现版）

    - Phase 1: 无需 ground truth，可自动从生产日志收集样本并评估
    - Phase 2: 需要 ground truth，支持手动标注和评估
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._openai_client = openai.OpenAI(
            api_key=EVAL_API_KEY,
            base_url=EVAL_BASE_URL,
            timeout=60.0,
        )
        self._eval_model = EVAL_LLM_MODEL
        self._embedding_model = None
        self._init_db()

    # ------------------------------------------------------------------
    #  数据库初始化
    # ------------------------------------------------------------------

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
                    CREATE TABLE IF NOT EXISTS ragas_eval_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        eval_time TEXT NOT NULL,
                        phase TEXT NOT NULL DEFAULT 'phase1',
                        sample_count INTEGER DEFAULT 0,
                        metrics_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ragas_ground_truth (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT NOT NULL UNIQUE,
                        ground_truth TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    #  Embedding 模型（懒加载）
    # ------------------------------------------------------------------

    def _get_embedding_model(self):
        """获取 sentence-transformers embedding 模型"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                EVAL_EMBEDDING_MODEL,
                device=self._detect_device(),
            )
        return self._embedding_model

    @staticmethod
    def _detect_device() -> str:
        """检测可用设备，优先 CUDA，不可用时回退 CPU"""
        try:
            import torch
            if torch.cuda.is_available():
                # 验证 CUDA 真实可用（is_available 为 True 不代表 PyTorch 编译了 CUDA）
                try:
                    torch.zeros(1).cuda()
                    return "cuda"
                except (AssertionError, RuntimeError):
                    logger.warning("CUDA 驱动存在但 PyTorch 未编译 CUDA 支持，回退到 CPU")
                    return "cpu"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    #  LLM 调用
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 并返回响应文本""" 
        try:
            resp = self._openai_client.chat.completions.create(
                model=self._eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            return ""

    def _call_llm_json(self, prompt: str) -> Optional[Dict]:
        """调用 LLM 并解析 JSON 响应"""
        text = self._call_llm(prompt)
        if not text:
            return None
        try:
            # 尝试提取 JSON 块
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                text = text[start:end]
            elif "```" in text:
                start = text.index("```") + 3
                end = text.index("```", start)
                text = text[start:end]
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("LLM JSON 解析失败: %s, 原始输出: %s", e, text[:200])
            return None

    # ------------------------------------------------------------------
    #  数据收集
    # ------------------------------------------------------------------

    def collect_eval_samples(self, limit: int = 50) -> List[Dict]:
        """
        从 answer_provenance 表收集评估样本

        Returns:
            [{"question": ..., "answer": ..., "contexts": [...]}, ...]
        """
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT question, answer, sources_json
                       FROM answer_provenance
                       WHERE answer IS NOT NULL AND answer != ''
                         AND sources_json IS NOT NULL AND sources_json != ''
                         AND sources_json != '[]'
                       ORDER BY id DESC LIMIT ?""",
                    (limit,)
                ).fetchall()

                if not rows:
                    return []

                samples = []
                for row in rows:
                    question = row["question"] or ""
                    answer = row["answer"] or ""
                    sources_json = row["sources_json"] or ""

                    try:
                        sources = json.loads(sources_json)
                    except (json.JSONDecodeError, TypeError):
                        sources = []

                    contexts = []
                    for s in sources:
                        content = s.get("full_content") or s.get("preview") or ""
                        content = content.strip()
                        if content:
                            contexts.append(content)

                    if question and answer and contexts:
                        samples.append({
                            "question": question,
                            "answer": answer,
                            "contexts": contexts,
                        })

                logger.info("从 answer_provenance 收集到 %d 条有效评估样本", len(samples))
                return samples
            finally:
                conn.close()

    # ==================================================================
    #  核心评估方法
    # ==================================================================

    def _eval_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness 评估

        流程:
          1. LLM 从 answer 中提取事实声明
          2. LLM 逐声明判断是否能从 contexts 中推导
          3. 分数 = 可推导声明数 / 总声明数
        """
        context_text = "\n\n---\n\n".join(contexts[:5])  # 最多取 5 段上下文
        prompt = FAITHFULNESS_PROMPT.format(context=context_text[:4000], answer=answer[:2000])
        result = self._call_llm_json(prompt)

        if not result or "statements" not in result:
            return 0.0

        statements = result["statements"]
        if not statements:
            return 0.0

        verdicts = [int(s.get("verdict", 0)) for s in statements]
        return round(sum(verdicts) / len(verdicts), 4) if verdicts else 0.0

    def _eval_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy 评估

        流程:
          1. LLM 基于 answer 反向生成 N 个问题
          2. 计算每个生成问题与原始问题的 cosine 相似度
          3. 分数 = 平均相似度
        """
        prompt = ANSWER_RELEVANCY_PROMPT.format(answer=answer[:3000])
        result = self._call_llm_json(prompt)

        if not result or "questions" not in result:
            return 0.0

        generated_questions = result["questions"]
        if not generated_questions:
            return 0.0

        try:
            model = self._get_embedding_model()
            orig_emb = model.encode(question, normalize_embeddings=True)
            gen_embs = model.encode(generated_questions, normalize_embeddings=True)

            # 确保 orig_emb 是 1D
            if orig_emb.ndim > 1:
                orig_emb = orig_emb[0]

            similarities = [float(np.dot(orig_emb, g)) for g in gen_embs]
            avg_sim = sum(similarities) / len(similarities)
            return round(max(0.0, min(1.0, avg_sim)), 4)
        except Exception as e:
            logger.warning("Answer Relevancy embedding 计算失败: %s", e)
            return 0.0

    def _eval_context_precision(self, question: str, contexts: List[str]) -> float:
        """
        Context Precision 评估

        流程:
          1. 对每个 context，LLM 判断是否与 question 相关
          2. 计算 precision@k（考虑排序位置权重）
          3. 加权平均

        公式: CP@k = (sum(precision@k * relevance_k)) / total_relevant
        """
        if not contexts:
            return 0.0

        # 取前 5 个上下文做精确计算（太多会卡）
        top_contexts = contexts[:5]
        relevances = []

        for ctx in top_contexts:
            prompt = CONTEXT_PRECISION_PROMPT.format(
                question=question[:500],
                context=ctx[:2000],
            )
            result = self._call_llm_json(prompt)
            if result:
                relevances.append(1 if result.get("relevant", False) else 0)
            else:
                relevances.append(0)

        if sum(relevances) == 0:
            return 0.0

        # 计算加权 Context Precision
        total_relevant = sum(relevances)
        weighted_sum = 0.0
        cumulative = 0

        for k, rel in enumerate(relevances, 1):
            cumulative += rel
            precision_at_k = cumulative / k
            weighted_sum += precision_at_k * rel

        return round(weighted_sum / total_relevant, 4)

    def _eval_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """
        Context Relevancy 评估（Phase 1，无需 ground truth）

        逐句判断检索上下文中的句子是否与问题相关，返回相关句子占比。

        与 Context Precision 的区别:
          - Context Precision: 逐文档（chunk）粒度的信噪比，关注"整段是否相关"
          - Context Relevancy: 逐句粒度的信噪比，关注"无关句子稀释了多少有效信息"

        流程:
          1. LLM 将 contexts 中的所有文本按句子拆分
          2. LLM 逐句判断是否与 question 相关
          3. 分数 = 相关句子数 / 总句子数
        """
        context_text = "\n\n---\n\n".join(contexts[:5])  # 最多取 5 段上下文
        prompt = CONTEXT_RELEVANCY_PROMPT.format(
            question=question[:500],
            context=context_text[:3000],
        )
        result = self._call_llm_json(prompt)

        if not result:
            return 0.0

        total = result.get("total_sentences", 0)
        relevant = result.get("relevant_sentences", 0)

        if total <= 0:
            return 0.0

        return round(relevant / total, 4)

    def _eval_context_entity_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """
        Context Entity Recall 评估（Phase 2，需 ground truth）

        ground truth 中提到的关键实体，有多少在检索上下文中出现。
        这是实体粒度的召回评估，补充了 Context Recall（声明粒度）的盲区。

        流程:
          1. LLM 从 ground_truth 中提取关键实体
          2. LLM 逐实体判断是否在 contexts 中出现
          3. 分数 = 出现实体数 / 总实体数
        """
        context_text = "\n\n---\n\n".join(contexts[:5])  # 最多取 5 段上下文
        prompt = CONTEXT_ENTITY_RECALL_PROMPT.format(
            context=context_text[:3000],
            ground_truth=ground_truth[:1500],
        )
        result = self._call_llm_json(prompt)

        if not result or "entities" not in result:
            return 0.0

        entities = result["entities"]
        if not entities:
            return 0.0

        found_count = sum(1 for e in entities if e.get("found", False))
        return round(found_count / len(entities), 4)

    def _eval_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """
        Context Recall 评估

        流程:
          1. LLM 将 ground_truth 分解为事实声明
          2. 逐声明判断是否能从 contexts 中推导
          3. 分数 = 可推导声明数 / 总声明数
        """
        context_text = "\n\n---\n\n".join(contexts[:5])
        prompt = CONTEXT_RECALL_PROMPT.format(
            context=context_text[:4000],
            ground_truth=ground_truth[:2000],
        )
        result = self._call_llm_json(prompt)

        if not result or "statements" not in result:
            return 0.0

        statements = result["statements"]
        if not statements:
            return 0.0

        verdicts = [int(s.get("verdict", 0)) for s in statements]
        return round(sum(verdicts) / len(verdicts), 4) if verdicts else 0.0

    def _eval_answer_correctness(self, question: str, answer: str, ground_truth: str) -> float:
        """
        Answer Correctness 评估

        流程:
          1. LLM 比较 answer 与 ground_truth 的事实一致性
          2. 综合考虑事实正确性和语义相似度
          3. 返回综合分数
        """
        prompt = ANSWER_CORRECTNESS_PROMPT.format(
            question=question[:1000],
            answer=answer[:2000],
            ground_truth=ground_truth[:2000],
        )
        result = self._call_llm_json(prompt)

        if not result or "score" not in result:
            return 0.0

        return round(max(0.0, min(1.0, float(result["score"]))), 4)

    def _eval_answer_similarity(self, answer: str, ground_truth: str) -> float:
        """
        Answer Semantic Similarity 评估

        使用 embedding 计算答案与 ground truth 的语义相似度
        """
        try:
            model = self._get_embedding_model()
            emb_ans = model.encode(answer, normalize_embeddings=True)
            emb_gt = model.encode(ground_truth, normalize_embeddings=True)

            if emb_ans.ndim > 1:
                emb_ans = emb_ans[0]
            if emb_gt.ndim > 1:
                emb_gt = emb_gt[0]

            similarity = float(np.dot(emb_ans, emb_gt))
            return round(max(0.0, min(1.0, similarity)), 4)
        except Exception as e:
            logger.warning("Answer Similarity 计算失败: %s", e)
            return 0.0

    # ==================================================================
    #  Phase 1：无需 ground truth 的评估
    # ==================================================================

    def run_phase1(self, sample_limit: int = 20) -> Dict:
        """
        执行 Phase 1 评估（无需 ground truth）

        评估指标: Faithfulness, Answer Relevancy, Context Precision, Context Relevancy
        """
        logger.info("========== Phase 1 RAGAS 评估开始 ==========")

        samples = self.collect_eval_samples(limit=sample_limit)
        if not samples:
            return {
                "phase": "phase1",
                "sample_count": 0,
                "error": "没有足够的评估样本，请先进行几次问答后再试",
                "timestamp": datetime.now().isoformat(),
            }

        logger.info("Phase 1 收集到 %d 条评估样本，开始逐条评估...", len(samples))

        f_scores, ar_scores, cp_scores, cr_scores = [], [], [], []

        for i, s in enumerate(samples):
            question = s["question"]
            answer = s["answer"]
            contexts = s["contexts"]

            logger.info("Phase 1 [%d/%d]: 评估 question='%s'", i + 1, len(samples), question[:50])

            # Faithfulness
            f = self._eval_faithfulness(question, answer, contexts)
            f_scores.append(f)
            logger.info("  Faithfulness: %.4f", f)

            # Answer Relevancy
            ar = self._eval_answer_relevancy(question, answer)
            ar_scores.append(ar)
            logger.info("  Answer Relevancy: %.4f", ar)

            # Context Precision
            cp = self._eval_context_precision(question, contexts)
            cp_scores.append(cp)
            logger.info("  Context Precision: %.4f", cp)

            # Context Relevancy (新增: 逐句粒度)
            cr = self._eval_context_relevancy(question, contexts)
            cr_scores.append(cr)
            logger.info("  Context Relevancy: %.4f", cr)

        metrics = {
            "faithfulness": round(np.mean(f_scores), 4) if f_scores else 0.0,
            "answer_relevancy": round(np.mean(ar_scores), 4) if ar_scores else 0.0,
            "context_precision": round(np.mean(cp_scores), 4) if cp_scores else 0.0,
            "context_relevancy": round(np.mean(cr_scores), 4) if cr_scores else 0.0,
        }

        logger.info("Phase 1 评估结果: %s", metrics)

        self._save_result(metrics, len(samples), phase="phase1")

        return {
            "phase": "phase1",
            "sample_count": len(samples),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    # ==================================================================
    #  Phase 2：需要 ground truth 的评估
    # ==================================================================

    def run_phase2(self, sample_limit: int = 20) -> Dict:
        """
        执行 Phase 2 评估（需要 ground truth）
        """
        logger.info("========== Phase 2 RAGAS 评估开始 ==========")

        samples = self._collect_phase2_samples(limit=sample_limit)
        if not samples:
            return {
                "phase": "phase2",
                "sample_count": 0,
                "error": "没有足够的 ground truth 样本，请先在评估页面标注 ground truth",
                "timestamp": datetime.now().isoformat(),
            }

        logger.info("Phase 2 收集到 %d 条带 ground truth 的评估样本", len(samples))

        f_scores, ar_scores, cp_scores, crel_scores = [], [], [], []
        cr_scores, cer_scores, ac_scores, as_scores = [], [], [], []

        for i, s in enumerate(samples):
            question = s["question"]
            answer = s["answer"]
            contexts = s["contexts"]
            ground_truth = s["ground_truth"]

            logger.info("Phase 2 [%d/%d]: 评估 question='%s'", i + 1, len(samples), question[:50])

            # Phase 1 指标
            f = self._eval_faithfulness(question, answer, contexts)
            f_scores.append(f)

            ar = self._eval_answer_relevancy(question, answer)
            ar_scores.append(ar)

            cp = self._eval_context_precision(question, contexts)
            cp_scores.append(cp)

            crel = self._eval_context_relevancy(question, contexts)
            crel_scores.append(crel)
            logger.info("  Context Relevancy: %.4f", crel)

            # Phase 2 专属指标
            cr = self._eval_context_recall(ground_truth, contexts)
            cr_scores.append(cr)
            logger.info("  Context Recall: %.4f", cr)

            cer = self._eval_context_entity_recall(ground_truth, contexts)
            cer_scores.append(cer)
            logger.info("  Context Entity Recall: %.4f", cer)

            ac = self._eval_answer_correctness(question, answer, ground_truth)
            ac_scores.append(ac)
            logger.info("  Answer Correctness: %.4f", ac)

            asim = self._eval_answer_similarity(answer, ground_truth)
            as_scores.append(asim)
            logger.info("  Answer Similarity: %.4f", asim)

        metrics = {
            "faithfulness": round(np.mean(f_scores), 4) if f_scores else 0.0,
            "answer_relevancy": round(np.mean(ar_scores), 4) if ar_scores else 0.0,
            "context_precision": round(np.mean(cp_scores), 4) if cp_scores else 0.0,
            "context_relevancy": round(np.mean(crel_scores), 4) if crel_scores else 0.0,
            "context_recall": round(np.mean(cr_scores), 4) if cr_scores else 0.0,
            "context_entity_recall": round(np.mean(cer_scores), 4) if cer_scores else 0.0,
            "answer_correctness": round(np.mean(ac_scores), 4) if ac_scores else 0.0,
            "answer_similarity": round(np.mean(as_scores), 4) if as_scores else 0.0,
        }

        logger.info("Phase 2 评估结果: %s", metrics)

        self._save_result(metrics, len(samples), phase="phase2")

        return {
            "phase": "phase2",
            "sample_count": len(samples),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    def _collect_phase2_samples(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT ap.question, ap.answer, ap.sources_json, gt.ground_truth
                       FROM answer_provenance ap
                       INNER JOIN ragas_ground_truth gt ON ap.question = gt.question
                       WHERE ap.answer IS NOT NULL AND ap.answer != ''
                         AND ap.sources_json IS NOT NULL AND ap.sources_json != ''
                         AND ap.sources_json != '[]'
                       ORDER BY ap.id DESC LIMIT ?""",
                    (limit,)
                ).fetchall()

                samples = []
                for row in rows:
                    question = row["question"] or ""
                    answer = row["answer"] or ""
                    ground_truth = row["ground_truth"] or ""
                    sources_json = row["sources_json"] or ""

                    try:
                        sources = json.loads(sources_json)
                    except (json.JSONDecodeError, TypeError):
                        sources = []

                    contexts = []
                    for s in sources:
                        content = s.get("full_content") or s.get("preview") or ""
                        content = content.strip()
                        if content:
                            contexts.append(content)

                    if question and answer and ground_truth and contexts:
                        samples.append({
                            "question": question,
                            "answer": answer,
                            "contexts": contexts,
                            "ground_truth": ground_truth,
                        })

                return samples
            finally:
                conn.close()

    # ==================================================================
    #  Ground Truth 管理
    # ==================================================================

    def add_ground_truth(self, question: str, ground_truth: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO ragas_ground_truth (question, ground_truth) VALUES (?, ?)",
                    (question, ground_truth)
                )
                conn.commit()
                return True
            except Exception as e:
                logger.error("添加 ground truth 失败: %s", e)
                return False
            finally:
                conn.close()

    def batch_add_ground_truth(self, entries: List[Dict]) -> int:
        count = 0
        for entry in entries:
            q = entry.get("question", "")
            gt = entry.get("ground_truth", "")
            if q and gt and self.add_ground_truth(q, gt):
                count += 1
        return count

    def get_all_ground_truths(self) -> List[Dict]:
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT id, question, ground_truth, created_at FROM ragas_ground_truth ORDER BY id DESC"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def delete_ground_truth(self, gt_id: int) -> bool:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM ragas_ground_truth WHERE id = ?", (gt_id,))
                conn.commit()
                return True
            except Exception as e:
                logger.error("删除 ground truth 失败: %s", e)
                return False
            finally:
                conn.close()

    def get_ground_truth_count(self) -> int:
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT COUNT(*) as cnt FROM ragas_ground_truth").fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()

    # ==================================================================
    #  结果查询
    # ==================================================================

    def _save_result(self, metrics: Dict, sample_count: int, phase: str = "phase1"):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO ragas_eval_results (eval_time, phase, sample_count, metrics_json) VALUES (?, ?, ?, ?)",
                    (datetime.now().isoformat(), phase, sample_count, json.dumps(metrics, ensure_ascii=False))
                )
                conn.commit()
            finally:
                conn.close()

    def get_recent_results(self, phase: str = None, limit: int = 10) -> List[Dict]:
        with self._lock:
            conn = self._get_conn()
            try:
                if phase:
                    rows = conn.execute(
                        "SELECT * FROM ragas_eval_results WHERE phase = ? ORDER BY id DESC LIMIT ?",
                        (phase, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM ragas_eval_results ORDER BY id DESC LIMIT ?",
                        (limit,)
                    ).fetchall()

                results = []
                for row in rows:
                    item = dict(row)
                    try:
                        item["metrics"] = json.loads(item["metrics_json"])
                    except (json.JSONDecodeError, TypeError):
                        item["metrics"] = {}
                    results.append(item)
                return results
            finally:
                conn.close()

    def get_latest_result(self, phase: str = "phase1") -> Optional[Dict]:
        results = self.get_recent_results(phase=phase, limit=1)
        return results[0] if results else None

    def get_trend(self, phase: str = "phase1", limit: int = 30) -> List[Dict]:
        results = self.get_recent_results(phase=phase, limit=limit)
        return list(reversed(results))


# ------------------------------------------------------------------
#  全局单例
# ------------------------------------------------------------------

ragas_evaluator = RAGASEvaluator()