"""
document_processor.py - 文档处理模块

对应 RAG 流程中的：
  Documents → Document Splitter → Chunks
负责：
  1. 加载各种格式的文档（txt/pdf/docx/md/excel/csv/html/sqlite）
  2. 将文档分割为适合检索的 Chunks（按章节结构切分）
"""
import os
import re
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SEMANTIC_CHUNKING_PERCENTILE

logger = logging.getLogger(__name__)


# ============================================================
# 增强版 PDF 加载器（支持表格提取）
# ============================================================

class EnhancedPDFLoader:
    """
    增强版 PDF 加载器，使用 pdfplumber 提取文本和表格。
    相比 PyPDFLoader，能够：
      - 识别并提取表格，格式化为结构化文本
      - 保留表格的列对应关系
      - 更好地处理中文字符
    如果 pdfplumber 不可用，自动回退到 PyPDFLoader。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber 未安装，回退到 PyPDFLoader")
            return self._fallback_load()

        docs = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1

                    # 先提取表格
                    tables = page.extract_tables()
                    table_texts = self._extract_table_texts(tables)

                    # 提取页面文本
                    page_text = page.extract_text()
                    if page_text:
                        page_text = page_text.strip()
                        # 移除表格原始文本（避免与格式化表格重复）
                        if table_texts:
                            page_text = self._remove_table_text(page_text, table_texts)

                    text_parts = []
                    if page_text:
                        text_parts.append(page_text)

                    # 追加格式化表格
                    if tables:
                        for table_idx, table in enumerate(tables):
                            if not table or len(table) < 1:
                                continue
                            formatted_table = self._format_table(table, table_idx + 1)
                            if formatted_table:
                                text_parts.append(formatted_table)

                    if text_parts:
                        combined_text = "\n\n".join(text_parts)
                        doc = Document(
                            page_content=combined_text,
                            metadata={
                                "source": self.file_path,
                                "page": page_num,
                                "total_pages": total_pages,
                                "table_count": len(tables),
                            },
                        )
                        docs.append(doc)

            if docs:
                logger.info(
                    "已加载 PDF（增强模式）：%s，共 %d 页，%d 个表格",
                    Path(self.file_path).name, total_pages,
                    sum(d.metadata.get("table_count", 0) for d in docs),
                )
            else:
                logger.warning("pdfplumber 未能提取内容，回退到 PyPDFLoader: %s", self.file_path)
                return self._fallback_load()

            return docs
        except Exception as e:
            logger.warning("pdfplumber 加载失败(%s)，回退到 PyPDFLoader: %s", e, self.file_path)
            return self._fallback_load()

    def _extract_table_texts(self, tables: list) -> set:
        """提取表格中所有非空单元格文本集合，用于去重"""
        texts = set()
        if not tables:
            return texts
        for table in tables:
            if not table:
                continue
            for row in table:
                if row:
                    for cell in row:
                        if cell and str(cell).strip():
                            texts.add(str(cell).strip())
        return texts

    def _remove_table_text(self, page_text: str, table_texts: set) -> str:
        """从页面文本中移除表格行（避免与格式化表格重复）"""
        lines = page_text.split("\n")
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered_lines.append(line)
                continue
            # 检查该行是否主要由表格单元格内容组成
            words = stripped.split()
            if len(words) >= 2:
                match_count = 0
                for word in words:
                    if word in table_texts:
                        match_count += 1
                # 如果该行超过一半的词都在表格文本中，视为表格行，跳过
                if match_count >= len(words) * 0.5:
                    continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines)

    def _format_table(self, table: list, table_idx: int) -> str:
        """将表格格式化为可读的结构化文本"""
        if not table or len(table) < 1:
            return ""

        # 过滤全空行
        filtered_rows = []
        for row in table:
            if row and any(cell and str(cell).strip() for cell in row):
                filtered_rows.append([str(cell).strip() if cell else "" for cell in row])

        if not filtered_rows:
            return ""

        num_cols = max(len(row) for row in filtered_rows)
        # 补齐列数
        for row in filtered_rows:
            while len(row) < num_cols:
                row.append("")

        lines = [f"[表格 {table_idx}]"]
        header = filtered_rows[0]
        lines.append(" | ".join(header))
        lines.append(" | ".join(["---"] * num_cols))

        for row in filtered_rows[1:]:
            lines.append(" | ".join(row))

        return "\n".join(lines)

    def _fallback_load(self) -> List[Document]:
        """回退到 pypdf 直接加载"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(self.file_path)
            total_pages = len(reader.pages)
            docs = []
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                text = page.extract_text()
                if text and text.strip():
                    doc = Document(
                        page_content=text.strip(),
                        metadata={
                            "source": self.file_path,
                            "page": page_num,
                            "total_pages": total_pages,
                        },
                    )
                    docs.append(doc)
            logger.info("已加载 PDF（回退模式）：%s，共 %d 页", Path(self.file_path).name, len(docs))
            return docs
        except Exception as e:
            logger.error("pypdf 加载失败 %s: %s", self.file_path, e)
            return []


# ============================================================
# 自定义数据源加载器
# ============================================================

class ExcelLoader:
    """
    加载 Excel 文件（.xlsx / .xls），将每个 Sheet 的每行/每个区块转为文本。
    """

    def __init__(self, file_path: str, sheet_name: str = None):
        self.file_path = file_path
        self.sheet_name = sheet_name

    def load(self) -> List[Document]:
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas 未安装，无法加载 Excel 文件。请运行：pip install pandas openpyxl")
            return []

        try:
            # 读取所有 Sheet
            if self.sheet_name:
                sheets = {self.sheet_name: pd.read_excel(self.file_path, sheet_name=self.sheet_name)}
            else:
                sheets = pd.read_excel(self.file_path, sheet_name=None)

            docs = []
            for name, df in sheets.items():
                # 将每行转为字符串（去掉空值）
                df = df.fillna("")
                for idx, row in df.iterrows():
                    row_text = " | ".join(f"{col}:{val}" for col, val in row.items() if str(val).strip())
                    if row_text.strip():
                        doc = Document(
                            page_content=f"[Sheet: {name}] {row_text}",
                            metadata={"source": self.file_path, "sheet": name, "row": int(idx)}
                        )
                        docs.append(doc)

            logger.info(f"已加载 Excel：{Path(self.file_path).name}，共 {len(docs)} 行")
            return docs
        except Exception as e:
            logger.error(f"加载 Excel 失败 {self.file_path}: {e}")
            return []


class CSVLoader:
    """
    加载 CSV 文件，将每行转为文本。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas 未安装，无法加载 CSV 文件。请运行：pip install pandas")
            return []

        try:
            df = pd.read_csv(self.file_path).fillna("")
            docs = []
            for idx, row in df.iterrows():
                row_text = " | ".join(f"{col}:{val}" for col, val in row.items() if str(val).strip())
                if row_text.strip():
                    doc = Document(
                        page_content=row_text,
                        metadata={"source": self.file_path, "row": int(idx)}
                    )
                    docs.append(doc)

            logger.info(f"已加载 CSV：{Path(self.file_path).name}，共 {len(docs)} 行")
            return docs
        except Exception as e:
            logger.error(f"加载 CSV 失败 {self.file_path}: {e}")
            return []


class WebPageLoader:
    """
    加载网页内容，提取标题和正文。
    """

    def __init__(self, url: str):
        self.url = url

    def load(self) -> List[Document]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("requests 或 beautifulsoup4 未安装，无法加载网页。请运行：pip install requests beautifulsoup4")
            return []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = requests.get(self.url, headers=headers, timeout=10)
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "lxml")

            # 提取标题
            title = soup.find("title")
            title_text = title.get_text().strip() if title else ""

            # 提取正文（去除脚本、样式、导航等）
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            paragraphs = soup.find_all("p")
            content = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

            if not content:
                # fallback: 提取所有文本
                content = soup.get_text(separator="\n", strip=True)[:5000]

            doc = Document(
                page_content=f"# {title_text}\n\n{content}",
                metadata={"source": self.url, "title": title_text}
            )
            logger.info(f"已加载网页：{self.url}，内容长度：{len(content)} 字符")
            return [doc]
        except Exception as e:
            logger.error(f"加载网页失败 {self.url}: {e}")
            return []


class SQLiteLoader:
    """
    加载 SQLite 数据库，将表内容按行转为文本。
    """

    def __init__(self, db_path: str, table_name: str = None, query: str = None):
        self.db_path = db_path
        self.table_name = table_name
        self.query = query  # 自定义 SQL 查询

    def load(self) -> List[Document]:
        if not Path(self.db_path).exists():
            logger.error(f"数据库文件不存在：{self.db_path}")
            return []

        docs = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 确定要读取哪些表
            if self.query:
                cursor.execute(self.query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
            elif self.table_name:
                cursor.execute(f"SELECT * FROM {self.table_name}")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
            else:
                # 读取所有表
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                rows = []
                for (table_name,) in tables:
                    if table_name.startswith("sqlite_"):
                        continue
                    cursor.execute(f"SELECT * FROM {table_name}")
                    cols = [desc[0] for desc in cursor.description]
                    for row in cursor.fetchall():
                        rows.append((table_name, cols, row))
                columns = None

            conn.close()

            if columns:
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    row_text = " | ".join(f"{k}:{v}" for k, v in row_dict.items() if str(v).strip())
                    if row_text.strip():
                        doc = Document(
                            page_content=row_text,
                            metadata={"source": self.db_path, "table": self.table_name or "query"}
                        )
                        docs.append(doc)
            else:
                # 多表模式
                for item in rows:
                    table_name, cols, row = item
                    row_text = " | ".join(f"{k}:{v}" for k, v in zip(cols, row) if str(v).strip())
                    if row_text.strip():
                        doc = Document(
                            page_content=f"[表: {table_name}] {row_text}",
                            metadata={"source": self.db_path, "table": table_name}
                        )
                        docs.append(doc)

            logger.info(f"已加载数据库：{self.db_path}，共 {len(docs)} 行")
            return docs
        except Exception as e:
            logger.error(f"加载 SQLite 失败 {self.db_path}: {e}")
            return []


# ============================================================
# 自定义章节感知切分器
# ============================================================

class SectionAwareSplitter:
    """
    按章节结构切分文档，优先在标题处断开。

    适用结构（中文企业制度文档）：
      1. 总则
      1.1 第一条
      2. 报销范围
      3.1 提交申请
      3.1.2 子条款

    切分逻辑：
      1. 按一级标题（^\d+\.\s）分割 → 大章节
      2. 大章节 > chunk_size 时，再按二级/三级标题切
      3. 最终仍超长时，使用 RecursiveCharacterTextSplitter 兜底
    """

    SECTION_PATTERNS = [
        re.compile(r"^(第?[一二三四五六七八九十百零\d]+[章节条款]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.?|\d+\.)\s+.+", re.MULTILINE),
        re.compile(r"^\d+\.\d+\.?\s+.+", re.MULTILINE),
        re.compile(r"^\d+\.\d+\.\d+\.?\s+.+", re.MULTILINE),
        re.compile(r"^#{1,3}\s+.+", re.MULTILINE),
    ]

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        sections = self._split_by_pattern(text, 0)

        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                sub_sections = self._split_by_pattern(section, 1)
                for sub in sub_sections:
                    if len(sub) <= self.chunk_size:
                        chunks.append(sub)
                    else:
                        chunks.extend(self._recursive_split(sub))

        chunks = self._merge_short_chunks(chunks)
        return chunks

    def _split_by_pattern(self, text: str, pattern_level: int) -> List[str]:
        if pattern_level >= len(self.SECTION_PATTERNS):
            return [text]

        pattern = self.SECTION_PATTERNS[pattern_level]
        matches = list(pattern.finditer(text))

        if not matches:
            return [text]

        sections = []
        # 处理第一个匹配之前的内容（前言/导语部分，之前会丢失）
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.append(preamble)

        for i, match in enumerate(matches):
            start = match.start()
            next_match = matches[i + 1] if i + 1 < len(matches) else None
            end = next_match.start() if next_match else len(text)
            content = text[start:end].strip()
            sections.append(content)

        return sections if sections else [text]

    def _recursive_split(self, text: str) -> List[str]:
        fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len,
        )
        return fallback.split_text(text)

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return chunks
        merged = []
        buffer = chunks[0]
        for chunk in chunks[1:]:
            if len(buffer) < self.chunk_size * 0.3 and len(buffer) + len(chunk) <= self.chunk_size * 1.5:
                buffer = buffer + "\n\n" + chunk
            else:
                merged.append(buffer)
                buffer = chunk
        merged.append(buffer)
        return merged


# ============================================================
# 语义分块器
# ============================================================

class SemanticChunker:
    """
    基于 Embedding 相似度的语义分块器。

    原理：
      1. 将文本按句子拆分
      2. 计算相邻句子的 Embedding 余弦相似度
      3. 在相似度骤降处（语义转折点）切分
      4. 确保每个 chunk 是语义完整的段落

    相比规则分块的优势：
      - 不会在句子中间切断
      - 同一主题的内容聚合在一起
      - 不同主题的内容自然分离
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        similarity_percentile: float = 90.0,
    ):
        """
        Args:
            chunk_size: 目标 chunk 大小（字符数）
            chunk_overlap: 重叠字符数
            similarity_percentile: 相似度分位数阈值（越低切得越细）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_percentile = similarity_percentile
        self._embeddings = None

    def _get_embeddings(self):
        """懒加载 Embedding 模型"""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

            model_path = EMBEDDING_MODEL_NAME
            if model_path.startswith("./") or model_path.startswith(".\\"):
                from config import BASE_DIR
                model_path = str(BASE_DIR / model_path)

            logger.info(f"语义分块器正在加载 Embedding 模型：{model_path}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": EMBEDDING_DEVICE},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _split_sentences(self, text: str) -> List[str]:
        """
        将文本拆分为句子列表。

        中文句子边界：。！？；\n
        英文句子边界：. ! ? ;
        """
        # 先按换行拆分
        paragraphs = text.split("\n")
        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # 按标点拆分句子
            parts = re.split(r"(?<=[。！？；.!?;])", para)
            for part in parts:
                part = part.strip()
                if part and len(part) >= 5:
                    sentences.append(part)

        return sentences

    def _compute_similarities(self, sentences: List[str]) -> List[float]:
        """
        计算相邻句子的余弦相似度。

        使用 BGE-M3 对每个句子编码，然后计算相邻向量的余弦相似度。
        由于向量已归一化，余弦相似度 = 向量点积。
        """
        if len(sentences) <= 1:
            return []

        embeddings = self._get_embeddings()
        vectors = embeddings.embed_documents(sentences)
        vectors = np.array(vectors)

        similarities = []
        for i in range(len(vectors) - 1):
            sim = np.dot(vectors[i], vectors[i + 1])
            similarities.append(float(sim))

        return similarities

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        根据相似度找到语义断点。

        策略：找到相似度低于分位数阈值的句子边界作为断点。
        """
        if not similarities:
            return []

        threshold = np.percentile(similarities, 100 - self.similarity_percentile)
        # 确保阈值不低于 0.3（避免切得太碎）
        threshold = max(threshold, 0.3)

        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)  # 在第 i+1 句之前断开

        return breakpoints

    def split_text(self, text: str) -> List[str]:
        """
        对文本进行语义分块。

        流程：
          1. 拆分为句子
          2. 计算相邻句子相似度
          3. 找到语义断点
          4. 在断点处切分，同时控制 chunk 大小
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        similarities = self._compute_similarities(sentences)
        breakpoints = set(self._find_breakpoints(similarities))

        # 按断点分组句子
        chunks = []
        current_chunk = []
        current_len = 0

        for i, sentence in enumerate(sentences):
            sent_len = len(sentence)

            # 检查是否需要在当前句子前断开
            should_break = (
                i in breakpoints
                or (current_len + sent_len > self.chunk_size and current_len > self.chunk_size * 0.3)
            )

            if should_break and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)
                # 重叠：保留最后一句作为下一个 chunk 的开头
                if self.chunk_overlap > 0 and len(current_chunk) >= 1:
                    overlap_sentence = current_chunk[-1]
                    current_chunk = [overlap_sentence]
                    current_len = len(overlap_sentence)
                else:
                    current_chunk = []
                    current_len = 0

            current_chunk.append(sentence)
            current_len += sent_len

        # 最后一个 chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(chunk_text)

        # 合并过短的 chunk
        chunks = self._merge_short_chunks(chunks)
        return chunks

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """合并过短的 chunk 到前一个 chunk"""
        if not chunks:
            return chunks
        merged = []
        buffer = chunks[0]
        for chunk in chunks[1:]:
            if len(buffer) < self.chunk_size * 0.3 and len(buffer) + len(chunk) <= self.chunk_size * 1.5:
                buffer = buffer + "\n\n" + chunk
            else:
                merged.append(buffer)
                buffer = chunk
        merged.append(buffer)
        return merged


# ============================================================
# 文档处理器
# ============================================================

# 所有支持的文件格式
SUPPORTED_EXTENSIONS = {
    ".txt": ("文本文件", None),
    ".pdf": ("PDF 文件", None),
    ".docx": ("Word 文件", None),
    ".doc": ("Word 文件", None),
    ".md": ("Markdown 文件", None),
    ".xlsx": ("Excel 文件", ExcelLoader),
    ".xls": ("Excel 文件", ExcelLoader),
    ".csv": ("CSV 文件", CSVLoader),
    ".html": ("HTML 文件", None),  # 用 UnstructuredHTMLLoader
    ".htm": ("HTML 文件", None),
    ".sqlite": ("SQLite 数据库", SQLiteLoader),
    ".db": ("SQLite 数据库", SQLiteLoader),
}


class DocumentProcessor:
    """
    文档处理器：负责加载和分割文档。
    支持格式：
      - 文本: txt, pdf, docx, doc, md, html, htm
      - Excel: xlsx, xls（按行加载）
      - CSV: csv（按行加载）
      - 网页: URL（自动抓取）
      - 数据库: sqlite, db（按行加载）
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.section_splitter = SectionAwareSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.semantic_splitter = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_percentile=SEMANTIC_CHUNKING_PERCENTILE,
        )

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len,
        )

    def load_documents(self, documents_dir: str = DOCUMENTS_DIR) -> List[Document]:
        """从指定目录加载所有支持格式的文档"""
        documents_path = Path(documents_dir)
        if not documents_path.exists():
            logger.warning(f"文档目录不存在：{documents_dir}")
            return []

        all_docs: List[Document] = []

        for file_path in documents_path.rglob("*"):
            if file_path.is_file():
                docs = self._load_single_file(str(file_path))
                if docs:
                    all_docs.extend(docs)
                    logger.info(f"已加载文档：{file_path.name}（{len(docs)} 段）")

        logger.info(f"共加载文档段落：{len(all_docs)} 段，来自目录：{documents_dir}")
        return all_docs

    def _load_text(self, file_path: str) -> List[Document]:
        """加载纯文本文件（.txt / .md），自动尝试多种编码"""
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                if content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path},
                    )
                    return [doc]
                return []
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error("加载文本文件失败 %s: %s", file_path, e)
                return []
        logger.error("无法识别文件编码 %s，已尝试: %s", file_path, encodings)
        return []

    def _load_docx(self, file_path: str) -> List[Document]:
        """加载 Word 文件（.docx / .doc）"""
        try:
            import docx2txt
            text = docx2txt.process(file_path)
            if text and text.strip():
                doc = Document(
                    page_content=text.strip(),
                    metadata={"source": file_path},
                )
                return [doc]
            return []
        except ImportError:
            logger.error("docx2txt 未安装，无法加载 Word 文件。请运行：pip install docx2txt")
            return []
        except Exception as e:
            logger.error("加载 Word 文件失败 %s: %s", file_path, e)
            return []

    def _load_html(self, file_path: str) -> List[Document]:
        """加载 HTML 文件"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            soup = BeautifulSoup(content, "lxml")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path},
                )
                return [doc]
            return []
        except ImportError:
            logger.error("beautifulsoup4 未安装，无法加载 HTML 文件。请运行：pip install beautifulsoup4 lxml")
            return []
        except Exception as e:
            logger.error("加载 HTML 文件失败 %s: %s", file_path, e)
            return []

    def _load_single_file(self, file_path: str) -> List[Document]:
        """根据文件扩展名选择合适的 Loader"""
        ext = Path(file_path).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(f"不支持的文件格式，已跳过：{file_path}")
            return []

        _, loader_cls = SUPPORTED_EXTENSIONS[ext]

        try:
            if loader_cls is not None:
                # 自定义加载器
                loader = loader_cls(file_path)
                return loader.load()
            elif ext in (".html", ".htm"):
                return self._load_html(file_path)
            elif ext == ".txt":
                return self._load_text(file_path)
            elif ext == ".pdf":
                loader = EnhancedPDFLoader(file_path)
                return loader.load()
            elif ext in (".docx", ".doc"):
                return self._load_docx(file_path)
            elif ext == ".md":
                return self._load_text(file_path)
            else:
                return []
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return []

    def load_url(self, url: str) -> List[Document]:
        """从 URL 加载网页内容"""
        loader = WebPageLoader(url)
        return loader.load()

    def load_sqlite(
        self,
        db_path: str,
        table_name: str = None,
        query: str = None,
    ) -> List[Document]:
        """从 SQLite 数据库加载内容"""
        loader = SQLiteLoader(db_path, table_name=table_name, query=query)
        return loader.load()

    def _has_section_structure(self, text: str) -> bool:
        """
        检测文本是否具有章节/条款结构。

        判断标准：
          1. 匹配到至少 2 个章节标题（避免误判）
          2. 标题密度合理（每 500 字符至少 1 个标题，排除只有一两个标题的长文）

        匹配的标题模式：
          - "第X章"、"第X条"、"第一条"
          - "1."、"1.1"、"1.1.1" 编号标题
          - "## " Markdown 标题
        """
        if not text or len(text) < 100:
            return False

        total_matches = 0
        for pattern in SectionAwareSplitter.SECTION_PATTERNS:
            matches = pattern.findall(text)
            total_matches += len(matches)

        # 至少 2 个标题，且密度合理
        if total_matches < 2:
            return False

        density = len(text) / max(total_matches, 1)
        if density > 2000:
            return False

        return True

    def _is_tabular_format(self, source: str) -> bool:
        """
        判断文档是否为表格类格式（Excel / CSV / 数据库）。

        这些格式在加载时已经按行切分好了，不需要再走分块器。
        """
        ext = Path(source).suffix.lower()
        return ext in {".xlsx", ".xls", ".csv", ".sqlite", ".db"}

    def _classify_document(self, doc: Document) -> str:
        """
        对单篇文档进行分类，返回分块策略标识。

        分类逻辑（优先级从高到低）：
          1. 表格类 → "tabular"（已按行切好，跳过）
          2. 有章节结构 → "section"（制度/合同/法律文档）
          3. 文本较长 → "semantic"（网页/报告/无结构文档）
          4. 短文本 → "fallback"（递归字符切兜底）
        """
        source = doc.metadata.get("source", "")
        text = doc.page_content

        # 1. 表格类
        if self._is_tabular_format(source):
            return "tabular"

        # 2. 有章节结构
        if self._has_section_structure(text):
            return "section"

        # 3. 长文本用语义分块
        if len(text) > self.chunk_size:
            return "semantic"

        # 4. 短文本兜底
        return "fallback"

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        组合拳分块：根据每篇文档的特征自动选择最优分块器。

        路由策略：
          - 表格类（xls/xlsx/csv/sqlite）→ 已按行切好，直接保留
          - 有章节结构（制度/合同）    → SectionAwareSplitter（条款精准切分）
          - 无结构长文本（网页/报告）  → SemanticChunker（语义边界切分）
          - 短文本 / 兜底             → RecursiveCharacterTextSplitter
        """
        if not documents:
            logger.warning("没有可供分割的文档")
            return []

        all_chunks = []
        stats = {"section": 0, "semantic": 0, "tabular": 0, "fallback": 0}

        for doc in documents:
            strategy = self._classify_document(doc)
            stats[strategy] += 1
            raw_text = doc.page_content

            if strategy == "tabular":
                # 表格类：加载时已按行切好，直接保留
                chunk_doc = Document(
                    page_content=raw_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_method": "tabular（表格按行）",
                    },
                )
                all_chunks.append(chunk_doc)
                continue

            # 选择分块器
            if strategy == "section":
                splitter = self.section_splitter
                method_name = "section（章节感知）"
            elif strategy == "semantic":
                splitter = self.semantic_splitter
                method_name = "semantic（语义边界）"
            else:
                splitter = self.fallback_splitter
                method_name = "fallback（递归字符）"

            # 执行分块
            try:
                if hasattr(splitter, "split_text"):
                    chunk_texts = splitter.split_text(raw_text)
                else:
                    chunk_texts = splitter.split_text(raw_text)
            except Exception as e:
                logger.warning(f"{method_name} 分块失败，回退到递归字符切：{e}")
                chunk_texts = self.fallback_splitter.split_text(raw_text)
                method_name = "fallback（异常回退）"

            for i, text in enumerate(chunk_texts):
                chunk_doc = Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunk_texts),
                        "chunk_method": method_name,
                    },
                )
                all_chunks.append(chunk_doc)

        logger.info(
            f"组合拳分块完成：{len(documents)} 篇文档 → {len(all_chunks)} 个 Chunks"
            f"（章节感知={stats['section']}篇, 语义边界={stats['semantic']}篇, "
            f"表格按行={stats['tabular']}篇, 递归兜底={stats['fallback']}篇）"
        )
        return all_chunks

    def load_and_split(self, documents_dir: str = DOCUMENTS_DIR) -> List[Document]:
        """一步完成加载 + 分割"""
        documents = self.load_documents(documents_dir)
        return self.split_documents(documents)
