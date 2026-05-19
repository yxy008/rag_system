"""
ingest.py - 文档入库脚本

使用方式：
  python ingest.py                    # 处理 documents/ 目录下的所有文档
  python ingest.py --dir ./my_docs    # 处理自定义目录
  python ingest.py --clear            # 清空数据库后重新入库

对应 RAG 流程的构建阶段：
  Documents → Document Splitter → Chunks → Embedding Model → Vector Database
"""
import argparse
import logging
import sys
from pathlib import Path

# 确保可以导入项目模块（脚本在 scripts/ 子目录下）
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="将文档入库到向量数据库")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="文档目录路径（默认使用 .env 中的 DOCUMENTS_DIR）",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="入库前先清空现有数据库",
    )
    args = parser.parse_args()

    from core.document_processor import DocumentProcessor
    from core.vector_store import VectorStoreManager
    from core.config import DOCUMENTS_DIR

    documents_dir = args.dir or DOCUMENTS_DIR

    # 检查文档目录
    if not Path(documents_dir).exists():
        logger.error(f"文档目录不存在：{documents_dir}")
        logger.info(f"请先将文档放入 {documents_dir} 目录后再运行本脚本")
        sys.exit(1)

    # 检查目录是否有文件
    supported_exts = {".txt", ".pdf", ".docx", ".md"}
    doc_files = [
        f for f in Path(documents_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in supported_exts
    ]
    if not doc_files:
        logger.error(f"目录 {documents_dir} 中没有支持的文档文件（.txt/.pdf/.docx/.md）")
        sys.exit(1)

    logger.info(f"找到 {len(doc_files)} 个文档文件：")
    for f in doc_files:
        logger.info(f"  - {f.name}")

    # 初始化模块
    vs_manager = VectorStoreManager()
    processor = DocumentProcessor(embeddings=vs_manager.embeddings)

    # 清空数据库（可选）
    if args.clear:
        logger.info("正在清空向量数据库...")
        vs_manager.clear_collection()

    # Step 1: 加载文档
    logger.info("=" * 50)
    logger.info("Step 1: 加载文档...")
    documents = processor.load_documents(documents_dir)
    if not documents:
        logger.error("文档加载失败，请检查文件格式")
        sys.exit(1)
    logger.info(f"✓ 加载完成：{len(documents)} 段原始文档")

    # Step 2: 分割文档
    logger.info("=" * 50)
    logger.info("Step 2: 分割文档为 Chunks...")
    chunks = processor.split_documents(documents)
    logger.info(f"✓ 分割完成：{len(chunks)} 个 Chunks")

    # Step 3: 向量化并写入向量数据库
    logger.info("=" * 50)
    logger.info("Step 3: 向量化并写入向量数据库（首次运行需下载 Embedding 模型，请稍候）...")
    count = vs_manager.add_documents(chunks)
    logger.info(f"✓ 写入完成：{count} 个 Chunks 已存入向量数据库")

    # 验证
    total = vs_manager.get_document_count()
    logger.info("=" * 50)
    logger.info(f"✓ 入库完成！向量数据库当前共有 {total} 个向量")
    if vs_manager._vector_store and hasattr(vs_manager._vector_store, '_persist_directory'):
        logger.info(f"  数据库路径：{vs_manager._vector_store._persist_directory}")
    logger.info("现在可以运行 python app.py 启动问答服务")


if __name__ == "__main__":
    main()
