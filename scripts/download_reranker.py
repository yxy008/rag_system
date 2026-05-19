"""下载 bge-reranker-large 模型并复制到项目目录"""
from huggingface_hub import snapshot_download
import os
import shutil
from pathlib import Path

# 1. 下载到缓存
print("正在从 HuggingFace 下载 bge-reranker-large...")
cache_dir = snapshot_download(
    "BAAI/bge-reranker-large",
    cache_dir=None,
    local_files_only=False,
)
print(f"缓存路径: {cache_dir}")

# 2. 列出下载的文件
files = os.listdir(cache_dir)
print(f"下载文件数: {len(files)}")
for f in files:
    fpath = os.path.join(cache_dir, f)
    size = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {f}: {size:.1f} MB")

# 3. 复制到项目目录
target = str(Path(__file__).parent.parent / "models" / "bge-reranker-large")
os.makedirs(target, exist_ok=True)
for f in files:
    src = os.path.join(cache_dir, f)
    dst = os.path.join(target, f)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
print(f"\n已复制到: {target}")

# 4. 验证模型可加载
from sentence_transformers import CrossEncoder
print("\n正在验证模型加载...")
model = CrossEncoder(target, device="cuda")
print(f"模型加载成功！max_length={model.max_length}")
print("全部完成！")
