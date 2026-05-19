"""查找模型缓存并复制到项目目录"""
from huggingface_hub import snapshot_download
import os
import shutil
from pathlib import Path

cache_dir = snapshot_download("BAAI/bge-reranker-large")
print("Cache dir:", cache_dir)
files = os.listdir(cache_dir)
print("Files:", files)

target = str(Path(__file__).parent.parent / "models" / "bge-reranker-large")
os.makedirs(target, exist_ok=True)
for f in files:
    src = os.path.join(cache_dir, f)
    dst = os.path.join(target, f)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    print(f"Copied: {f}")
print("Done!")
