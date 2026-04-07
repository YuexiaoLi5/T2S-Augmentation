#!/usr/bin/env python3
"""从 HuggingFace 下载 Qwen 模型"""
import os
from huggingface_hub import hf_hub_download, list_repo_files

# 设置缓存目录
cache_dir = "D:/HF_Models"
repo_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
hf_token = os.environ.get("HF_TOKEN", "")

print(f"从 HuggingFace 下载模型: {repo_id}")
print(f"缓存目录: {cache_dir}")

# 列出所有模型文件
print("\n获取文件列表...")
files = list_repo_files(repo_id, token=hf_token)
model_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt'))]
print(f"需要下载的模型文件: {len(model_files)} 个")

# 下载所有模型文件
for i, filename in enumerate(model_files):
    print(f"[{i+1}/{len(model_files)}] 下载 {filename}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            token=hf_token
        )
    except Exception as e:
        print(f"  下载失败: {e}")

print("\n下载完成!")

# 打印模型保存位置
import subprocess
result = subprocess.run(
    ["powershell", "-Command", f"Get-ChildItem -Path '{cache_dir}' -Recurse -Directory | Where-Object {{$_.Name -eq 'snapshots'}}"],
    capture_output=True, text=True
)
print(result.stdout)
