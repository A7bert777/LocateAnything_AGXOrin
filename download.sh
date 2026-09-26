#!/bin/bash
# ============================================================
# LocateAnything-3B 资源下载脚本
# 适用于 NVIDIA AGX Orin 64GB
# 目标路径: 脚本所在目录（自动识别，相对路径）
# ============================================================

set -e

# 自动获取脚本所在目录（相对路径，支持项目整体移动/重命名）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
MODEL_DIR="$PROJECT_DIR/models"
CODE_DIR="$PROJECT_DIR/code"
VENV_DIR="$PROJECT_DIR/venv"

# ------------------------------------------------------------
# 若在中国大陆网络环境，取消下面这行注释可加速下载（使用 hf-mirror 镜像）
# export HF_ENDPOINT=https://hf-mirror.com
# ------------------------------------------------------------

# ------------------------------------------------------------
# 检查 HF_TOKEN：未认证下载会被限速，强烈建议设置
# 获取方式：登录 https://huggingface.co/settings/tokens 创建并复制 token
# 设置方式：export HF_TOKEN=hf_xxxxxxxxxxxxx
# ------------------------------------------------------------
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  提示：未检测到 HF_TOKEN，未认证下载可能被限速。"
    echo "    建议先执行：export HF_TOKEN=hf_你的token"
    echo "    （先在 https://huggingface.co/settings/tokens 创建 token）"
    echo ""
fi

echo "============================================"
echo "第 1 步：创建 Python 虚拟环境（解决 PEP 668 限制）"
echo "============================================"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "虚拟环境已创建: $VENV_DIR"
else
    echo "虚拟环境已存在，跳过创建"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------
# 禁用 hf-xet 传输协议，改用传统 HTTP 下载（避免大文件下载时的
# "File reconstruction error: CAS Client Error" 兼容性 Bug）
# ------------------------------------------------------------
export HF_HUB_DISABLE_XET=1

echo "============================================"
echo "第 2 步：升级 pip 并安装 HuggingFace 下载工具"
echo "============================================"
pip install -U pip
pip install -U "huggingface_hub[cli]"

echo ""
echo "============================================"
echo "第 3 步：创建目录结构"
echo "============================================"
mkdir -p "$MODEL_DIR" "$CODE_DIR"

echo ""
echo "============================================"
echo "第 4 步：下载官方模型权重（BF16 原始精度，约 7.66GB）"
echo "  仓库: nvidia/LocateAnything-3B"
echo "  注意：下载大文件时会显示实时进度条（每个文件单独显示）"
echo "  （已禁用 xet 协议，改用稳定的传统 HTTP 下载）"
echo "============================================"
# 使用 hf download 命令（支持断点续传，自带实时进度条）
# 已通过 HF_HUB_DISABLE_XET=1 禁用 xet 协议，避免大文件下载报错
hf download nvidia/LocateAnything-3B \
    --local-dir "$MODEL_DIR/LocateAnything-3B"

# 备选命令（若 hf 命令不可用，使用 huggingface-cli）:
# huggingface-cli download nvidia/LocateAnything-3B \
#     --local-dir "$MODEL_DIR/LocateAnything-3B"

echo ""
echo "============================================"
echo "第 5 步：检查官方 demo 部署代码（NVlabs/Eagle）"
echo "  核心推理脚本位于 Embodied/locateanything_worker.py"
echo "============================================"
# 说明：官方代码已随本仓库一起分发（code/Eagle），
#       正常情况下无需再下载。仅当目录缺失时才联网克隆。
if [ -d "$CODE_DIR/Eagle/Embodied" ]; then
    echo "✅ 官方代码已存在，跳过下载"
else
    echo "未找到 code/Eagle，正在从 GitHub 克隆官方代码..."
    git clone https://github.com/NVlabs/Eagle.git "$CODE_DIR/Eagle"
fi

echo ""
echo "============================================"
echo "第 6 步（可选）：下载社区 TensorRT 参考实现"
echo "  仓库: carpedm20/locate-anything-tensorrt"
echo "  注意：当前版本针对 RTX 5090 (SM120)，仅供部署思路参考"
echo "============================================"
hf download carpedm20/locate-anything-tensorrt \
    --local-dir "$MODEL_DIR/locate-anything-tensorrt-reference"

echo ""
echo "============================================"
echo "✅ 所有资源下载完成！"
echo "  模型权重: $MODEL_DIR/LocateAnything-3B"
echo "  官方代码: $CODE_DIR/Eagle/Embodied"
echo "  TRT参考 : $MODEL_DIR/locate-anything-tensorrt-reference"
echo ""
echo "  后续使用环境："
echo "    source $VENV_DIR/bin/activate"
echo "============================================"
