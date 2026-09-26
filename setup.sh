#!/bin/bash
# ============================================================
# LocateAnything-3B 在 AGX Orin 64GB 上的一键环境配置脚本
# 适用: JetPack 6.2 / Ubuntu 24.04 / Python 3.10
# ============================================================
set -e

# 自动获取脚本所在目录（相对路径，支持项目整体移动/重命名）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
MODEL_DIR="$PROJECT_DIR/models/LocateAnything-3B"
VENV_DIR="$PROJECT_DIR/venv310"
EMBODIED_DIR="$PROJECT_DIR/code/Eagle/Embodied"

# NVIDIA Jetson 专用 torch wheel（Python 3.10 / CUDA 12.6）
TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

echo "============================================================"
echo "LocateAnything-3B 环境配置脚本"
echo "============================================================"

# ------------------------------------------------------------
# 第 1 步：创建 Python 3.10 虚拟环境
# ------------------------------------------------------------
echo ""
echo "[1/4] 创建 Python 3.10 虚拟环境..."
if [ ! -d "$VENV_DIR" ]; then
    # 用 --without-pip 创建（因为系统 python3.10 缺 ensurepip）
    python3.10 -m venv --without-pip "$VENV_DIR"
    echo "    虚拟环境已创建: $VENV_DIR"
else
    echo "    虚拟环境已存在，跳过"
fi

# 引导 pip（如果 venv 内没有 pip）
if [ ! -f "$VENV_DIR/bin/pip" ]; then
    echo "    正在引导 pip..."
    if [ -f "$PROJECT_DIR/get-pip.py" ]; then
        "$VENV_DIR/bin/python" "$PROJECT_DIR/get-pip.py" 2>/dev/null || true
    else
        echo "    下载 get-pip.py..."
        "$VENV_DIR/bin/python" -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '$PROJECT_DIR/get-pip.py')"
        "$VENV_DIR/bin/python" "$PROJECT_DIR/get-pip.py"
    fi
fi

# 固化 LD_LIBRARY_PATH（torch 需要 nvidia-cusparselt 的 libcusparseLt.so.0）
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
echo "    LD_LIBRARY_PATH 已设置: $LD_LIBRARY_PATH"

# ------------------------------------------------------------
# 第 2 步：安装 PyTorch（NVIDIA Jetson 专用 GPU 版）+ cuSparseLt
# ------------------------------------------------------------
echo ""
echo "[2/4] 安装 PyTorch (Jetson GPU 版) + cuSparseLt..."
if "$VENV_DIR/bin/python" -c "import torch" 2>/dev/null; then
    echo "    PyTorch 已安装，跳过"
else
    echo "    下载并安装 torch wheel（约 807MB，需要几分钟）..."
    "$VENV_DIR/bin/pip" install "$TORCH_WHEEL_URL"
fi

# 检查并安装 libcusparseLt.so.0（torch 硬依赖，JetPack 系统缺失）
if [ ! -f "$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0" ]; then
    echo "    安装 nvidia-cusparselt-cu12（提供 libcusparseLt.so.0）..."
    "$VENV_DIR/bin/pip" install nvidia-cusparselt-cu12==0.8.1
else
    echo "    libcusparseLt.so.0 已存在，跳过"
fi

# 检查并降级 numpy（torch 2.5.0 需要 numpy<2）
if "$VENV_DIR/bin/pip" show numpy 2>/dev/null | grep -qE "Version: 2\."; then
    echo "    降级 numpy 到 1.26.4（torch 兼容性要求）..."
    "$VENV_DIR/bin/pip" install "numpy==1.26.4"
else
    echo "    numpy 版本兼容，跳过"
fi

# ------------------------------------------------------------
# 第 3 步：安装模型推理依赖
# ------------------------------------------------------------
echo ""
echo "[3/4] 安装模型推理依赖..."
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# ------------------------------------------------------------
# 第 4 步：验证环境
# ------------------------------------------------------------
echo ""
echo "[4/4] 验证环境..."
"$VENV_DIR/bin/python" -c "
import torch
print(f'    ✅ torch {torch.__version__}')
print(f'    ✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    ✅ CUDA version: {torch.version.cuda}')
    print(f'    ✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'    ✅ transformers (依赖已安装)')
print(f'    ✅ 模型路径: ${MODEL_DIR}')
"

echo ""
echo "============================================================"
echo "⚠️  重要：每次使用前需要设置 LD_LIBRARY_PATH"
echo "  已自动导出，若新开终端请执行:"
echo "  export LD_LIBRARY_PATH=\$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparselt/lib:\$LD_LIBRARY_PATH"
echo "============================================================"

echo ""
echo "============================================================"
echo "✅ 环境配置完成！"
echo ""
echo "接下来运行推理测试："
echo "  source $VENV_DIR/bin/activate"
echo "  export PYTHONPATH=$EMBODIED_DIR:\$PYTHONPATH"
echo "  python $PROJECT_DIR/inference_test.py"
echo ""
echo "或使用 YOLOv8 风格检测工具："
echo "  python $PROJECT_DIR/detect.py --image 图片路径 --categories person car"
echo "============================================================"