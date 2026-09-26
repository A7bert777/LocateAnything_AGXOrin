#!/bin/bash
# ============================================================
# LocateAnything-3B 便捷启动脚本
# 自动设置 LD_LIBRARY_PATH + PYTHONPATH，然后运行指定脚本
#
# 用法:
#   ./run.sh inference_test.py
#   ./run.sh detect.py --image test.jpg --categories person car
# ============================================================

# 自动获取脚本所在目录（相对路径，支持项目整体移动/重命名）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/venv310"
EMBODIED_DIR="$PROJECT_DIR/code/Eagle/Embodied"
MODEL_DIR="$PROJECT_DIR/models/LocateAnything-3B"

# 设置 torch 所需库路径（libcusparseLt.so.0）
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

# 设置推理代码路径
export PYTHONPATH="$EMBODIED_DIR:$MODEL_DIR:$PYTHONPATH"

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 运行传入的命令
"$VENV_DIR/bin/python" "$@"