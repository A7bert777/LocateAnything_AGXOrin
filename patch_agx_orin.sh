#!/bin/bash
# ============================================================
# AGX Orin 适配补丁脚本（幂等，可重复执行）
#
# 修复 3 个在 Jetson 上必然出现的环境问题：
#   1. transformers 5.x 误判 Jetson torch 版本 → 禁用 PyTorch 后端
#   2. PyPI 版 torchvision 与 NVIDIA 定制版 torch ABI 不匹配（nms 算子缺失）
#   3. decord 在 aarch64 无预编译 wheel，但被 transformers 静态检查拦截
#
# 用法: ./patch_agx_orin.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv310"
SITE_PKGS="$VENV_DIR/lib/python3.10/site-packages"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 未找到 venv310，请先执行 ./setup.sh"
    exit 1
fi

# torch 需要 libcusparseLt.so.0
export LD_LIBRARY_PATH="$SITE_PKGS/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

echo "============================================================"
echo "AGX Orin 适配补丁"
echo "============================================================"

# ------------------------------------------------------------
# 修复 1：锁定 transformers < 5（4.57.1 才是正确版本）
# ------------------------------------------------------------
echo ""
echo "[1/3] 检查 transformers 版本..."
CUR_TV=$("$VENV_DIR/bin/pip" show transformers 2>/dev/null | awk '/^Version:/{print $2}')
case "$CUR_TV" in
    4.*) echo "    ✅ transformers $CUR_TV（4.x，无需处理）" ;;
    *)
        echo "    ⚠️  当前 transformers $CUR_TV 为 5.x，会误判 Jetson torch 版本"
        echo "    正在降级到 4.57.1..."
        "$VENV_DIR/bin/pip" install "transformers==4.57.1"
        echo "    ✅ 已降级到 4.57.1"
        ;;
esac

# ------------------------------------------------------------
# 修复 2：安装图像/视频依赖（必须 --no-deps，保护 NVIDIA torch）
# ------------------------------------------------------------
echo ""
echo "[2/3] 安装 torchvision / opencv / lmdb..."
echo "    （使用 --no-deps，防止通用版 torch 覆盖 Jetson GPU wheel）"
"$VENV_DIR/bin/pip" install --no-deps \
    "torchvision==0.20.0" \
    "opencv-python-headless==4.10.0.84" \
    "lmdb"

# 给 torchvision/__init__.py 打补丁：跳过 ABI 不兼容的 _meta_registrations
TV_INIT="$SITE_PKGS/torchvision/__init__.py"
if [ -f "$TV_INIT" ] && ! grep -q "AGX Orin 适配" "$TV_INIT"; then
    echo "    正在为 torchvision 打 ABI 兼容补丁..."
    cp "$TV_INIT" "$TV_INIT.bak"
    # 将 _meta_registrations 的导入包进 try/except，ABI 不兼容时降级跳过
    python3 - "$TV_INIT" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    src = f.read()
old = "from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip"
new = (
    "try:\n"
    "    from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip\n"
    "except Exception:  # noqa: BLE001\n"
    "    # AGX Orin 适配：PyPI 版 torchvision 的 _meta_registrations 与 NVIDIA 定制版\n"
    "    # torch 2.5.0a0 (nv24.08) ABI 不匹配（报 operator torchvision::nms does not exist）。\n"
    "    # 本工程只用 transforms.functional 的纯 Python 函数，故降级跳过。\n"
    "    from torchvision import datasets, io, models, ops, transforms, utils  # usort:skip"
)
if old in src:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src.replace(old, new, 1))
    print("    ✅ torchvision 补丁已应用")
else:
    print("    ℹ️  torchvision 补丁已存在或格式不符，跳过")
PYEOF
else
    echo "    ✅ torchvision 补丁已应用，跳过"
fi

# ------------------------------------------------------------
# 修复 3：decord 空实现（aarch64 无 wheel，仅视频用，本工程不需要）
# ------------------------------------------------------------
echo ""
echo "[3/3] 检查 decord..."
if "$VENV_DIR/bin/python" -c "import decord" 2>/dev/null; then
    echo "    ✅ decord 可用，跳过"
else
    echo "    ⚠️  aarch64 无 decord wheel，创建空实现（仅供图像检测无需视频解码）"
    mkdir -p "$SITE_PKGS/decord"
    cat > "$SITE_PKGS/decord/__init__.py" <<'PYEOF'
"""decord 空实现（AGX Orin / aarch64 适配）

decord 在 aarch64 无预编译 wheel。上游模型代码中它本就是可选依赖
（try/except 包裹，仅供视频解码）。本工程只做静态图像检测，不需要它。
但 transformers 的 check_imports() 会静态扫描源码，decord 无法导入即报错，
故提供空实现以绕过检查。

如需真实视频解码，请从源码编译: https://github.com/dmlc/decord
"""
__version__ = "0.0.0-stub"
__all__ = []
PYEOF
    echo "    ✅ decord 空实现已创建"
fi

# ------------------------------------------------------------
# 验证
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "验证修复结果..."
echo "============================================================"
"$VENV_DIR/bin/python" - <<'PYEOF'
import torch
ok = []
ok.append(f"torch        : {torch.__version__}")
ok.append(f"CUDA available: {torch.cuda.is_available()}")
try:
    import transformers
    from transformers.utils import is_torch_available
    ok.append(f"transformers : {transformers.__version__}")
    ok.append(f"is_torch_available(): {is_torch_available()}")
except Exception as e:
    ok.append(f"transformers : ❌ {e}")
try:
    from torchvision.transforms import functional as TF
    import torch as t
    TF.resize(t.rand(1,3,8,8), [4,4])
    import torchvision
    ok.append(f"torchvision  : {torchvision.__version__} (TF 功能正常)")
except Exception as e:
    ok.append(f"torchvision  : ❌ {e}")
for m in ok:
    print("    " + m)
PYEOF

echo ""
echo "============================================================"
echo "✅ 补丁完成！现在可以运行:"
echo "   ./run.sh detect.py --image 002.jpg --categories car bicycle shoe"
echo "============================================================"