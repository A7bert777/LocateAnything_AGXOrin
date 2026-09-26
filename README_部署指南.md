 # LocateAnything-3B 在 AGX Orin 64GB 上的部署指南

## 一、资源位置汇总

| 资源 | 位置 | 说明 |
|------|------|------|
| 官方模型权重 | `nvidia/LocateAnything-3B` | 基于 Qwen2.5-3B-Instruct，BF16，约 7.66GB |
| 官方 demo 代码 | `NVlabs/Eagle` → `Embodied/` | 核心脚本 `locateanything_worker.py` |
| 在线 Demo | `nvidia/LocateAnything` (HF Spaces) | 参考交互界面 |

---

## 二、手动下载命令

> **⚠️ 关于 PEP 668 限制**：Ubuntu 24.04 的系统 Python 受保护，禁止直接 `pip install`。
> 必须先创建并激活虚拟环境（venv）。

### 1. 创建虚拟环境并安装下载工具
```bash
cd <本项目根目录>

# 创建虚拟环境（只需执行一次）
python3 -m venv venv

# 激活虚拟环境（每次新终端都要执行）
source venv/bin/activate

# 升级 pip 并安装下载工具
pip install -U pip
pip install -U "huggingface_hub[cli]"
```

> **大陆网络加速**（可选）：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 2. 下载模型权重（BF16 原始精度）
```bash
cd <本项目根目录>

# 方式 A：新版 hf 命令（推荐，支持断点续传）
hf download nvidia/LocateAnything-3B --local-dir ./models/LocateAnything-3B

# 方式 B：旧版命令
# huggingface-cli download nvidia/LocateAnything-3B --local-dir ./models/LocateAnything-3B
```

模型文件构成：
```
model-00001-of-00002.safetensors   (约 4.96GB)
model-00002-of-00002.safetensors   (约 2.70GB)
model.safetensors.index.json
+ 自定义推理代码（modeling_locateanything.py 等）
+ tokenizer / processor 配置
```

### 3. 下载官方 demo 代码
```bash
cd <本项目根目录>
git clone https://github.com/NVlabs/Eagle.git ./code/Eagle
```

---

## 二之二、一键脚本方式（推荐）

如果觉得手动步骤麻烦，直接运行项目里的脚本即可（脚本已自动处理 venv）：
```bash
cd <本项目根目录>
./download.sh
```
脚本会自动：创建虚拟环境 → 安装下载工具 → 下载模型权重 → 检查官方代码（已随仓库分发则跳过）。

核心推理脚本路径：
```
code/Eagle/Embodied/locateanything_worker.py
```

---

## 三、⚠️ 关键技术提示（部署前必读）

### 1. 官方暂不支持 TensorRT
根据模型官方 README 的明确说明：
> "The inference setup uses standard VLM generation with BF16 precision and KV cache.
> **TensorRT, TensorRT-LLM, and Triton are not yet supported.**"

这意味着：
- **FP16 部署**：可以走 PyTorch + CUDA（Orin 支持 FP16），但官方代码默认用 **BF16**（`torch.bfloat16`）。Orin 的 GPU（Ampere 架构）对 BF16 和 FP16 支持都有限，需要验证。
- **INT8 部署**：无法直接套用官方 TensorRT 方案，需要自行做 **模型量化 + TensorRT 引擎构建**，属于额外的工程量。

### 2. 模型需要 `trust_remote_code=True`
模型仓库使用大量自定义代码（`modeling_locateanything.py`、`modeling_vit.py`、`mask_magi_utils.py` 等），不是标准 transformers 架构，加载时必须：
```python
AutoModel.from_pretrained(model_path, trust_remote_code=True)
```

### 3. 官方推荐依赖版本
```
transformers==4.57.1
tokenizers==0.22.0
numpy==1.25.0
Pillow==11.1.0
opencv-python-headless==4.11.0.86
peft
torchvision
decord==0.6.0
```

### 4. 官方推理 API 示例
```python
from locateanything_worker import LocateAnythingWorker

worker = LocateAnythingWorker("nvidia/LocateAnything-3B")

# 目标检测（用自然语言列出类别）
print(worker.detect(img, ["person", "car", "bicycle"])["answer"])

# 视觉定位
print(worker.ground_multi(img, "people wearing red shirts")["answer"])

# 文字检测
print(worker.detect_text(img)["answer"])

# GUI 元素定位
print(worker.ground_gui(img, "the search button", output_type="point")["answer"])
```

---

## 四、FP16 / INT8 部署路线图（后续实施）

### 路线 A：FP16 部署（较简单，优先）
1. 先以官方 BF16 代码在 Orin 上跑通基准（验证环境可用）。
2. 将 `dtype` 改为 `torch.float16`，测试 Orin 上的吞吐与精度。
   - 注意：Orin（Ampere 架构，SM87）FP16 加速正常，但 BF16 支持有限。
3. 验证 `detect` / `ground_multi` 输出是否满足目标检测需求（与 YOLOv8 对比）。

### 路线 B：INT8 部署（较复杂，需要量化）
1. 确认 Orin 的 TensorRT 版本（JetPack 版本对应）。
2. 将模型导出为 ONNX（需手动处理自定义模块）。
3. 使用 TensorRT 构建 INT8 引擎（需校准数据集）。
4. 由于官方不支持 TRT，需要参考社区 `locate-anything-tensorrt` 的实现思路，
   并针对 Orin（SM87）重新适配。
5. 备选方案：PyTorch 的 INT8 量化（如 `torch.quantization` 或 bitsandbytes）。

> **建议顺序**：先路线 A 跑通 FP16，再尝试路线 B 的 INT8 优化。

---

## 五、磁盘空间预估

| 项目 | 大小 |
|------|------|
| 模型权重（BF16 原始） | ~7.66GB |
| 官方代码仓库 | ~几 MB |
| ONNX 中间产物（后续） | ~7GB |
| TensorRT INT8 引擎（后续） | ~2GB |

AGX Orin 64GB 内存充足，磁盘需预留至少 **30GB+** 空间。