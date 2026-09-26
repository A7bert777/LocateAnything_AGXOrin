# LocateAnything-3B on AGX Orin 64GB

在 NVIDIA AGX Orin 64GB 上部署与运行 **LocateAnything-3B**（多模态目标检测 / 视觉定位模型）的完整工程。

基于官方模型 [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) 与官方代码 [`NVlabs/Eagle`](https://github.com/NVlabs/Eagle)，提供开箱即用的命令行检测工具。

---

## 特性

- **YOLOv8 风格命令行接口**：`detect.py` 支持自定义类别列表、输出像素坐标、保存标注图与 JSON
- **相对路径设计**：项目可任意移动、重命名，无需修改代码
- **一键环境重建**：`download.sh` + `setup.sh` 自动完成模型下载与环境配置
- **AGX Orin 专用适配**：解决 Jetson 上 `libcusparseLt.so.0` 缺失、numpy 版本冲突、PEP 668 限制等问题

---

## 硬件与系统要求

| 项目 | 要求 |
|------|------|
| 硬件 | NVIDIA **AGX Orin 64GB** |
| 系统 | JetPack 6.2 / Ubuntu 24.04 |
| Python | 3.10（系统需有 `python3.10`） |
| Python 版本 | 磁盘剩余空间建议 ≥ **20GB** |
| 网络 | 首次需联网下载模型（约 7.7GB） |

---

## 快速开始

> **⚠️ 重要说明**
> 本仓库**不包含**模型权重（约 7.3GB）和虚拟环境（约 2.2GB）——它们体积过大且与本机绑定。
> 请按下面三步在本地自动重建，整个过程无需手动干预。

### 步骤 1：克隆仓库

```bash
git clone git@github.com:A7bert777/LocateAnything_AGXOrin.git
cd LocateAnything_AGXOrin

# 首次使用，赋予脚本可执行权限
chmod +x download.sh setup.sh run.sh
```

### 步骤 2：下载模型权重

```bash
./download.sh
```

该脚本会自动：
1. 创建 `venv/` 下载环境
2. 从 HuggingFace 下载模型权重到 `models/LocateAnything-3B/`（约 7.7GB）
3. 检查官方代码 `code/Eagle/`（**已随仓库分发，会直接跳过**）

> **大陆网络加速**（可选）：编辑 `download.sh`，取消 `export HF_ENDPOINT=https://hf-mirror.com` 一行的注释。

### 步骤 3：配置推理环境

```bash
./setup.sh
```

该脚本会自动：
1. 创建 Python 3.10 虚拟环境 `venv310/`
2. 安装 **Jetson 专用 PyTorch wheel**（约 807MB，CUDA 12.6）
3. 安装 `libcusparseLt.so.0`（Jetson 上 torch 的硬依赖）
4. 安装 `requirements.txt` 中的全部推理依赖

### 步骤 4：运行检测

```bash
# 基本用法
./run.sh detect.py --image 002.jpg --categories car bicycle shoe

# 指定输出图片与 JSON 结果
./run.sh detect.py --image 002.jpg --categories person car \
    --output result.png --json result.json

# 使用 FP16 精度（Orin 上通常比 BF16 更快）
./run.sh detect.py --image 002.jpg --categories car --fp16

# 运行完整自检（环境信息 + 模型加载 + 检测）
./run.sh inference_test.py
```

**输出示例：**

```
[LocateAnything] 输入图片: 002.jpg (640x480)
[LocateAnything] 设备: cuda, 精度: torch.bfloat16
[LocateAnything] ✅ 模型加载完成，耗时 6.26 秒

============================================================
✅ 检测完成！共检测到 2 个目标
   推理耗时: 2.18 秒
============================================================
   [1] object: (314.9, 50.9) -> (453.1, 128.2) 尺寸: 138.2x77.3
   [2] object: (433.9, 112.8) -> (545.9, 219.8) 尺寸: 112.0x107.0

[LocateAnything] 标注图片已保存: detection_result.png
```

---

## 项目结构

```
LocateAnything_AGXOrin/
├── run.sh                      # 启动脚本（自动设置环境变量并运行）
├── setup.sh                    # 环境配置脚本（重建 venv310 + 装 torch）
├── download.sh                 # 资源下载脚本（模型权重）
├── detect.py                   # 目标检测工具（YOLOv8 风格 CLI）
├── inference_test.py           # 推理自检脚本
├── requirements.txt            # 推理依赖清单
├── README_部署指南.md           # 详细部署指南（含原理说明）
│
├── code/Eagle/Embodied/        # NVIDIA 官方推理代码（随仓库分发）
│   └── locateanything_worker.py   # 核心推理封装类
│
├── models/                     # 模型权重（gitignore，需下载）
│   └── LocateAnything-3B/
│
├── venv/                       # 下载用环境（gitignore，自动创建）
└── venv310/                    # 推理环境（gitignore，自动创建）
```

---

## 命令行参数（detect.py）

| 参数 | 必填 | 说明 |
|------|------|------|
| `--image` | ✅ | 输入图片路径 |
| `--categories` | ✅ | 目标类别列表，空格分隔，如 `person car bicycle` |
| `--output` | | 标注图输出路径（默认 `detection_result.png`） |
| `--json` | | 输出 JSON 结果路径 |
| `--model` | | 模型路径（默认 `models/LocateAnything-3B`） |
| `--device` | | `cuda` 或 `cpu`（默认 `cuda`） |
| `--fp16` | | 使用 FP16 精度（默认 BF16） |

---

## 重要提示

### 1. 官方暂不支持 TensorRT
据模型官方 README 明确说明：

> *"The inference setup uses standard VLM generation with BF16 precision and KV cache. **TensorRT, TensorRT-LLM, and Triton are not yet supported.**"*

因此本工程采用 **PyTorch + CUDA** 路径。转 TensorRT 需自行为自定义算子（`magi_attention`、MoonViT 等）编写 C++ 插件，工程量极大。

### 2. 精度选择
- 官方默认 **BF16**（`torch.bfloat16`）
- Orin（Ampere 架构，SM87）上 **FP16 吞吐通常更优**，可用 `--fp16` 对比测试

### 3. 首次运行较慢属正常现象
- **冷启动**：重启后首次运行需从磁盘读取 7.3GB 权重，加载约 30s
- **热启动**：权重进入页缓存后，加载约 6~7s，推理约 2~4s

### 4. 功耗模式
默认 `MAXN`（模式 0）性能最高。若长时间连续推理导致过热，可通过 `sudo nvpmodel -m 3` 切换到 50W 模式。

---

## 依赖版本（关键）

```
transformers>=4.57.1
tokenizers>=0.22.0
numpy==1.26.4          # torch 2.5.0 要求 numpy<2
Pillow>=11.1.0
safetensors>=0.4.0
accelerate>=1.0.0
peft>=0.20.0
nvidia-cusparselt-cu12==0.8.1   # 提供 libcusparseLt.so.0
```

**PyTorch 需单独安装**（Jetson 专用 wheel，由 `setup.sh` 自动处理）：
```
torch-2.5.0a0+...-cp310-cp310-linux_aarch64.whl
```

---

## 参考资源

| 资源 | 链接 |
|------|------|
| 官方模型 | [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) |
| 官方代码 | [NVlabs/Eagle](https://github.com/NVlabs/Eagle) |
| 在线 Demo | [nvidia/LocateAnything](https://huggingface.co/spaces/nvidia/LocateAnything) |
| 详细部署指南 | 见本仓库 `README_部署指南.md` |

---

## 许可

本工程中的脚本（`run.sh`、`setup.sh`、`download.sh`、`detect.py`、`inference_test.py`）为部署工具。
模型权重与 `code/Eagle` 中的官方代码遵循其各自的原始许可（详见 `code/Eagle/LICENSE`）。