#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocateAnything-3B 推理测试脚本
在 AGX Orin 上验证模型加载 + 目标检测功能（BF16 精度）

用法:
    source venv310/bin/activate
    export PYTHONPATH=$(pwd)/code/Eagle/Embodied:$PYTHONPATH
    python inference_test.py
"""

import os
import sys
import time

# 项目根目录：自动取本脚本所在目录（相对路径，支持项目整体移动/重命名）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "LocateAnything-3B")
EMBODIED_DIR = os.path.join(PROJECT_DIR, "code", "Eagle", "Embodied")

# 将官方推理代码目录加入 sys.path
if EMBODIED_DIR not in sys.path:
    sys.path.insert(0, EMBODIED_DIR)

import torch
from PIL import Image, ImageDraw
from locateanything_worker import LocateAnythingWorker


def draw_boxes(image: Image.Image, boxes: list[dict], categories: list[str]) -> Image.Image:
    """在图片上画出检测框，返回绘制后的图片。"""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image


def main():
    print("=" * 60)
    print("LocateAnything-3B 在 AGX Orin 上的推理测试")
    print("=" * 60)

    # 1. 确认环境
    print(f"\n[1] 环境信息:")
    print(f"    torch 版本: {torch.__version__}")
    print(f"    CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA 版本: {torch.version.cuda}")
        print(f"    GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"    GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"    模型路径: {MODEL_DIR}")
    print(f"    模型路径存在: {os.path.exists(MODEL_DIR)}")

    # 2. 加载模型（BF16 默认精度）
    print(f"\n[2] 加载模型（BF16）...")
    t0 = time.time()
    worker = LocateAnythingWorker(
        model_path=MODEL_DIR,
        device="cuda",
        dtype=torch.bfloat16,   # 官方默认 BF16
    )
    print(f"    ✅ 模型加载完成，耗时 {time.time() - t0:.2f} 秒")

    # 3. 准备测试图片（用模型自带的一张示例图）
    test_img_path = os.path.join(MODEL_DIR, "assets", "coco_lvis.png")
    if not os.path.exists(test_img_path):
        # 如果没有示例图，生成一张纯色测试图
        print(f"    ⚠️ 未找到示例图，生成测试图")
        img = Image.new("RGB", (640, 480), color=(255, 255, 255))
    else:
        print(f"    使用示例图: {test_img_path}")
        img = Image.open(test_img_path).convert("RGB")

    print(f"    图片尺寸: {img.size}")

    # 4. 目标检测测试
    categories = ["person", "car", "bicycle"]
    print(f"\n[3] 目标检测测试，类别: {categories}")
    t1 = time.time()
    result = worker.detect(img, categories)
    detect_time = time.time() - t1
    print(f"    ✅ 检测完成，耗时 {detect_time:.2f} 秒")
    print(f"    检测结果:")
    answer = result["answer"]
    print(f"    {answer}")
    print()


    # 5. 解析检测框
    boxes = LocateAnythingWorker.parse_boxes(answer, img.width, img.height)
    print(f"[4] 检测到 {len(boxes)} 个目标框:")
    for i, box in enumerate(boxes):
        print(f"    框 {i+1}: x1={box['x1']:.1f}, y1={box['y1']:.1f}, "
              f"x2={box['x2']:.1f}, y2={box['y2']:.1f} "
              f"(宽={box['x2']-box['x1']:.1f}, 高={box['y2']-box['y1']:.1f})")

    # 6. 保存标注后的图片
    if boxes:
        img_drawn = draw_boxes(img, boxes, categories)
        out_path = os.path.join(PROJECT_DIR, "detection_result.png")
        img_drawn.save(out_path)
        print(f"\n[5] ✅ 检测结果已保存到: {out_path}")

    print("\n" + "=" * 60)
    print("✅ 推理测试完成！LocateAnything-3B 在 AGX Orin 上运行成功")
    print("=" * 60)


if __name__ == "__main__":
    main()