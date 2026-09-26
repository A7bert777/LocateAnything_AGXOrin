#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocateAnything-3B 目标检测工具（YOLOv8 风格命令行接口）

在 AGX Orin 上实现类似 YOLOv8 的目标检测功能：
    - 支持自定义类别列表
    - 输出检测框坐标（像素坐标）
    - 保存标注后的图片
    - 支持输出 JSON 结果

用法:
    source venv310/bin/activate
    export PYTHONPATH=$(pwd)/code/Eagle/Embodied:$PYTHONPATH

    # 基本检测
    python detect.py --image test.jpg --categories person car bicycle

    # 指定输出路径
    python detect.py --image test.jpg --categories person car --output result.jpg

    # 输出 JSON 结果
    python detect.py --image test.jpg --categories person --json result.json
"""

import argparse
import json
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

import re

# 固定的类别颜色（用于画框）
COLORS = [
    (255, 0, 0),     # 红
    (0, 255, 0),     # 绿
    (0, 0, 255),     # 蓝
    (255, 255, 0),   # 黄
    (255, 0, 255),   # 品红
    (0, 255, 255),   # 青
    (255, 128, 0),   # 橙
    (128, 0, 255),   # 紫
    (0, 128, 255),   # 天蓝
    (255, 0, 128),   # 粉红
]


class LocateAnythingDetector:
    """YOLOv8 风格的目标检测器，封装 LocateAnything-3B 模型。"""

    def __init__(self, model_path: str = MODEL_DIR, device: str = "cuda", dtype=torch.bfloat16):
        """
        初始化检测器。

        Args:
            model_path: 模型路径
            device: "cuda" 或 "cpu"
            dtype: torch.bfloat16 (BF16) 或 torch.float16 (FP16)
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.worker = None
        self._load_time = 0.0

    def load(self):
        """加载模型（BF16/FP16）。"""
        print(f"[LocateAnything] 正在加载模型: {self.model_path}")
        print(f"[LocateAnything] 设备: {self.device}, 精度: {self.dtype}")
        t0 = time.time()
        self.worker = LocateAnythingWorker(
            model_path=self.model_path,
            device=self.device,
            dtype=self.dtype,
        )
        self._load_time = time.time() - t0
        print(f"[LocateAnything] ✅ 模型加载完成，耗时 {self._load_time:.2f} 秒")
        return self

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """
        执行目标检测（YOLOv8 风格）。

        Args:
            image: PIL Image (RGB)
            categories: 类别列表，如 ["person", "car", "bicycle"]

        Returns:
            dict: {
                "boxes": [{"x1","y1","x2","y2","category"}...],
                "raw_answer": 模型原始输出,
                "time": 推理耗时(秒),
            }
        """
        if self.worker is None:
            self.load()

        t0 = time.time()
        result = self.worker.detect(image, categories, **kwargs)
        elapsed = time.time() - t0

        answer = result["answer"]
        w, h = image.size

        # 联合解析：从模型输出文本中同时提取类别标签和框坐标
        # 模型输出格式: <类别文本><box><x1><y1><x2><y2></box> ...
        boxes = self._parse_detections(answer, w, h)

        return {
            "boxes": boxes,
            "raw_answer": answer,
            "time": elapsed,
        }

    @staticmethod
    def _parse_detections(answer: str, w: int, h: int) -> list[dict]:
        """
        从模型输出文本中解析出类别+框的配对列表。

        模型输出中，每个检测目标的形式为:
            <类别词><box><x1><y1><x2><y2></box>

        其中坐标是 [0, 1000] 归一化整数。

        Returns:
            list[dict]: 每个元素包含 "category", "x1", "y1", "x2", "y2"（像素坐标）
        """
        detections = []
        # 匹配模式: 类别文本(非<>) 后跟 <box><数字><数字><数字><数字></box>
        pattern = r'([^<>\n]+?)\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>'

        for m in re.finditer(pattern, answer):
            category_raw = m.group(1).strip()
            x1, y1, x2, y2 = [int(g) for g in m.groups()[1:]]

            # 模型输出的类别词可能包含 </c> 分隔符，清理它
            category = category_raw.replace("</c>", "").strip()
            if not category:
                category = "object"

            detections.append({
                "category": category,
                "x1": x1 / 1000 * w,
                "y1": y1 / 1000 * h,
                "x2": x2 / 1000 * w,
                "y2": y2 / 1000 * h,
            })

        # 如果联合解析没结果，退回到纯坐标解析
        if not detections:
            boxes = LocateAnythingWorker.parse_boxes(answer, w, h)
            for box in boxes:
                box["category"] = "object"
            detections = boxes

        return detections

    def ground(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """视觉定位（自然语言描述）。"""
        if self.worker is None:
            self.load()
        result = self.worker.ground_multi(image, phrase, **kwargs)
        w, h = image.size
        boxes = self._parse_detections(result["answer"], w, h)
        # 视觉定位时，每个框都属于同一个 phrase
        for box in boxes:
            box["category"] = phrase
        return {
            "boxes": boxes,
            "raw_answer": result["answer"],
        }


def draw_detection(image: Image.Image, boxes: list[dict]) -> Image.Image:
    """在图片上绘制检测框和标签。"""
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 标签文字
        label = box.get("category", f"obj{i}")
        text = f"{label}"
        text_bbox = draw.textbbox((x1, y1), text)
        draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill=color)
        draw.text((text_bbox[0], text_bbox[1]), text, fill=(255, 255, 255))

    return image


def main():
    parser = argparse.ArgumentParser(description="LocateAnything-3B 目标检测（YOLOv8 风格）")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--categories", required=True, nargs="+", help="目标类别列表，如: person car bicycle")
    parser.add_argument("--output", help="输出图片路径（默认: detection_result.png）")
    parser.add_argument("--json", help="输出 JSON 结果路径（可选）")
    parser.add_argument("--model", default=MODEL_DIR, help="模型路径")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 精度（默认 BF16）")
    args = parser.parse_args()

    # 1. 检查输入图片
    if not os.path.exists(args.image):
        print(f"❌ 错误: 图片不存在: {args.image}")
        sys.exit(1)

    # 2. 加载图片
    img = Image.open(args.image).convert("RGB")
    print(f"[LocateAnything] 输入图片: {args.image} ({img.width}x{img.height})")

    # 3. 选择精度
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    precision_name = "FP16" if args.fp16 else "BF16"

    # 4. 初始化检测器
    detector = LocateAnythingDetector(
        model_path=args.model,
        device=args.device,
        dtype=dtype,
    )
    detector.load()

    # 5. 执行检测
    print(f"\n[LocateAnything] 检测目标类别: {args.categories} (精度: {precision_name})")
    result = detector.detect(img, args.categories)

    # 6. 输出结果
    boxes = result["boxes"]
    print(f"\n{'='*60}")
    print(f"✅ 检测完成！共检测到 {len(boxes)} 个目标")
    print(f"   推理耗时: {result['time']:.2f} 秒")
    print(f"{'='*60}")
    for i, box in enumerate(boxes):
        print(f"   [{i+1}] {box.get('category','obj')}: "
              f"({box['x1']:.1f}, {box['y1']:.1f}) -> "
              f"({box['x2']:.1f}, {box['y2']:.1f}) "
              f"尺寸: {box['x2']-box['x1']:.1f}x{box['y2']-box['y1']:.1f}")

    if not boxes:
        print("   (未检测到任何目标)")

    # 7. 保存标注图片
    if boxes:
        img_drawn = draw_detection(img.copy(), boxes)
        out_path = args.output or "detection_result.png"
        img_drawn.save(out_path)
        print(f"\n[LocateAnything] 标注图片已保存: {out_path}")

    # 8. 保存 JSON 结果
    if args.json:
        json_data = {
            "image": args.image,
            "categories": args.categories,
            "precision": precision_name,
            "time": result["time"],
            "boxes": boxes,
            "raw_answer": result["raw_answer"],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"[LocateAnything] JSON 结果已保存: {args.json}")


if __name__ == "__main__":
    main()