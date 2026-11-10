# سكريبت يشغّل الموديل المدرَّب فعليًا على صور حقيقية ويرسم الصناديق (bounding boxes)
# عليها، ويحفظ كل نتيجة كصورة JPG جاهزة للعرض في البرزنتيشن.
# ملحوظة: نفس المنطق بالظبط اللي في detection/detect.py، بس بنضيف خطوة الرسم والحفظ.

import os

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "best.pt")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "detection_demo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# لون مختلف لكل نوع مشكلة، عشان يبقى واضح بصريًا في العرض
COLORS = {
    "pothole": (220, 50, 50),    # أحمر
    "garbage": (60, 130, 60),    # أخضر
    "puddle": (50, 110, 200),    # أزرق
}


def load_detection_model():
    model = YOLO(MODEL_PATH)
    return model


def detect_problem(image_path, model, confidence_threshold=0.4):
    results = model(image_path, verbose=False)[0]

    problems = []
    for box in results.boxes:
        confidence = float(box.conf[0])
        if confidence < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

        problems.append(
            {
                "problem_type": results.names[class_id],
                "confidence": confidence,
                "bounding_box": {
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max),
                },
            }
        )

    return problems


def draw_and_save(image_path, detections, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        box = det["bounding_box"]
        color = COLORS.get(det["problem_type"], (255, 255, 0))
        x_min, y_min, x_max, y_max = box["x_min"], box["y_min"], box["x_max"], box["y_max"]

        # نرسم صندوق سميك واضح
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)

        # نكتب النوع ونسبة الثقة فوق الصندوق
        label = f"{det['problem_type']} {det['confidence']*100:.0f}%"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_y = max(0, y_min - text_h - 8)
        draw.rectangle(
            [x_min, label_y, x_min + text_w + 10, label_y + text_h + 8],
            fill=color,
        )
        draw.text((x_min + 5, label_y + 2), label, fill=(255, 255, 255), font=font)

    image.save(output_path, quality=95)
    print(f"  تم الحفظ في: {output_path}")


if __name__ == "__main__":
    dataset_dir = os.path.join(
        SCRIPT_DIR, "..", "..", "phase 2 _ data set", "dataset", "dataset"
    )

    test_images = [
        os.path.join(
            dataset_dir,
            "Pothole detection2",
            "test",
            "images",
            "img-105_jpg.rf.3fe9dff3d1631e79ecb480ff403bcb86.jpg",
        ),
        os.path.join(
            dataset_dir,
            "puddle detection3",
            "test",
            "images",
            "-_PNG.rf.31096e80c5900a83e622a130f167a10a.jpg",
        ),
        os.path.join(
            dataset_dir,
            "Garbage detection1",
            "Garbage Detection.v1i.yolov8",
            "test",
            "images",
            "img128_jpg.rf.4330285cd01cc7116e34bba831c79458.jpg",
        ),
    ]

    print("تحميل الموديل المدرَّب...")
    model = load_detection_model()
    print("تم التحميل.\n")

    for image_path in test_images:
        image_name = os.path.basename(image_path)
        print(f"صورة: {image_name}")

        detections = detect_problem(image_path, model)
        print(f"  عدد الاكتشافات: {len(detections)}")
        for d in detections:
            print(f"    - {d['problem_type']} (ثقة: {d['confidence']*100:.1f}%)")

        output_filename = f"result_{image_name.rsplit('.', 1)[0]}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        draw_and_save(image_path, detections, output_path)
        print()

    print(f"كل الصور النهائية محفوظة في: {OUTPUT_DIR}")
