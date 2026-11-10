# سكريبت ديمو بصري لمكوّن "كشف التكرار" (Duplicate Detection)
# بياخد صورتين حقيقيتين من الداتا، يحطهم جنب بعض، ويكتب تحتهم قرار الموديل (نفس similarity.py بالظبط)
# الهدف: نثبت للدكتور إن الفلاتر (النوع -> المسافة -> الوقت -> التشابه البصري) شغالة فعليًا على حالات حقيقية

import os
import sys
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from similarity import load_clip_model, is_possible_duplicate  # noqa: E402

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "duplicate_demo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "..", "phase 2 _ data set", "dataset", "dataset")
POTHOLE_TRAIN = os.path.join(DATASET_DIR, "Pothole detection2", "train", "images")
GARBAGE_TEST = os.path.join(
    DATASET_DIR, "Garbage detection1", "Garbage Detection.v1i.yolov8", "test", "images"
)

DECISION_LABELS = {
    "duplicate_confirmed": ("DUPLICATE", (190, 40, 40)),
    "possible_duplicate_needs_review": ("POSSIBLE DUPLICATE - NEEDS REVIEW", (190, 140, 20)),
    "different": ("DIFFERENT", (40, 130, 60)),
}


def make_side_by_side(image_path_a, image_path_b, result, title, out_path):
    """بيحط الصورتين جنب بعض فوق، وتحتهم شريط فيه القرار والتفاصيل"""
    img_a = Image.open(image_path_a).convert("RGB")
    img_b = Image.open(image_path_b).convert("RGB")

    target_h = 480
    img_a = img_a.resize((int(img_a.width * target_h / img_a.height), target_h))
    img_b = img_b.resize((int(img_b.width * target_h / img_b.height), target_h))

    gap = 16
    band_h = 150
    canvas_w = img_a.width + gap + img_b.width
    canvas_h = target_h + band_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 248, 242))
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (img_a.width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_big = ImageFont.truetype("arialbd.ttf", 30)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_title = font_big = font_small = ImageFont.load_default()

    label_text, color = DECISION_LABELS.get(result["decision"], ("?", (0, 0, 0)))

    draw.rectangle([0, target_h, canvas_w, canvas_h], fill=(250, 248, 242))
    draw.line([0, target_h, canvas_w, target_h], fill=(200, 195, 180), width=2)

    draw.text((14, target_h + 10), title, fill=(30, 30, 30), font=font_title)
    draw.rectangle([14, target_h + 46, 14 + 420, target_h + 92], fill=color)
    draw.text((26, target_h + 54), label_text, fill=(255, 255, 255), font=font_big)

    sim = result.get("similarity")
    sim_text = f"نسبة التشابه البصري: {sim*100:.1f}%" if sim is not None else "تشابه بصري: لم يُحسب"
    reason_text = f"السبب: {result['reason']}"
    draw.text((14, target_h + 100), sim_text, fill=(60, 60, 60), font=font_small)
    draw.text((420, target_h + 100), reason_text, fill=(60, 60, 60), font=font_small)

    canvas.save(out_path, quality=95)
    print(f"  تم الحفظ في: {out_path}")


if __name__ == "__main__":
    print("تحميل موديل CLIP...")
    model, preprocess = load_clip_model()
    print("تم التحميل.\n")

    pothole_img_1 = os.path.join(
        POTHOLE_TRAIN, "img-105_jpg.rf.3fe9dff3d1631e79ecb480ff403bcb86.jpg"
    )
    if not os.path.exists(pothole_img_1):
        all_imgs = sorted(os.listdir(POTHOLE_TRAIN))
        pothole_img_1 = os.path.join(POTHOLE_TRAIN, all_imgs[0])
        pothole_img_2 = os.path.join(POTHOLE_TRAIN, all_imgs[1])
    else:
        all_imgs = sorted(os.listdir(POTHOLE_TRAIN))
        other_imgs = [f for f in all_imgs if f not in pothole_img_1]
        pothole_img_2 = os.path.join(POTHOLE_TRAIN, other_imgs[0])

    garbage_imgs = sorted(os.listdir(GARBAGE_TEST))
    garbage_img_1 = os.path.join(GARBAGE_TEST, garbage_imgs[0])

    base_time = datetime(2026, 3, 1, 10, 0, 0)

    # ===== حالة 1: نفس الصورة بالظبط (محاكاة لبلاغين اتصوروا لنفس المكان) =====
    print("=" * 60)
    print("حالة 1: نفس الصورة بالظبط - بلاغين لنفس الحفرة بالظبط")
    report_1a = {
        "image_path": pothole_img_1, "lat": 30.0444, "lng": 31.2357,
        "timestamp": base_time, "problem_type": "pothole",
    }
    report_1b = {
        "image_path": pothole_img_1, "lat": 30.0444 + 0.0001, "lng": 31.2357 + 0.0001,
        "timestamp": datetime(2026, 3, 1, 14, 0, 0), "problem_type": "pothole",
    }
    result_1 = is_possible_duplicate(report_1a, report_1b, model, preprocess)
    print(f"  النتيجة: {result_1}")
    make_side_by_side(
        report_1a["image_path"], report_1b["image_path"], result_1,
        "حالة 1: نفس الصورة - بلاغين لنفس الحفرة",
        os.path.join(OUTPUT_DIR, "case1_same_image.jpg"),
    )
    print()

    # ===== حالة 2: صورتين حقيقيتين مختلفتين، نفس النوع، قريبين جغرافيًا وزمنيًا =====
    print("=" * 60)
    print("حالة 2: صورتين مختلفتين حقيقيتين - نفس النوع (pothole)، قريبين جغرافيًا وزمنيًا")
    report_2a = {
        "image_path": pothole_img_1, "lat": 30.0500, "lng": 31.2400,
        "timestamp": base_time, "problem_type": "pothole",
    }
    report_2b = {
        "image_path": pothole_img_2, "lat": 30.0500 + 0.0002, "lng": 31.2400 + 0.0002,
        "timestamp": datetime(2026, 3, 2, 9, 0, 0), "problem_type": "pothole",
    }
    result_2 = is_possible_duplicate(report_2a, report_2b, model, preprocess)
    print(f"  النتيجة: {result_2}")
    make_side_by_side(
        report_2a["image_path"], report_2b["image_path"], result_2,
        "حالة 2: صورتين مختلفتين حقيقيتين - نفس النوع",
        os.path.join(OUTPUT_DIR, "case2_different_images_same_type.jpg"),
    )
    print()

    # ===== حالة 3: نوع مختلف تمامًا =====
    print("=" * 60)
    print("حالة 3: نوع مختلف تمامًا (pothole مقابل garbage) -> يرفض فورًا من غير حساب تشابه بصري")
    report_3a = {
        "image_path": pothole_img_1, "lat": 30.0600, "lng": 31.2500,
        "timestamp": base_time, "problem_type": "pothole",
    }
    report_3b = {
        "image_path": garbage_img_1, "lat": 30.0600, "lng": 31.2500,
        "timestamp": base_time, "problem_type": "garbage",
    }
    result_3 = is_possible_duplicate(report_3a, report_3b, model, preprocess)
    print(f"  النتيجة: {result_3}")
    make_side_by_side(
        report_3a["image_path"], report_3b["image_path"], result_3,
        "حالة 3: نوع مختلف (pothole مقابل garbage)",
        os.path.join(OUTPUT_DIR, "case3_different_type.jpg"),
    )
    print()

    print(f"كل صور الديمو محفوظة في: {OUTPUT_DIR}")
