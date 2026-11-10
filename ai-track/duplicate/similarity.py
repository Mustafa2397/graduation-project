import math
from datetime import datetime

import numpy as np
import open_clip
import torch
from PIL import Image


def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    return model, preprocess


def get_image_embedding(image_path, model, preprocess):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(image_input)

    return embedding[0].numpy()


def haversine_distance(lat1, lng1, lat2, lng2):
    earth_radius_meters = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_meters * c


def is_possible_duplicate(
    report_a,
    report_b,
    model,
    preprocess,
    max_distance_meters=50,
    max_days_apart=7,
):
    if report_a["problem_type"] != report_b["problem_type"]:
        return {"decision": "different", "reason": "نوع مختلف", "similarity": None}

    distance = haversine_distance(
        report_a["lat"], report_a["lng"], report_b["lat"], report_b["lng"]
    )
    if distance > max_distance_meters:
        return {"decision": "different", "reason": "بعيدين جغرافيًا", "similarity": None}

    days_apart = abs(report_a["timestamp"] - report_b["timestamp"]).days
    if days_apart > max_days_apart:
        return {"decision": "different", "reason": "بعيدين زمنيًا", "similarity": None}

    embedding_a = get_image_embedding(report_a["image_path"], model, preprocess)
    embedding_b = get_image_embedding(report_b["image_path"], model, preprocess)

    similarity = float(
        np.dot(embedding_a, embedding_b)
        / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))
    )

    if similarity >= 0.85:
        decision = "duplicate_confirmed"
    elif similarity >= 0.65:
        decision = "possible_duplicate_needs_review"
    else:
        decision = "different"

    return {"decision": decision, "reason": "تشابه بصري", "similarity": similarity}


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pothole_images_dir = os.path.join(
        script_dir,
        "..",
        "..",
        "phase 2 _ data set",
        "dataset",
        "dataset",
        "Pothole detection2",
        "train",
        "images",
    )

    image_names = sorted(os.listdir(pothole_images_dir))[:2]
    image_path_1 = os.path.join(pothole_images_dir, image_names[0])
    image_path_2 = os.path.join(pothole_images_dir, image_names[1])

    print("الصورتين المستخدمتين:")
    print(f"  صورة 1: {image_names[0]}")
    print(f"  صورة 2: {image_names[1]}")
    print()

    print("تحميل موديل CLIP (أول مرة بيحمّل الأوزان من الإنترنت)...")
    model, preprocess = load_clip_model()
    print("تم تحميل الموديل.")
    print()

    base_timestamp = datetime(2026, 1, 1, 12, 0, 0)

    # مثال 1: صورتين مختلفتين، نفس problem_type، قريبين جغرافيًا وزمنيًا
    report_1a = {
        "image_path": image_path_1,
        "lat": 30.0444,
        "lng": 31.2357,
        "timestamp": base_timestamp,
        "problem_type": "pothole",
    }
    report_1b = {
        "image_path": image_path_2,
        "lat": 30.0444 + 0.0001,
        "lng": 31.2357 + 0.0001,
        "timestamp": datetime(2026, 1, 2, 12, 0, 0),
        "problem_type": "pothole",
    }

    print("مثال 1: نفس النوع، قريبين جغرافيًا وزمنيًا -> المفروض يحسب تشابه بصري فعليًا")
    print(f"  المدخلات: report_a={report_1a}")
    print(f"             report_b={report_1b}")
    result_1 = is_possible_duplicate(report_1a, report_1b, model, preprocess)
    print(f"  الناتج: {result_1}")
    print()

    # مثال 2: نفس الصورتين، لكن problem_type مختلف
    report_2a = report_1a
    report_2b = dict(report_1b, problem_type="garbage")

    print("مثال 2: نوع مختلف -> المفروض يرجع different فورًا من غير حساب تشابه بصري")
    print(f"  المدخلات: report_a={report_2a}")
    print(f"             report_b={report_2b}")
    result_2 = is_possible_duplicate(report_2a, report_2b, model, preprocess)
    print(f"  الناتج: {result_2}")
    print()

    # مثال 3: نفس الصورتين، نفس problem_type، لكن بعيدين جغرافيًا جدًا
    report_3a = report_1a
    report_3b = dict(report_1b, lat=report_1a["lat"] + 5, lng=report_1a["lng"] + 5)

    print("مثال 3: بعيدين جغرافيًا -> المفروض يرجع different بسبب المسافة من غير حساب تشابه بصري")
    print(f"  المدخلات: report_a={report_3a}")
    print(f"             report_b={report_3b}")
    result_3 = is_possible_duplicate(report_3a, report_3b, model, preprocess)
    print(f"  الناتج: {result_3}")
