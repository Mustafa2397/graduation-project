import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duplicate.similarity import get_image_embedding, load_clip_model


def verify_resolution(before_image_path, after_image_path, model, preprocess):
    embedding_before = get_image_embedding(before_image_path, model, preprocess)
    embedding_after = get_image_embedding(after_image_path, model, preprocess)

    similarity = float(
        np.dot(embedding_before, embedding_after)
        / (np.linalg.norm(embedding_before) * np.linalg.norm(embedding_after))
    )

    change_score = 1 - similarity

    if change_score >= 0.35:
        status = "resolved_confirmed"
    elif change_score >= 0.15:
        status = "needs_manual_review"
    else:
        status = "not_resolved"

    return {"status": status, "change_score": change_score, "similarity": similarity}


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "..", "..", "phase 2 _ data set", "dataset", "dataset")

    pothole_images_dir = os.path.join(dataset_dir, "Pothole detection2", "train", "images")
    puddle_images_dir = os.path.join(dataset_dir, "puddle detection3", "train", "images")

    pothole_images = sorted(os.listdir(pothole_images_dir))
    puddle_images = sorted(os.listdir(puddle_images_dir))

    pothole_image_1 = os.path.join(pothole_images_dir, pothole_images[0])
    pothole_image_2 = os.path.join(pothole_images_dir, pothole_images[1])
    puddle_image_1 = os.path.join(puddle_images_dir, puddle_images[0])

    print("تحميل موديل CLIP...")
    model, preprocess = load_clip_model()
    print("تم تحميل الموديل.")
    print()

    # مثال 1: محلول بوضوح -- قبل: حفرة، بعد: صورة مختلفة تمامًا (بركة مياه)
    print("مثال 1: محلول بوضوح (قبل: حفرة، بعد: صورة مختلفة تمامًا من نوع تاني)")
    print(f"  قبل: {os.path.basename(pothole_image_1)}")
    print(f"  بعد: {os.path.basename(puddle_image_1)}")
    result_1 = verify_resolution(pothole_image_1, puddle_image_1, model, preprocess)
    print(f"  الناتج: {result_1}")
    print()

    # مثال 2: لسه نفس المشكلة -- نفس الصورة مرتين
    print("مثال 2: لسه نفس المشكلة (نفس الصورة قبل وبعد)")
    print(f"  قبل: {os.path.basename(pothole_image_1)}")
    print(f"  بعد: {os.path.basename(pothole_image_1)}")
    result_2 = verify_resolution(pothole_image_1, pothole_image_1, model, preprocess)
    print(f"  الناتج: {result_2}")
    print()

    # مثال 3: منطقة رمادية -- صورتين حفرة مختلفتين لكن من نفس النوع
    print("مثال 3: منطقة رمادية (صورتين حفرة مختلفتين لكن من نفس النوع)")
    print(f"  قبل: {os.path.basename(pothole_image_1)}")
    print(f"  بعد: {os.path.basename(pothole_image_2)}")
    result_3 = verify_resolution(pothole_image_1, pothole_image_2, model, preprocess)
    print(f"  الناتج: {result_3}")
