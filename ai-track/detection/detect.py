import os

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "best.pt")


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


if __name__ == "__main__":
    dataset_dir = os.path.join(SCRIPT_DIR, "..", "..", "phase 2 _ data set", "dataset", "dataset")

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

    model = load_detection_model()

    for image_path in test_images:
        print(f"صورة: {os.path.basename(image_path)}")
        detections = detect_problem(image_path, model)
        print(f"  الناتج: {detections}")
        print()
