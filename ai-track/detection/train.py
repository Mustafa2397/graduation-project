import argparse
import os

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML_PATH = os.path.join(SCRIPT_DIR, "..", "data.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML_PATH,
        epochs=args.epochs,
        project="runs",
        name="problem_detection_v1",
    )


if __name__ == "__main__":
    main()
