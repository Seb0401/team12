"""YOLOv8 fine-tuning script for shovel component detection."""

import sys
from pathlib import Path

from ultralytics import YOLO


def train(
    dataset_yaml: str = "dataset/labeling/data.yaml",
    base_model: str = "yolo26n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
) -> None:
    """
    Fine-tune YOLOv8 on labeled shovel frames.

    @param dataset_yaml - Path to dataset config
    @param base_model - Pretrained YOLO weights to start from
    @param epochs - Training epochs
    @param imgsz - Input image size
    @param batch - Batch size
    """
    model = YOLO(base_model)

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/train",
        name="shovel",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
    )

    best_pt = Path("runs/train/shovel/weights/best.pt")
    target = Path("solution/detection/model/best.pt")
    target.parent.mkdir(parents=True, exist_ok=True)

    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, target)
        print(f"Best weights copied to {target}")
    else:
        print(f"WARNING: {best_pt} not found. Check training output.")


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    train(epochs=epochs)
