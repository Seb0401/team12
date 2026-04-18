"""Extract evenly-spaced frames from video for YOLO labeling."""

import sys
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 100,
) -> None:
    """
    Extract N evenly-spaced frames from a video for manual labeling.

    @param video_path - Path to input video
    @param output_dir - Directory to save extracted JPEGs
    @param num_frames - Number of frames to extract
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // num_frames)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        filename = out / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(filename), frame)
        saved += 1

    cap.release()
    print(f"Extracted {saved} frames to {out}/")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "inputs/shovel_left.mp4"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset/images"
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    extract_frames(video, out_dir, count)
