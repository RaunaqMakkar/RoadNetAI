import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    # CLI options for model/video paths and sampling + inference settings.
    parser = argparse.ArgumentParser(
        description="Sample video frames, run YOLO segmentation, and export detections to JSON."
    )
    parser.add_argument("--model", default="best.pt", help="Path to YOLO model (.pt)")
    parser.add_argument("--video", default="manhole2_video.mp4", help="Input video path")
    parser.add_argument(
        "--output-json",
        default="detections_manhole2_video.json",
        help="Base output JSON file name/path",
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=1.0,
        help="Sample one frame every N seconds (must be > 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device, e.g. "cpu" or "cuda"',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output-json instead of creating a unique file per run",
    )
    return parser.parse_args()


def resolve_output_path(output_json, video_path, overwrite=False):
    base_path = Path(output_json)
    suffix = base_path.suffix if base_path.suffix else ".json"

    if overwrite:
        return base_path if base_path.suffix else base_path.with_suffix(".json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(video_path).stem
    candidate = base_path.with_name(
        f"{base_path.stem}_{video_stem}_{timestamp}{suffix}"
    )

    # Avoid collisions if two runs start within the same second.
    counter = 1
    while candidate.exists():
        candidate = base_path.with_name(
            f"{base_path.stem}_{video_stem}_{timestamp}_{counter}{suffix}"
        )
        counter += 1

    return candidate


def mask_area_from_result(masks_data, index):
    # Convert mask tensor to numpy (if needed) and count foreground pixels.
    if masks_data is None:
        return None

    if hasattr(masks_data, "cpu"):
        masks_data = masks_data.cpu()
    if hasattr(masks_data, "numpy"):
        masks_data = masks_data.numpy()

    if index >= len(masks_data):
        return None

    return int((masks_data[index] > 0.5).sum())


def detection_to_dict(result, index, timestamp_seconds, frame_number):
    # Extract one detection into a JSON-serializable structure.
    box = result.boxes[index]

    class_id = int(box.cls.item())
    class_name = result.names.get(class_id, str(class_id))
    confidence = float(box.conf.item())
    bbox_xyxy = [float(v) for v in box.xyxy[0].tolist()]

    masks_data = result.masks.data if result.masks is not None else None
    mask_area_pixels = mask_area_from_result(masks_data, index)

    return {
        "class": class_name,
        "class_id": class_id,
        "confidence": confidence,
        "bbox_xyxy": bbox_xyxy,
        "mask_area_pixels": mask_area_pixels,
        "timestamp_seconds": round(float(timestamp_seconds), 3),
        "frame_number": int(frame_number),
    }


def build_payload(args, fps, frame_count, detections):
    # Top-level JSON object with run metadata + all detections.
    return {
        "video": str(Path(args.video)),
        "model": str(Path(args.model)),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "sample_every_seconds": float(args.sample_every),
        "confidence_threshold": float(args.conf),
        "device": args.device,
        "total_detections": len(detections),
        "detections": detections,
    }


def process_video(args):
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be greater than 0")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise IOError(f"Error opening video file: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Unable to determine FPS from input video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Number of frames to skip between sampled frames.
    sample_stride = max(1, int(round(args.sample_every * fps)))

    detections = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference only on sampled frames.
        if frame_number % sample_stride == 0:
            timestamp_seconds = frame_number / fps
            results = model(frame, conf=args.conf, device=args.device, verbose=False)
            result = results[0]

            # Store each instance prediction from this sampled frame.
            if result.boxes is not None and len(result.boxes) > 0:
                for det_idx in range(len(result.boxes)):
                    detections.append(
                        detection_to_dict(
                            result=result,
                            index=det_idx,
                            timestamp_seconds=timestamp_seconds,
                            frame_number=frame_number,
                        )
                    )

        frame_number += 1

    cap.release()

    payload = build_payload(args, fps=fps, frame_count=frame_count, detections=detections)

    output_path = resolve_output_path(
        output_json=args.output_json,
        video_path=args.video,
        overwrite=args.overwrite,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist readable JSON for downstream analytics pipelines.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path, len(detections)


def main():
    args = parse_args()
    output_path, det_count = process_video(args)
    print(f"Detections written: {det_count}")
    print(f"JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
