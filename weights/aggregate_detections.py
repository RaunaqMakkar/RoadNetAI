import argparse
import json
import re
from pathlib import Path


def parse_args():
    # Command-line settings for input/output files and merge sensitivity.
    parser = argparse.ArgumentParser(
        description="Aggregate repeated frame detections into consolidated road issues."
    )
    parser.add_argument(
        "--input-json",
        default="detections.json",
        help="Path to raw detections JSON from video_processor.py",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to aggregated issues JSON (default: aggregated_<input_stem>.json)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for merging detections into the same issue",
    )
    return parser.parse_args()


def compute_iou(box_a, box_b):
    # Standard IoU between two [x1, y1, x2, y2] boxes.
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def class_prefix(label):
    # Build short ID prefix (e.g., pothole -> POT).
    letters = re.sub(r"[^A-Za-z]", "", label).upper()
    if not letters:
        return "ISS"
    return (letters[:3]).ljust(3, "X")


def find_best_match(issues, detection, iou_threshold):
    # Find an existing issue with same class and strongest IoU above threshold.
    det_class = detection.get("class", "")
    det_bbox = detection.get("bbox_xyxy", [0, 0, 0, 0])
    best_index = None
    best_iou = 0.0

    for idx, issue in enumerate(issues):
        if issue["type"] != det_class:
            continue
        iou = compute_iou(det_bbox, issue["last_bbox_xyxy"])
        if iou >= iou_threshold and iou > best_iou:
            best_iou = iou
            best_index = idx

    return best_index


def create_issue(detection, issue_number):
    # Initialize a new consolidated issue from first sighting.
    area = detection.get("mask_area_pixels")
    area = float(area) if area is not None else 0.0
    cls = detection.get("class", "unknown")
    return {
        "issue_id": f"{class_prefix(cls)}_{issue_number:03d}",
        "type": cls,
        "class_id": detection.get("class_id"),
        "first_seen": float(detection.get("timestamp_seconds", 0.0)),
        "last_seen": float(detection.get("timestamp_seconds", 0.0)),
        "frames_detected": 1,
        "avg_area_pixels": area,
        "max_area_pixels": area,
        "last_bbox_xyxy": detection.get("bbox_xyxy", [0, 0, 0, 0]),
        "frame_numbers": [int(detection.get("frame_number", 0))],
        "confidences": [float(detection.get("confidence", 0.0))],
    }


def update_issue(issue, detection):
    # Merge a new detection into an existing issue cluster.
    ts = float(detection.get("timestamp_seconds", 0.0))
    area_value = detection.get("mask_area_pixels")
    area = float(area_value) if area_value is not None else 0.0
    confidence = float(detection.get("confidence", 0.0))
    frame_number = int(detection.get("frame_number", 0))

    prev_count = len(issue["frame_numbers"])

    issue["last_seen"] = ts
    issue["last_bbox_xyxy"] = detection.get("bbox_xyxy", issue["last_bbox_xyxy"])

    issue["frame_numbers"].append(frame_number)
    issue["confidences"].append(confidence)

    # Running mean/max area, and unique frame count for this issue.
    new_count = prev_count + 1
    issue["avg_area_pixels"] = ((issue["avg_area_pixels"] * prev_count) + area) / new_count
    issue["max_area_pixels"] = max(issue["max_area_pixels"], area)
    issue["frames_detected"] = len(set(issue["frame_numbers"]))


def finalize_issues(issues):
    # Drop internal tracking fields and keep dashboard-friendly summary fields.
    finalized = []
    for issue in issues:
        confidences = issue["confidences"]
        finalized.append(
            {
                "issue_id": issue["issue_id"],
                "type": issue["type"],
                "class_id": issue["class_id"],
                "first_seen": round(issue["first_seen"], 3),
                "last_seen": round(issue["last_seen"], 3),
                "frames_detected": issue["frames_detected"],
                "avg_area_pixels": round(issue["avg_area_pixels"], 2),
                "max_area_pixels": int(round(issue["max_area_pixels"])),
                "avg_confidence": round(sum(confidences) / len(confidences), 4)
                if confidences
                else 0.0,
                "max_confidence": round(max(confidences), 4) if confidences else 0.0,
            }
        )
    return finalized


def aggregate_detections(detections, iou_threshold):
    # Process detections in chronological order to merge repeated sightings.
    ordered = sorted(
        detections,
        key=lambda d: (
            float(d.get("timestamp_seconds", 0.0)),
            int(d.get("frame_number", 0)),
        ),
    )

    issues = []
    for detection in ordered:
        best_match_index = find_best_match(issues, detection, iou_threshold)
        if best_match_index is None:
            # No matching issue found -> open a new issue cluster.
            issues.append(create_issue(detection, issue_number=len(issues) + 1))
        else:
            # Matching issue found -> merge detection into that issue.
            update_issue(issues[best_match_index], detection)

    return finalize_issues(issues)


def resolve_output_path(input_json, output_json):
    # Use explicit output path if provided, else auto-name from input file.
    if output_json:
        return Path(output_json)
    input_path = Path(input_json)
    return input_path.with_name(f"aggregated_{input_path.stem}.json")


def main():
    args = parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    # Load raw detection payload emitted by video_processor.py.
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    detections = payload.get("detections", [])
    issues = aggregate_detections(detections, iou_threshold=args.iou_threshold)

    # Final output: metadata + consolidated issues list.
    output_payload = {
        "source_file": str(input_path),
        "video": payload.get("video"),
        "model": payload.get("model"),
        "iou_threshold": args.iou_threshold,
        "total_raw_detections": len(detections),
        "total_issues": len(issues),
        "issues": issues,
    }

    output_path = resolve_output_path(args.input_json, args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Raw detections: {len(detections)}")
    print(f"Aggregated issues: {len(issues)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


