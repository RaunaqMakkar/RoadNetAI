import argparse
import json
from pathlib import Path


CLASS_WEIGHTS = {
    "open_manhole": 1.0,
    "pothole": 0.8,
    "manhole": 0.6,
    "road_crack": 0.4,
}
DEFAULT_CLASS_WEIGHT = 0.5

PRIORITY_ACTIONS = {
    "Critical": "Immediate repair within 24 hours",
    "High": "Repair within 3 days",
    "Medium": "Schedule maintenance",
    "Low": "Monitor condition",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Repair Priority Score (RPS) for aggregated road issues."
    )
    parser.add_argument(
        "--input-json",
        default="aggregated_detections_manhole_video_manhole_video_20260217_163033.json",
        help="Input aggregated issues JSON path",
    )
    parser.add_argument(
        "--output-json",
        default="issues_with_rps.json",
        help="Output JSON path with RPS scores",
    )
    return parser.parse_args()


def load_issues(input_path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either {"issues": [...]} or a direct list for compatibility.
    if isinstance(data, dict):
        issues = data.get("issues", [])
    elif isinstance(data, list):
        issues = data
    else:
        issues = []

    return issues


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_max_area(issues):
    max_area = 0.0
    for issue in issues:
        area = max(0.0, safe_float(issue.get("max_area_pixels", 0)))
        if area > max_area:
            max_area = area
    return max_area


def normalize_area(max_area_pixels, max_area_in_video):
    if max_area_in_video <= 0:
        return 0.0
    value = max(0.0, safe_float(max_area_pixels)) / max_area_in_video
    return min(max(value, 0.0), 1.0)


def get_class_weight(issue_type):
    return CLASS_WEIGHTS.get(str(issue_type), DEFAULT_CLASS_WEIGHT)


def get_time_factor(first_seen, last_seen):
    duration_seconds = max(0.0, safe_float(last_seen) - safe_float(first_seen))
    return min(duration_seconds / 5.0, 1.0), duration_seconds


def get_frequency_factor(frames_detected):
    frames = max(0, safe_int(frames_detected))
    return min(frames / 10.0, 1.0)


def compute_rps(normalized_area, class_weight, time_factor, frequency_factor):
    raw_score = 100.0 * (
        0.5 * normalized_area
        + 0.2 * class_weight
        + 0.2 * time_factor
        + 0.1 * frequency_factor
    )
    # Guardrails to keep scores in [0, 100].
    return round(min(max(raw_score, 0.0), 100.0), 2)


def classify_priority(rps_score):
    if rps_score >= 80.0:
        return "Critical"
    if rps_score >= 50.0:
        return "High"
    if rps_score >= 20.0:
        return "Medium"
    return "Low"


def build_issue_with_rps(issue, max_area_in_video):
    first_seen = safe_float(issue.get("first_seen", 0.0))
    last_seen = safe_float(issue.get("last_seen", 0.0))
    frames_detected = max(0, safe_int(issue.get("frames_detected", 0)))
    max_area_pixels = max(0.0, safe_float(issue.get("max_area_pixels", 0.0)))

    normalized_area = normalize_area(max_area_pixels, max_area_in_video)
    class_weight = get_class_weight(issue.get("type", ""))
    time_factor, _duration_seconds = get_time_factor(first_seen, last_seen)
    frequency_factor = get_frequency_factor(frames_detected)

    rps_score = compute_rps(
        normalized_area=normalized_area,
        class_weight=class_weight,
        time_factor=time_factor,
        frequency_factor=frequency_factor,
    )
    priority = classify_priority(rps_score)

    result = dict(issue)
    result["first_seen"] = first_seen
    result["last_seen"] = last_seen
    result["frames_detected"] = frames_detected
    result["max_area_pixels"] = int(round(max_area_pixels))
    result["rps_score"] = rps_score
    result["priority"] = priority
    result["recommended_action"] = PRIORITY_ACTIONS[priority]
    return result


def build_output_payload(issues_with_rps):
    return {
        "total_issues": len(issues_with_rps),
        "issues": issues_with_rps,
    }


def run_rps_engine(input_path, output_path):
    issues = load_issues(input_path)

    if not issues:
        payload = build_output_payload([])
    else:
        max_area_in_video = get_max_area(issues)
        issues_with_rps = [
            build_issue_with_rps(issue, max_area_in_video=max_area_in_video)
            for issue in issues
        ]
        payload = build_output_payload(issues_with_rps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    payload = run_rps_engine(input_path=input_path, output_path=output_path)
    print(f"Total issues processed: {payload['total_issues']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()