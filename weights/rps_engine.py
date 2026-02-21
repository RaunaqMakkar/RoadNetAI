import argparse
import json
import math
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
    "Moderate": "Schedule maintenance",
    "Low": "Monitor condition",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute stabilized Repair Priority Score (RPS) for aggregated road issues."
    )
    parser.add_argument(
        "--input-json",
        default="aggregated_detections_manhole2_video_manhole2_video_20260220_212216.json",
        help="Input aggregated issues JSON path",
    )
    parser.add_argument(
        "--output-json",
        default="issues_with_stable_rps.json",
        help="Output JSON path with stabilized RPS scores",
    )
    return parser.parse_args()


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


def clamp(value, low, high):
    return min(max(value, low), high)


def load_issues(input_path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("issues", [])
    if isinstance(data, list):
        return data
    return []


def get_max_area(issues):
    max_area = 0.0
    for issue in issues:
        area = max(0.0, safe_float(issue.get("max_area_pixels", 0.0)))
        if area > max_area:
            max_area = area
    return max_area


def area_factor_log(max_area_pixels, max_area_in_video):
    area = max(0.0, safe_float(max_area_pixels))
    max_area = max(0.0, safe_float(max_area_in_video))

    # A = log(1 + area) / log(1 + max_area), with safe denominator handling.
    denom = math.log1p(max_area)
    if denom <= 0:
        return 0.0

    return clamp(math.log1p(area) / denom, 0.0, 1.0)


def class_weight(issue_type):
    return clamp(CLASS_WEIGHTS.get(str(issue_type), DEFAULT_CLASS_WEIGHT), 0.0, 1.0)


def duration_seconds(first_seen, last_seen):
    return max(0.0, safe_float(last_seen) - safe_float(first_seen))


def time_factor(duration):
    # T = 1 - exp(-duration / 3)
    return clamp(1.0 - math.exp(-max(0.0, duration) / 3.0), 0.0, 1.0)


def frequency_factor(frames_detected):
    # F = 1 - exp(-frames_detected / 5)
    frames = max(0.0, float(max(0, safe_int(frames_detected))))
    return clamp(1.0 - math.exp(-frames / 5.0), 0.0, 1.0)


def severity(area_component, cls_weight):
    # S = 0.7 * A + 0.3 * class_weight
    return clamp(0.7 * area_component + 0.3 * cls_weight, 0.0, 1.0)


def raw_score(severity_value, t_factor, f_factor, cls_weight):
    # R_raw = 100 * (0.5*S + 0.2*T + 0.1*F + 0.2*class_weight)
    score = 100.0 * (
        0.5 * severity_value
        + 0.2 * t_factor
        + 0.1 * f_factor
        + 0.2 * cls_weight
    )
    return max(0.0, score)


def confidence_factor(avg_confidence):
    conf = clamp(safe_float(avg_confidence), 0.0, 1.0)
    return 0.5 + 0.5 * conf


def final_score(raw, conf_factor):
    # R_final = min(max(R_raw * confidence_factor, 0), 100)
    return round(clamp(raw * conf_factor, 0.0, 100.0), 2)


def classify_priority(score):
    if score >= 85.0:
        return "Critical"
    if score >= 65.0:
        return "High"
    if score >= 40.0:
        return "Moderate"
    return "Low"


def score_issue(issue, max_area_in_video):
    first_seen = safe_float(issue.get("first_seen", 0.0))
    last_seen = safe_float(issue.get("last_seen", 0.0))
    frames = max(0, safe_int(issue.get("frames_detected", 0)))
    max_area_pixels = max(0.0, safe_float(issue.get("max_area_pixels", 0.0)))
    avg_conf = clamp(safe_float(issue.get("avg_confidence", 0.0)), 0.0, 1.0)

    a = area_factor_log(max_area_pixels, max_area_in_video)
    w = class_weight(issue.get("type", ""))
    d = duration_seconds(first_seen, last_seen)
    t = time_factor(d)
    f = frequency_factor(frames)
    s = severity(a, w)

    r_raw = raw_score(s, t, f, w)
    c_factor = confidence_factor(avg_conf)
    r_final = final_score(r_raw, c_factor)
    priority = classify_priority(r_final)

    result = dict(issue)
    result["first_seen"] = first_seen
    result["last_seen"] = last_seen
    result["frames_detected"] = frames
    result["max_area_pixels"] = int(round(max_area_pixels))
    result["avg_confidence"] = avg_conf
    result["rps_score"] = r_final
    result["priority"] = priority
    result["recommended_action"] = PRIORITY_ACTIONS[priority]
    return result


def build_output(issues):
    return {
        "total_issues": len(issues),
        "issues": issues,
    }


def run_engine(input_path, output_path):
    issues = load_issues(input_path)

    if not issues:
        payload = build_output([])
    else:
        max_area = get_max_area(issues)
        scored = [score_issue(issue, max_area) for issue in issues]
        payload = build_output(scored)

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

    payload = run_engine(input_path=input_path, output_path=output_path)
    print(f"Total issues processed: {payload['total_issues']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()