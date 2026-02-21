import json
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_ticket(issue, ticket_number):
    now_ts = utc_timestamp_iso()

    return {
        "ticket_id": f"TKT_{ticket_number:03d}",
        "issue_id": issue.get("issue_id"),
        "type": issue.get("type"),
        "priority": issue.get("priority"),
        "rps_score": issue.get("rps_score"),
        "first_seen": issue.get("first_seen"),
        "last_seen": issue.get("last_seen"),
        "frames_detected": issue.get("frames_detected"),
        "max_area_pixels": issue.get("max_area_pixels"),
        "avg_confidence": issue.get("avg_confidence"),
        "recommended_action": issue.get("recommended_action"),
        "location": {
            "latitude": None,
            "longitude": None,
            "road_name": "Unknown",
            "zone": "Ward 1",
        },
        "status": "Open",
        "assigned_department": "Public Works Department",
        "assigned_to": None,
        "created_at": now_ts,
        "updated_at": now_ts,
        "source": "AI_CAMERA",
        "is_verified": False,
    }


def load_issues(input_path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        issues = data.get("issues", [])
    elif isinstance(data, list):
        issues = data
    else:
        issues = []

    return issues


def generate_tickets(issues):
    tickets = []
    for idx, issue in enumerate(issues, start=1):
        tickets.append(create_ticket(issue, idx))
    return tickets


def save_tickets(output_path, tickets):
    payload = {
        "total_tickets": len(tickets),
        "tickets": tickets,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    input_path = Path("issues_with_stable_rps.json")
    output_path = Path("tickets.json")

    # Handle missing input safely by exiting cleanly with a clear message.
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    issues = load_issues(input_path)
    tickets = generate_tickets(issues)
    save_tickets(output_path, tickets)

    print(f"Generated {len(tickets)} tickets")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()