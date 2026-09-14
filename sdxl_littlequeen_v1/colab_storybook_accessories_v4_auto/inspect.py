from pathlib import Path
import json
import subprocess


root = Path("/content/lq_storybook_accessories_v4_auto")
summary_path = root / "summary.json"
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completed_finals = [
        record["id"]
        for record in summary.get("records", [])
        if "final" in record.get("artifacts", {})
        and (root / record["artifacts"]["final"]).is_file()
    ]
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "record_count": len(summary.get("records", [])),
                "preflight_accepted_count": summary.get("preflight_accepted_count"),
                "accepted_count": summary.get("accepted_count"),
                "selected_count": summary.get("selected_count"),
                "completed_finals": completed_finals,
                "archive_bytes": (
                    Path("/content/lq_storybook_accessories_v4_auto_results.tar.gz").stat().st_size
                    if Path(
                        "/content/lq_storybook_accessories_v4_auto_results.tar.gz"
                    ).is_file()
                    else None
                ),
            },
            indent=2,
        )
    )
else:
    print("summary.json is missing")
print(
    subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
)
print(
    subprocess.run(
        ["pgrep", "-af", "remote_storybook|python"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
)
