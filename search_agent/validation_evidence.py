"""Reassemble a bounded Cloud Logging JSON export; fail on missing or changed chunks."""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path


def reconstruct(entries: list[dict]) -> dict[str, bytes]:
    groups: dict[str, dict] = {}
    for entry in entries:
        payload = entry.get("jsonPayload", entry)
        if "textPayload" in entry:
            try:
                payload = json.loads(entry["textPayload"])
            except (ValueError, TypeError):
                continue
        if not isinstance(payload, dict) or payload.get("validation_evidence") not in (
            "chunk",
            "complete",
        ):
            continue
        name = payload["artifact"]
        if not re.fullmatch(r"validation/[a-f0-9]{32}/(baseline|index|search|final)\.json", name):
            raise ValueError("Unexpected artifact name")
        if not 1 <= payload["parts"] <= 82:
            raise ValueError("Invalid part count")
        group = groups.setdefault(
            name,
            {
                "sha256": payload["sha256"],
                "parts": payload["parts"],
                "chunks": {},
                "complete": False,
            },
        )
        if (group["sha256"], group["parts"]) != (payload["sha256"], payload["parts"]):
            raise ValueError("Conflicting artifact metadata")
        if payload["validation_evidence"] == "complete":
            group["complete"] = True
        else:
            part = payload["part"]
            if (
                not isinstance(part, int)
                or not 0 <= part < group["parts"]
                or len(payload["base64"]) > 16384
            ):
                raise ValueError("Invalid chunk")
            raw = base64.b64decode(payload["base64"], validate=True)
            if part in group["chunks"] and group["chunks"][part] != raw:
                raise ValueError("Conflicting duplicate chunk")
            group["chunks"][part] = raw
    result = {}
    for name, group in groups.items():
        if not group["complete"] or set(group["chunks"]) != set(range(group["parts"])):
            raise ValueError("Incomplete evidence; retain logs and retry read-only export")
        raw = b"".join(group["chunks"][i] for i in range(group["parts"]))
        if len(raw) > 1_000_000 or hashlib.sha256(raw).hexdigest() != group["sha256"]:
            raise ValueError("Evidence checksum mismatch")
        json.loads(raw)
        result[name] = raw
    if not result:
        raise ValueError("No evidence found")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument(
        "output", type=Path, help="New directory; existing files are never replaced"
    )
    args = parser.parse_args()
    if args.logs.stat().st_size > 10_000_000:
        raise ValueError("Log export exceeds 10 MB")
    artifacts = reconstruct(json.loads(args.logs.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in artifacts.items():
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    print(
        json.dumps(
            {
                "artifacts": list(artifacts),
                "final_present": any(n.endswith("/final.json") for n in artifacts),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
