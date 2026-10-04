"""Download the public PhysioNet/CinC 2019 training sets.

The training release is one pipe-delimited file per patient under training_setA
and training_setB. This module uses the PhysioNet directory indexes so only
files that are actually published are requested. Each worker reuses one HTTPS
connection, because a fresh handshake per small file is slower than the transfer.

Documented cohort sizes are 20,336 (set A) and 20,000 (set B). The index is
treated as the file list; the documented counts are checked, not assumed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import threading
import urllib.request
from pathlib import Path
from queue import Queue

from src.data.schema import DOCUMENTED_SET_A_COUNT, DOCUMENTED_SET_B_COUNT, TRAINING_FILES_URL

USER_AGENT = "sepsis-early-prediction-research-phase1/0.1"
FILE_HREF = re.compile(r'href="(p\d+\.psv)"')
HOST = "physionet.org"
SETS = {
    "A": "training_setA",
    "B": "training_setB",
}
DOCUMENTED_COUNTS = {"A": DOCUMENTED_SET_A_COUNT, "B": DOCUMENTED_SET_B_COUNT}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_dir() -> Path:
    return project_root() / "data" / "raw"


def index_html(set_code: str, refresh: bool = False) -> str:
    folder = SETS[set_code]
    cache = raw_dir() / "_indexes" / f"set{set_code}.html"
    if cache.exists() and not refresh and cache.stat().st_size > 0:
        return cache.read_text(encoding="utf-8", errors="replace")
    url = f"{TRAINING_FILES_URL}{folder}/"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        text = response.read().decode("utf-8", errors="replace")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def list_patient_files(set_code: str, refresh_index: bool = False) -> list[str]:
    names = FILE_HREF.findall(index_html(set_code, refresh=refresh_index))
    # The autoindex repeats the name in the link text; href capture is one per file.
    unique = list(dict.fromkeys(names))
    return unique


def _fetch(connection: http.client.HTTPSConnection, path: str) -> tuple[int, bytes]:
    connection.request("GET", path, headers={"User-Agent": USER_AGENT, "Connection": "keep-alive"})
    response = connection.getresponse()
    body = response.read()
    return response.status, body


def _worker(work: Queue, root: Path, counts: dict, lock: threading.Lock, failures: list[str]) -> None:
    connection = http.client.HTTPSConnection(HOST, timeout=60)
    while True:
        item = work.get()
        try:
            if item is None:
                connection.close()
                return
            folder, name = item
            destination = root / folder / name
            if destination.exists() and destination.stat().st_size > 0:
                with lock:
                    counts["skipped"] += 1
                continue
            url_path = f"/files/challenge-2019/1.0.0/training/{folder}/{name}"
            status = 0
            body = b""
            for _attempt in range(3):
                try:
                    status, body = _fetch(connection, url_path)
                    break
                except Exception:
                    connection.close()
                    connection = http.client.HTTPSConnection(HOST, timeout=60)
                    status = 0
            if status == 200 and body:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".part")
                temporary.write_bytes(body)
                temporary.replace(destination)
                with lock:
                    counts["downloaded"] += 1
                    done = sum(counts.values())
            elif status == 404:
                with lock:
                    counts["missing"] += 1
                    done = sum(counts.values())
            else:
                with lock:
                    counts["failed"] += 1
                    failures.append(f"{folder}/{name} status={status}")
                    done = sum(counts.values())
            if done % 1000 == 0:
                print(
                    f"  {done}/{work.qsize() + done} downloaded={counts['downloaded']} "
                    f"skipped={counts['skipped']} missing={counts['missing']} failed={counts['failed']}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            with lock:
                counts["failed"] += 1
                failures.append(str(exc))
        finally:
            work.task_done()


def download_training(workers: int = 12, refresh_index: bool = False) -> dict:
    root = raw_dir()
    summary: dict = {"sets": {}, "failures": []}
    jobs: list[tuple[str, str]] = []
    listed: dict[str, int] = {}
    for set_code, folder in SETS.items():
        names = list_patient_files(set_code, refresh_index=refresh_index)
        listed[set_code] = len(names)
        (root / folder).mkdir(parents=True, exist_ok=True)
        print(f"Set {set_code}: {len(names)} files listed (documented {DOCUMENTED_COUNTS[set_code]})", flush=True)
        jobs.extend((folder, name) for name in names)

    counts = {"downloaded": 0, "skipped": 0, "missing": 0, "failed": 0}
    failures: list[str] = []
    lock = threading.Lock()
    work: Queue = Queue()
    for job in jobs:
        work.put(job)
    threads = [
        threading.Thread(target=_worker, args=(work, root, counts, lock, failures), daemon=True)
        for _ in range(workers)
    ]
    for thread in threads:
        thread.start()
    finished = 0
    total = len(jobs)
    while finished < total:
        work.join()
        finished = total
        with lock:
            snapshot = dict(counts)
        print(
            f"  {finished}/{total} downloaded={snapshot['downloaded']} skipped={snapshot['skipped']} "
            f"missing={snapshot['missing']} failed={snapshot['failed']}",
            flush=True,
        )
    for _ in threads:
        work.put(None)
    for thread in threads:
        thread.join()

    for set_code, folder in SETS.items():
        on_disk = list((root / folder).glob("p*.psv"))
        summary["sets"][set_code] = {
            "listed": listed[set_code],
            "on_disk": len(on_disk),
            "documented": DOCUMENTED_COUNTS[set_code],
            "matches_documented_count": len(on_disk) == DOCUMENTED_COUNTS[set_code],
        }
    summary["transfer"] = counts
    summary["failures"] = failures[:50]
    manifest_path = root / "download_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["sets"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PhysioNet 2019 training sets A and B.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--refresh-index", action="store_true")
    args = parser.parse_args()
    summary = download_training(workers=args.workers, refresh_index=args.refresh_index)
    failed = summary["transfer"]["failed"]
    if failed:
        raise SystemExit(f"{failed} downloads failed")
    for set_code, counts in summary["sets"].items():
        if counts["on_disk"] != counts["documented"]:
            raise SystemExit(
                f"Set {set_code} has {counts['on_disk']} files on disk; "
                f"the challenge documentation states {counts['documented']}."
            )


if __name__ == "__main__":
    main()
