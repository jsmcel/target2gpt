#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import requests

from target2_ingest import DEFAULT_MAX_PAGES, INDEX_URL, PROCESSED, ROOT, USER_AGENT, discover_professional_use_docs, utc_now


STATE_PATH = ROOT / "output" / "target2_doc_watch_state.json"
REPORT_DIR = ROOT / "output" / "target2_doc_watch"
LATEST_JSON = REPORT_DIR / "latest.json"
LATEST_MD = REPORT_DIR / "latest.md"
GENERATED_DOC_PREFIXES = ("cr-", "acr-", "ms-")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def compact_doc(doc) -> dict[str, Any]:
    return {
        "id": doc.id,
        "title": doc.title,
        "url": doc.url,
        "category": doc.category,
        "family": doc.family,
        "release": doc.release,
        "revision_status": doc.revision_status,
        "contexts": doc.contexts,
    }


def signature(doc: dict[str, Any]) -> str:
    payload = {
        key: doc.get(key)
        for key in ["title", "url", "category", "family", "release", "revision_status", "contexts"]
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_processed_docs() -> list[dict[str, Any]]:
    path = PROCESSED / "documents.json"
    data = load_json(path, [])
    if not isinstance(data, list):
        return []
    return [
        doc
        for doc in data
        if not str(doc.get("id", "")).startswith(GENERATED_DOC_PREFIXES)
        and doc.get("id") not in {"cr-index", "acr-index"}
        and str(doc.get("source_host", "")).lower() != "local"
    ]


def fetch_current_docs() -> tuple[str, list[dict[str, Any]], list[str]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Cache-Control": "no-cache",
        }
    )
    response = session.get(INDEX_URL, timeout=60)
    response.raise_for_status()
    index_hash = sha256_text(response.text)
    docs, _, crawled_pages = discover_professional_use_docs(
        session,
        force=True,
        include_media=False,
        max_pages=DEFAULT_MAX_PAGES,
    )
    return index_hash, [compact_doc(doc) for doc in docs], crawled_pages


def head_metadata(urls: list[str], limit: int | None = None) -> dict[str, dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Cache-Control": "no-cache"})
    result: dict[str, dict[str, Any]] = {}
    for idx, url in enumerate(urls):
        if limit is not None and idx >= limit:
            break
        try:
            response = session.head(url, timeout=25, allow_redirects=True)
            if response.status_code >= 400:
                response = session.get(url, timeout=25, stream=True, allow_redirects=True)
                response.close()
            result[url] = {
                "status_code": response.status_code,
                "etag": response.headers.get("etag", ""),
                "last_modified": response.headers.get("last-modified", ""),
                "content_length": response.headers.get("content-length", ""),
                "content_type": response.headers.get("content-type", ""),
            }
        except Exception as exc:
            result[url] = {"error": repr(exc)}
    return result


def is_volatile_ecb_head_change(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Ignore ECB CDN ETag churn when no stable document metadata changed."""
    before_etag = str(before.get("etag", ""))
    after_etag = str(after.get("etag", ""))
    if not (before_etag.startswith('"myra-') and after_etag.startswith('"myra-')):
        return False
    stable_fields = ["status_code", "last_modified", "content_length", "content_type"]
    if any(before.get(field, "") != after.get(field, "") for field in stable_fields):
        return False
    return not before.get("last_modified") and not before.get("content_length")


def compare(current_docs: list[dict[str, Any]], processed_docs: list[dict[str, Any]], previous_state: dict[str, Any]) -> dict[str, Any]:
    current_by_url = {doc["url"]: doc for doc in current_docs}
    processed_by_url = {doc.get("url"): doc for doc in processed_docs if doc.get("url")}
    previous_by_url = previous_state.get("head", {}) if isinstance(previous_state.get("head"), dict) else {}

    added = [doc for url, doc in current_by_url.items() if url not in processed_by_url]
    removed = [doc for url, doc in processed_by_url.items() if url not in current_by_url]
    changed_listing = []
    for url, doc in current_by_url.items():
        old = processed_by_url.get(url)
        if old and signature(doc) != signature(old):
            changed_listing.append({"before": old, "after": doc})

    return {
        "added": added,
        "removed": removed,
        "changed_listing": changed_listing,
        "previous_head": previous_by_url,
    }


def run_step(args: list[str]) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60 * 60,
    )
    return {
        "returncode": proc.returncode,
        "command": " ".join(args),
        "duration_seconds": round(time.time() - started, 1),
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-6000:],
    }


def run_rebuild_pipeline() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = REPORT_DIR / f"processed-backup-{stamp}"
    if PROCESSED.exists():
        shutil.copytree(PROCESSED, backup_dir)

    steps = [
        [sys.executable, "target2_ingest.py", "--refresh-index"],
    ]

    results = []
    for step in steps:
        result = run_step(step)
        results.append(result)
        if result["returncode"] != 0:
            if backup_dir.exists():
                if PROCESSED.exists():
                    shutil.rmtree(PROCESSED)
                shutil.copytree(backup_dir, PROCESSED)
                result["processed_restored_from"] = str(backup_dir)
            return {"returncode": result["returncode"], "steps": results, "backup": str(backup_dir)}
    return {"returncode": 0, "steps": results, "backup": str(backup_dir)}


def write_report(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (REPORT_DIR / f"{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# target2 Documentation Watch",
        "",
        f"- Checked at: {report['checked_at']}",
        f"- Current documents: {report['current_documents']}",
        f"- Processed documents: {report['processed_documents']}",
        f"- Change detected: {report['change_detected']}",
        f"- Rebuild triggered: {report['rebuild_triggered']}",
        f"- Rebuild status: {report.get('rebuild_status', 'not_run')}",
        f"- Ignored volatile HEAD changes: {len(report['changes'].get('ignored_head', []))}",
        "",
        "## Added",
    ]
    for doc in report["changes"]["added"][:50]:
        lines.append(f"- {doc.get('title')} - {doc.get('url')}")
    if not report["changes"]["added"]:
        lines.append("- None")
    lines.extend(["", "## Removed"])
    for doc in report["changes"]["removed"][:50]:
        lines.append(f"- {doc.get('title')} - {doc.get('url')}")
    if not report["changes"]["removed"]:
        lines.append("- None")
    lines.extend(["", "## Listing Changes"])
    for item in report["changes"]["changed_listing"][:50]:
        after = item.get("after", {})
        lines.append(f"- {after.get('title')} - {after.get('url')}")
    if not report["changes"]["changed_listing"]:
        lines.append("- None")
    LATEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch ECB target2 professional-use documentation and rebuild the RAG index when it changes.")
    parser.add_argument("--check-only", action="store_true", help="detect changes but do not run target2_ingest.py")
    parser.add_argument("--force-rebuild", action="store_true", help="rebuild even if no change is detected")
    parser.add_argument("--skip-head", action="store_true", help="skip HEAD metadata probing")
    parser.add_argument("--verbose", action="store_true", help="print the full JSON report")
    args = parser.parse_args(argv)

    previous_state = load_json(STATE_PATH, {})
    processed_docs = load_processed_docs()
    index_hash, current_docs, crawled_pages = fetch_current_docs()
    diff = compare(current_docs, processed_docs, previous_state)

    urls = [doc["url"] for doc in current_docs]
    current_head = {} if args.skip_head else head_metadata(urls)
    previous_head = diff.pop("previous_head", {})
    changed_head = []
    ignored_head = []
    if previous_head and current_head:
        for url, meta in current_head.items():
            before = previous_head.get(url)
            if before and meta != before:
                item = {"url": url, "before": before, "after": meta}
                if is_volatile_ecb_head_change(before, meta):
                    ignored_head.append(item)
                else:
                    changed_head.append(item)

    change_detected = bool(diff["added"] or diff["changed_listing"] or changed_head)
    rebuild_triggered = bool((change_detected or args.force_rebuild) and not args.check_only)
    ingest_result: dict[str, Any] | None = None
    rebuild_status = "not_run"
    if rebuild_triggered:
        ingest_result = run_rebuild_pipeline()
        rebuild_status = "ok" if ingest_result["returncode"] == 0 else "failed"

    state = {
        "checked_at": utc_now(),
        "index_url": INDEX_URL,
        "index_hash": index_hash,
        "crawled_pages": crawled_pages,
        "documents": current_docs,
        "head": current_head,
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "checked_at": state["checked_at"],
        "index_url": INDEX_URL,
        "crawled_pages": crawled_pages,
        "current_documents": len(current_docs),
        "processed_documents": len(processed_docs),
        "change_detected": change_detected,
        "rebuild_triggered": rebuild_triggered,
        "rebuild_status": rebuild_status,
        "changes": {**diff, "changed_head": changed_head, "ignored_head": ignored_head},
        "ingest": ingest_result,
    }
    write_report(report)
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "checked_at": report["checked_at"],
                    "current_documents": report["current_documents"],
                    "processed_documents": report["processed_documents"],
                    "added": len(report["changes"]["added"]),
                    "removed_not_triggering": len(report["changes"]["removed"]),
                    "changed_listing": len(report["changes"]["changed_listing"]),
                    "changed_head": len(report["changes"]["changed_head"]),
                    "ignored_head": len(report["changes"]["ignored_head"]),
                    "change_detected": report["change_detected"],
                    "rebuild_triggered": report["rebuild_triggered"],
                    "rebuild_status": report["rebuild_status"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 1 if rebuild_status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())


