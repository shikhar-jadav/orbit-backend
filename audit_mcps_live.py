"""Live-streaming MCP marketplace auditor with progress bar and incremental saving.

Usage:
    python audit_mcps_live.py
    python audit_mcps_live.py --concurrency 25 --timeout 4.0 --batch-size 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import connector_registry
from verify_marketplace import REPORT_PATH, verify_single_connector


def _render_bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _build_report(
    final_results: List[Dict[str, Any]],
    elapsed_total: float,
) -> Dict[str, Any]:
    summary: Dict[str, int] = {}
    for r in final_results:
        st = str(r.get("verification_status") or "unverified")
        summary[st] = summary.get(st, 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_audited": len(final_results),
        "elapsed_seconds": round(elapsed_total, 2),
        "summary": summary,
        "stats": {
            "with_url": sum(1 for r in final_results if r.get("server_url")),
            "without_url": sum(1 for r in final_results if not r.get("server_url")),
            "reachable": sum(1 for r in final_results if r.get("reachable")),
            "handshake_ok": sum(1 for r in final_results if r.get("handshake_ok")),
            "tools_discovered": sum(1 for r in final_results if r.get("tools_discovered")),
            "total_tools": sum(int(r.get("tools_count") or 0) for r in final_results),
            "oauth_supported": sum(1 for r in final_results if r.get("oauth_supported")),
            "dynamic_registration_supported": sum(
                1 for r in final_results if r.get("dynamic_registration_supported")
            ),
        },
        "results": final_results,
    }


def _save_report(report: Dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


async def run_live_audit(
    *,
    concurrency: int = 25,
    timeout: float = 4.0,
    batch_size: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    manifests = connector_registry.load_marketplace_manifests()
    total = len(manifests)
    results: List[Optional[Dict[str, Any]]] = [None] * total

    counts = {
        "done": 0,
        "handshake_ok": 0,
        "oauth": 0,
        "dcr": 0,
        "no_url": 0,
        "failed": 0,
    }
    started = time.monotonic()

    def _record_result(idx: int, rec: Dict[str, Any]) -> None:
        if results[idx] is not None:
            return
        results[idx] = rec
        counts["done"] += 1
        if not rec.get("server_url"):
            counts["no_url"] += 1
            tag = "NO_URL"
        elif rec.get("handshake_ok"):
            counts["handshake_ok"] += 1
            tag = f"MCP_OK({rec.get('transport')}, tools={rec.get('tools_count', 0)})"
        elif rec.get("oauth_supported"):
            counts["oauth"] += 1
            if rec.get("dynamic_registration_supported"):
                counts["dcr"] += 1
                tag = "OAUTH+DCR"
            else:
                tag = "OAUTH"
        else:
            counts["failed"] += 1
            tag = f"FAIL({rec.get('status_code') or 'err'})"

        if verbose:
            print(f"\n[{counts['done']}/{total}] {rec['id']}: {tag}", flush=True)

        done = counts["done"]
        elapsed = time.monotonic() - started
        pct = (done / total * 100.0) if total else 100.0
        bar = _render_bar(done, total)
        if done % 10 == 0 or done == total:
            sys.stdout.write(
                f"\r{bar} {done}/{total} ({pct:5.1f}%) | "
                f"mcp_ok={counts['handshake_ok']} oauth={counts['oauth']} "
                f"dcr={counts['dcr']} no_url={counts['no_url']} fail={counts['failed']} "
                f"| {elapsed:.1f}s"
            )
            sys.stdout.flush()

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _worker(idx: int, manifest: connector_registry.ConnectorManifest) -> None:
        async with sem:
            try:
                rec = await verify_single_connector(manifest, timeout=timeout)
            except BaseException as exc:
                rec = {
                    "id": manifest.id,
                    "name": manifest.name,
                    "provider": manifest.provider,
                    "server_url": manifest.server_url,
                    "source": manifest.source,
                    "reachable": False,
                    "status_code": None,
                    "handshake_ok": False,
                    "transport": None,
                    "auth_discovered": None,
                    "oauth_supported": False,
                    "dynamic_registration_supported": False,
                    "tools_discovered": False,
                    "tools_count": 0,
                    "last_verified_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                    "verification_status": connector_registry.STATUS_UNVERIFIED,
                }
        _record_result(idx, rec)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        tasks = [asyncio.create_task(_worker(i, manifests[i])) for i in range(start, end)]
        done_tasks, pending_tasks = await asyncio.wait(tasks, timeout=max(18.0, timeout * 4.5))
        for t in pending_tasks:
            t.cancel()
        for i in range(start, end):
            if results[i] is None:
                m = manifests[i]
                _record_result(
                    i,
                    {
                        "id": m.id,
                        "name": m.name,
                        "provider": m.provider,
                        "server_url": m.server_url,
                        "source": m.source,
                        "reachable": False,
                        "status_code": None,
                        "handshake_ok": False,
                        "transport": None,
                        "auth_discovered": None,
                        "oauth_supported": False,
                        "dynamic_registration_supported": False,
                        "tools_discovered": False,
                        "tools_count": 0,
                        "last_verified_at": datetime.now(timezone.utc).isoformat(),
                        "error": "Batch probe timed out.",
                        "verification_status": connector_registry.STATUS_UNVERIFIED,
                    },
                )
        partial = [r for r in results if r is not None]
        _save_report(_build_report(partial, time.monotonic() - started))

    print(flush=True)
    final_results = [r for r in results if r is not None]
    report = _build_report(final_results, time.monotonic() - started)
    _save_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-streaming MCP marketplace auditor.")
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(
        run_live_audit(
            concurrency=args.concurrency,
            timeout=args.timeout,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
    )
    print("\n=== MCP Marketplace Audit Complete ===")
    print(f"Total audited : {report['total_audited']} in {report['elapsed_seconds']}s")
    print("Stats         :", json.dumps(report["stats"], indent=2))
    print("Status summary:", json.dumps(report["summary"], indent=2))
    print(f"Saved report  : {REPORT_PATH}")


if __name__ == "__main__":
    main()
