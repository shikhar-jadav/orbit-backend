"""Safe, read-only verification utility for Orbit's MCP marketplace catalogue.

Usage:
    python verify_marketplace.py --limit 50 --concurrency 10
    python verify_marketplace.py --all --concurrency 20

Performs ONLY read-only MCP handshake + tools/list probes (never executes tools).
Saves structured results to data/marketplace_verification.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import connector_registry
import mcp_client

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "data" / "marketplace_verification.json"


async def verify_single_connector(
    manifest: connector_registry.ConnectorManifest,
    *,
    timeout: float = 6.0,
) -> Dict[str, Any]:
    """Run a safe, read-only protocol probe against a single connector manifest."""
    now_iso = datetime.now(timezone.utc).isoformat()
    base_record: Dict[str, Any] = {
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
        "last_verified_at": now_iso,
        "error": None,
        "verification_status": manifest.verification_status,
    }

    if manifest.verification_status == connector_registry.STATUS_UNSUPPORTED:
        base_record["error"] = manifest.verification_notes
        return base_record

    if not manifest.server_url:
        base_record["verification_status"] = connector_registry.STATUS_UNVERIFIED
        base_record["error"] = "No remote server URL."
        return base_record

    try:
        probe = await asyncio.wait_for(
            mcp_client.probe_mcp_server(manifest.server_url, timeout=timeout),
            timeout=timeout + 3.0,
        )
    except asyncio.TimeoutError:
        base_record["verification_status"] = connector_registry.STATUS_UNVERIFIED
        base_record["error"] = "Timed out while probing MCP server."
        return base_record
    except Exception as exc:
        base_record["verification_status"] = connector_registry.STATUS_UNVERIFIED
        base_record["error"] = str(exc)
        return base_record

    base_record["reachable"] = bool(probe.get("reachable"))
    base_record["status_code"] = probe.get("status_code")
    base_record["handshake_ok"] = bool(probe.get("handshake_ok"))
    base_record["transport"] = probe.get("transport")
    base_record["error"] = probe.get("error")

    oauth_meta = probe.get("oauth_metadata")
    if isinstance(oauth_meta, dict) and oauth_meta.get("authorization_endpoint"):
        base_record["oauth_supported"] = True
        base_record["dynamic_registration_supported"] = bool(oauth_meta.get("registration_endpoint"))
        base_record["auth_discovered"] = "oauth2"
    elif probe.get("status_code") in (401, 403):
        base_record["auth_discovered"] = probe.get("signin_provider_status") or "auth_required"

    tools = probe.get("tools") or []
    base_record["tools_count"] = len(tools)
    base_record["tools_discovered"] = len(tools) > 0

    if manifest.verification_status == connector_registry.STATUS_SETUP_REQUIRED:
        base_record["verification_status"] = connector_registry.STATUS_SETUP_REQUIRED
    elif probe.get("handshake_ok") and probe.get("detected_auth") in ("none", "later"):
        base_record["verification_status"] = connector_registry.STATUS_VERIFIED
    elif probe.get("status_code") in (401, 403) or base_record["oauth_supported"]:
        base_record["verification_status"] = connector_registry.STATUS_AUTH_REQUIRED
    elif not probe.get("reachable"):
        base_record["verification_status"] = connector_registry.STATUS_UNVERIFIED
    else:
        # Reachable HTTP 200 without valid MCP handshake
        base_record["verification_status"] = connector_registry.STATUS_UNVERIFIED
        if not base_record["error"]:
            base_record["error"] = "Endpoint responded without a valid MCP initialize handshake."

    return base_record


async def run_verification_audit(
    *,
    limit: Optional[int] = None,
    concurrency: int = 15,
    timeout: float = 5.0,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Audit the marketplace catalogue with bounded concurrency."""
    manifests = connector_registry.load_marketplace_manifests()
    if limit is not None and limit > 0:
        manifests = manifests[:limit]

    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Optional[Dict[str, Any]]] = [None] * len(manifests)

    async def _worker(idx: int, m: connector_registry.ConnectorManifest) -> None:
        async with sem:
            results[idx] = await verify_single_connector(m, timeout=timeout)

    started = time.monotonic()
    await asyncio.gather(*(_worker(i, m) for i, m in enumerate(manifests)))
    elapsed = round(time.monotonic() - started, 2)

    final_results = [r for r in results if r is not None]
    summary: Dict[str, int] = {}
    for r in final_results:
        st = str(r.get("verification_status") or "unverified")
        summary[st] = summary.get(st, 0) + 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_audited": len(final_results),
        "elapsed_seconds": elapsed,
        "summary": summary,
        "results": final_results,
    }
    if save_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Orbit MCP marketplace catalogue.")
    parser.add_argument("--limit", type=int, default=50, help="Number of entries to verify (ignored if --all).")
    parser.add_argument("--all", action="store_true", help="Verify all entries in the catalogue.")
    parser.add_argument("--concurrency", type=int, default=15, help="Max concurrent probes.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-server probe timeout in seconds.")
    args = parser.parse_args()

    lim = None if args.all else args.limit
    report = asyncio.run(
        run_verification_audit(limit=lim, concurrency=args.concurrency, timeout=args.timeout)
    )
    print(f"Audited {report['total_audited']} connectors in {report['elapsed_seconds']}s")
    print("Summary:", json.dumps(report["summary"], indent=2))
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
