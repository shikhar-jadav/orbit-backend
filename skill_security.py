"""NVIDIA SkillSpector integration for scanning personal skills before storage."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

BLOCKED_SKILL_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bat",
    ".cmd",
    ".ps1",
    ".vbs",
    ".scr",
    ".msi",
    ".com",
}

SUSPICIOUS_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |PGP )?PRIVATE KEY-----"),
)

BLOCK_SEVERITIES = {"HIGH", "CRITICAL"}
RISK_SCORE_BLOCK_THRESHOLD = 50
MAX_ISSUES_RETURNED = 12


def find_skillspector_binary() -> Optional[str]:
    """Locate the local NVIDIA SkillSpector CLI executable if installed."""
    override = (os.environ.get("SKILLSPECTOR_BIN") or "").strip()
    if override and Path(override).is_file():
        return override

    found = shutil.which("skillspector")
    if found:
        return found

    exe_name = "skillspector.exe" if os.name == "nt" else "skillspector"
    candidate = Path.home() / ".local" / "bin" / exe_name
    if candidate.is_file():
        return str(candidate)
    return None


def _sanitize_relative_path(raw_path: str) -> str:
    """Validate and normalize a relative file path inside a skill bundle."""
    norm = (raw_path or "").replace("\\", "/").lstrip("/")
    parts = [p for p in norm.split("/") if p]
    if not parts or any(p in {".", ".."} for p in parts) or ":" in parts[0]:
        raise ValueError(f"Unsafe file path '{raw_path}' inside skill bundle.")
    return "/".join(parts)


def synthesize_skill_markdown(name: str, description: str, prompt: str) -> str:
    """Render canonical YAML-frontmatter SKILL.md content from form fields."""
    meta = {
        "name": (name or "").strip()[:120] or "unnamed-skill",
        "description": (description or "").strip()[:600] or "Custom personal workspace skill.",
    }
    frontmatter = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    body = (prompt or "").strip()
    return f"---\n{frontmatter}\n---\n\n{body}\n"


def _run_preflight_checks(
    name: str,
    texts: Sequence[str],
    filenames: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Run immediate deterministic checks for executables, unsafe paths, and leaked secrets."""
    for fname in filenames:
        try:
            _sanitize_relative_path(fname)
        except ValueError as err:
            return _blocked_result(
                scanner="Preflight Security Check",
                version=None,
                reason=str(err),
                issues=[
                    {
                        "id": "PREFLIGHT-PATH",
                        "category": "Path Traversal",
                        "pattern": "Unsafe Relative Path",
                        "severity": "CRITICAL",
                        "location": fname,
                        "finding": str(err),
                        "remediation": "Use relative paths within the skill folder without '..' or drive letters.",
                    }
                ],
                score=100,
                severity="CRITICAL",
                can_trust_override=False,
            )
        ext = Path(fname).suffix.lower()
        if ext in BLOCKED_SKILL_EXTENSIONS:
            reason = f"Security scan blocked executable file '{fname}' inside skill bundle."
            return _blocked_result(
                scanner="Preflight Security Check",
                version=None,
                reason=reason,
                issues=[
                    {
                        "id": "PREFLIGHT-BIN",
                        "category": "Executable Payload",
                        "pattern": "Disallowed Binary/Script Extension",
                        "severity": "CRITICAL",
                        "location": fname,
                        "finding": reason,
                        "remediation": "Remove compiled binaries or shell batch executables from the skill.",
                    }
                ],
                score=95,
                severity="CRITICAL",
                can_trust_override=False,
            )

    for content in texts:
        for pat in SUSPICIOUS_SECRET_PATTERNS:
            if pat.search(content):
                reason = (
                    f"Security scan detected what looks like an embedded API key or private key in '{name}'."
                )
                return _blocked_result(
                    scanner="Preflight Security Check",
                    version=None,
                    reason=reason,
                    issues=[
                        {
                            "id": "PREFLIGHT-SECRET",
                            "category": "Credential Exposure",
                            "pattern": "Hardcoded Secret or Private Key",
                            "severity": "HIGH",
                            "location": "SKILL.md",
                            "finding": reason,
                            "remediation": "Remove hardcoded secrets and configure credentials via connectors or env vars.",
                        }
                    ],
                    score=80,
                    severity="HIGH",
                )
    return None


def _blocked_result(
    *,
    scanner: str,
    version: Optional[str],
    reason: str,
    issues: List[Dict[str, Any]],
    score: int,
    severity: str,
    recommendation: str = "DO_NOT_INSTALL",
    max_issue_severity: Optional[str] = None,
    can_trust_override: bool = True,
) -> Dict[str, Any]:
    return {
        "allowed": False,
        "status": "blocked",
        "scanner": scanner,
        "version": version,
        "score": score,
        "severity": severity,
        "recommendation": recommendation,
        "max_issue_severity": max_issue_severity or severity,
        "issue_count": len(issues),
        "issues": issues[:MAX_ISSUES_RETURNED],
        "checks": [reason],
        "reason": reason,
        "warnings": [reason],
        "can_trust_override": can_trust_override,
    }


def _compact_issue(raw: Dict[str, Any]) -> Dict[str, Any]:
    loc = raw.get("location")
    if isinstance(loc, dict):
        loc_str = str(loc.get("file") or "SKILL.md")
        line = loc.get("start_line")
        if line:
            loc_str = f"{loc_str}:{line}"
    else:
        loc_str = str(loc or "SKILL.md")
    return {
        "id": str(raw.get("id") or raw.get("rule_id") or ""),
        "category": str(raw.get("category") or "Security"),
        "pattern": str(raw.get("pattern") or raw.get("message") or ""),
        "severity": str(raw.get("severity") or "MEDIUM").upper(),
        "location": loc_str,
        "finding": str(raw.get("finding") or raw.get("message") or ""),
        "remediation": str(raw.get("remediation") or ""),
    }


DEFAULT_SCAN_TIMEOUT = 10.0
MAX_SCAN_TIMEOUT = 10.0
MAX_SKILLSPECTOR_BUNDLE_FILES = 4
MAX_SKILLSPECTOR_FILE_BYTES = 120_000

_SKIP_SCAN_DIRS = {
    "assets",
    "frame-presets",
    "presets",
    "templates",
    "fixtures",
    "samples",
    "mockups",
    "fonts",
    "images",
    "media",
    "dist",
    "build",
    "node_modules",
    "__pycache__",
    ".git",
}
_SKIP_SCAN_EXTENSIONS = {
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".map",
}
_CODE_EXTENSIONS = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".sh", ".bash", ".zsh"}
_MANIFEST_FILENAMES = {
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "cargo.toml",
    "plugin.json",
    "mcp.json",
}


def _select_files_for_skillspector(
    bundled_files: Sequence[Tuple[str, bytes]],
) -> List[Tuple[str, bytes]]:
    """Select high-risk scripts, manifests, and core docs so CLI scan finishes within 10s."""
    candidates: List[Tuple[int, str, bytes]] = []
    for rel_path, content in bundled_files:
        safe_rel = _sanitize_relative_path(rel_path)
        if safe_rel.upper() == "SKILL.MD":
            continue
        parts = [p.lower() for p in safe_rel.split("/")]
        filename = parts[-1]
        ext = Path(filename).suffix.lower()
        in_skip_dir = any(part in _SKIP_SCAN_DIRS for part in parts[:-1])
        is_code = ext in _CODE_EXTENSIONS

        if not is_code and (in_skip_dir or ext in _SKIP_SCAN_EXTENSIONS):
            continue
        if len(content) > MAX_SKILLSPECTOR_FILE_BYTES:
            continue

        if is_code:
            is_test = ".test." in filename or "_test." in filename or filename.startswith("test_")
            priority = 1 if is_test else 0
        elif filename in _MANIFEST_FILENAMES:
            priority = 2
        else:
            priority = 3
        candidates.append((priority, safe_rel, content))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(rel_path, content) for _, rel_path, content in candidates[:MAX_SKILLSPECTOR_BUNDLE_FILES]]


def _write_bundle_to_dir(
    root_dir: Path,
    skill_md_text: str,
    bundled_files: Sequence[Tuple[str, bytes]],
) -> int:
    """Materialize SKILL.md and prioritized bundle files under *root_dir*."""
    root_resolved = root_dir.resolve()
    skill_md_path = root_resolved / "SKILL.md"
    skill_md_path.write_text(skill_md_text, encoding="utf-8")

    selected = _select_files_for_skillspector(bundled_files)
    for safe_rel, content in selected:
        dest = (root_resolved / safe_rel).resolve()
        if not dest.is_relative_to(root_resolved):
            raise ValueError(f"Unsafe path traversal detected: '{safe_rel}'")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    return 1 + len(selected)


def _resolve_scan_timeout(file_count: int, explicit_timeout: Optional[float]) -> float:
    """Enforce the strict 10-second maximum security scan budget."""
    if explicit_timeout is not None:
        return min(MAX_SCAN_TIMEOUT, max(1.0, float(explicit_timeout)))
    env_override = (os.environ.get("SKILLSPECTOR_SCAN_TIMEOUT") or "").strip()
    if env_override:
        try:
            return min(MAX_SCAN_TIMEOUT, max(1.0, float(env_override)))
        except ValueError:
            pass
    return DEFAULT_SCAN_TIMEOUT


def _invoke_skillspector_sync(
    binary: str,
    skill_dir: Path,
    timeout: float = DEFAULT_SCAN_TIMEOUT,
) -> Dict[str, Any]:
    """Run `skillspector scan <dir> --no-llm --format json` synchronously."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_COLOR"] = "1"
    env.setdefault("TERM", "dumb")
    env.setdefault("SKILLSPECTOR_OSV_TIMEOUT", "5")

    run_kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "env": env,
        "check": False,
    }
    if os.name == "nt":
        run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # Omit --output on Windows/all OSes so SkillSpector streams clean JSON to stdout.
    proc = subprocess.run(
        [binary, "scan", str(skill_dir), "--no-llm", "--format", "json"],
        **run_kwargs,
    )

    stdout_text = (proc.stdout or "").strip()
    stderr_text = (proc.stderr or "").strip()
    if not stdout_text:
        raise RuntimeError(
            f"SkillSpector returned no JSON output (exit {proc.returncode}): {stderr_text}"
        )

    # Locate JSON payload cleanly in case any CLI notice preceded it
    start_idx = stdout_text.find("{")
    if start_idx == -1:
        raise RuntimeError(f"SkillSpector emitted non-JSON output: {stdout_text[:240]}")
    try:
        payload = json.loads(stdout_text[start_idx:])
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Failed to parse SkillSpector JSON report: {err}") from err
    if not isinstance(payload, dict):
        raise RuntimeError("SkillSpector returned unexpected JSON structure.")
    return payload


def scan_skill_bundle_sync(
    *,
    name: str,
    skill_md_text: str,
    bundled_files: Sequence[Tuple[str, bytes]] = (),
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Scan a skill bundle with preflight checks and NVIDIA SkillSpector CLI."""
    text_samples: List[str] = [skill_md_text]
    filenames: List[str] = ["SKILL.md"]
    for rel, raw in bundled_files:
        filenames.append(rel)
        if len(raw) <= 250_000:
            text_samples.append(raw.decode("utf-8", errors="ignore"))

    preflight_block = _run_preflight_checks(name, text_samples, filenames)
    if preflight_block is not None:
        return preflight_block

    binary = find_skillspector_binary()
    if not binary:
        if os.environ.get("ORBIT_REQUIRE_SKILLSPECTOR", "").strip() == "1":
            return _blocked_result(
                scanner="NVIDIA SkillSpector",
                version=None,
                reason="SkillSpector security scanner binary is required but not installed.",
                issues=[],
                score=100,
                severity="CRITICAL",
            )
        return {
            "allowed": True,
            "status": "passed",
            "scanner": "Preflight Security Check",
            "version": None,
            "score": 0,
            "severity": "LOW",
            "recommendation": "SAFE",
            "max_issue_severity": "NONE",
            "issue_count": 0,
            "issues": [],
            "checks": [
                "YAML & path traversal validation passed",
                "No executable binaries or embedded secrets detected",
            ],
            "warnings": [],
        }

    effective_timeout = _resolve_scan_timeout(len(filenames), timeout)

    try:
        with tempfile.TemporaryDirectory(prefix="orbit_skillspector_") as tmp:
            bundle_dir = Path(tmp) / "skill"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            _write_bundle_to_dir(bundle_dir, skill_md_text, bundled_files)
            report = _invoke_skillspector_sync(binary, bundle_dir, timeout=effective_timeout)
    except ValueError as err:
        return _blocked_result(
            scanner="NVIDIA SkillSpector",
            version=None,
            reason=str(err),
            issues=[],
            score=100,
            severity="CRITICAL",
        )
    except subprocess.TimeoutExpired:
        msg = (
            f"NVIDIA SkillSpector scan timed out after {int(effective_timeout)}s while inspecting "
            f"{len(filenames)} file{'s' if len(filenames) != 1 else ''}. "
            "Remove large build/cache artifacts (such as __pycache__ or node_modules) and retry."
        )
        return _blocked_result(
            scanner="NVIDIA SkillSpector",
            version=None,
            reason=msg,
            issues=[],
            score=0,
            severity="MEDIUM",
            recommendation="SCAN_TIMEOUT",
            max_issue_severity="TIMEOUT",
        )
    except Exception as err:
        return _blocked_result(
            scanner="NVIDIA SkillSpector",
            version=None,
            reason=f"SkillSpector security scan failed: {err}",
            issues=[],
            score=0,
            severity="MEDIUM",
            recommendation="SCAN_ERROR",
            max_issue_severity="ERROR",
        )

    risk = report.get("risk_assessment") or {}
    meta = report.get("metadata") or {}
    raw_issues = report.get("issues") or []

    score = int(risk.get("score") or 0)
    severity = str(risk.get("severity") or "LOW").upper()
    recommendation = str(risk.get("recommendation") or "SAFE").upper()
    max_issue_severity = str(risk.get("max_issue_severity") or "NONE").upper()
    version = str(meta.get("skillspector_version") or "") or None
    compact_issues = [_compact_issue(item) for item in raw_issues if isinstance(item, dict)]

    blocked = (
        recommendation == "DO_NOT_INSTALL"
        or severity in BLOCK_SEVERITIES
        or max_issue_severity in BLOCK_SEVERITIES
        or score >= RISK_SCORE_BLOCK_THRESHOLD
    )

    if blocked:
        top = compact_issues[0] if compact_issues else None
        summary = (
            f"{top['category']} ({top['severity']}): {top['finding']}"
            if top
            else f"Risk score {score}/100 ({recommendation})"
        )
        reason = (
            f"NVIDIA SkillSpector blocked '{name}' "
            f"(score {score}/100, verdict {recommendation}, max severity {max_issue_severity}): {summary}"
        )
        return _blocked_result(
            scanner="NVIDIA SkillSpector",
            version=version,
            reason=reason,
            issues=compact_issues,
            score=score,
            severity=severity,
            recommendation=recommendation,
            max_issue_severity=max_issue_severity,
        )

    warnings = [
        f"[{issue['severity']}] {issue['category']}: {issue['finding']}"
        for issue in compact_issues[:4]
    ]
    checks = [
        f"NVIDIA SkillSpector v{version or '2.x'}: {recommendation} (score {score}/100)",
        f"Static rules & bundle files inspected ({len(filenames)} file{'s' if len(filenames) != 1 else ''})",
    ]
    if not compact_issues:
        checks.append("Zero security findings detected")

    return {
        "allowed": True,
        "status": "passed_with_warnings" if recommendation == "CAUTION" else "passed",
        "scanner": "NVIDIA SkillSpector",
        "version": version,
        "score": score,
        "severity": severity,
        "recommendation": recommendation,
        "max_issue_severity": max_issue_severity,
        "issue_count": len(compact_issues),
        "issues": compact_issues[:MAX_ISSUES_RETURNED],
        "checks": checks,
        "warnings": warnings,
    }


async def scan_skill_bundle(
    *,
    name: str,
    skill_md_text: str,
    bundled_files: Sequence[Tuple[str, bytes]] = (),
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Async wrapper running SkillSpector scan on a worker thread."""
    return await asyncio.to_thread(
        scan_skill_bundle_sync,
        name=name,
        skill_md_text=skill_md_text,
        bundled_files=bundled_files,
        timeout=timeout,
    )
