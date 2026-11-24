"""
Policy Engine & Trust Layer for AI Business Employee Execution.
Implements the Trust Engine blueprint: Action classification, risk policy evaluation,
interrupt payloads, approval governance, and audit verification.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import re


@dataclass
class PolicyDecision:
    requires_approval: bool
    risk_level: str  # "low", "medium", "high", "critical"
    policy_rule: str
    action_summary: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    options: List[Dict[str, str]] = field(default_factory=list)
    blocked: bool = False


# Known safe/read tools that are allowed by default
SAFE_READ_TOOLS = {
    "calculate",
    "get_current_time",
    "get_weather",
    "search_available_tools",
    "render_a2ui_canvas",
    "render_generative_ui",
    "render_visual",
    "manage_memory",
}

# Regex patterns for destructive or critical actions
CRITICAL_PATTERNS = [
    re.compile(r"(delete|drop|truncate|purge|destroy|wipe|unlink)", re.IGNORECASE),
    re.compile(r"(pay|payment|transfer|charge|refund|payout)", re.IGNORECASE),
]

# Regex patterns for high risk / external mutation
HIGH_RISK_PATTERNS = [
    re.compile(r"(run_workspace_code|exec|shell|terminal|bash|eval)", re.IGNORECASE),
    re.compile(r"(send_email|send_message|email|slack|tweet|post|publish)", re.IGNORECASE),
    re.compile(r"(create_user|delete_user|reset_password|change_role|revoke)", re.IGNORECASE),
]

# Regex patterns for medium risk workspace mutations
MEDIUM_RISK_PATTERNS = [
    re.compile(r"(create_file|workspace_files|write_workspace_file|edit_workspace_file|update|modify|patch)", re.IGNORECASE),
]


def classify_action_risk(
    tool_name: str,
    tool_args: Any,
    connector_info: Optional[Dict[str, Any]] = None,
) -> tuple[str, str, str]:
    """
    Classifies a proposed action into (risk_level, policy_rule, action_summary).
    Risk levels:
    - low: Read-only, mathematical, formatting, or safe queries
    - medium: Workspace writes, sandboxed file modifications
    - high: Code execution, sending external messages, publishing, external mutations
    - critical: Destructive operations, drops, payments, credential changes
    """
    clean_name = tool_name.lower().replace("-", "_")
    conn_name = connector_info.get("name", "Workspace") if connector_info else "Workspace"

    if clean_name == "workspace_files" and isinstance(tool_args, dict):
        action = str(tool_args.get("action", "")).lower()
        if action in {"list", "read"}:
            return (
                "low",
                "Auto-allowed: Workspace file listing or reading.",
                f"{'List' if action == 'list' else 'Read'} workspace files",
            )

    if clean_name == "manage_memory" and isinstance(tool_args, dict):
        action = str(tool_args.get("action", "")).lower()
        if action == "delete":
            return (
                "critical",
                "Fresh approval required: Permanently delete stored memory.",
                "Delete a saved memory, rule, or episode",
            )

    # 1. Critical actions (payments, drops, permanent deletes)
    for pat in CRITICAL_PATTERNS:
        if pat.search(clean_name):
            return (
                "critical",
                "Explicit approval required: Consequential financial or destructive action.",
                f"Execute destructive/payment action '{tool_name}' on {conn_name}",
            )

    # 2. High risk actions (code execution, external communication, user admin)
    for pat in HIGH_RISK_PATTERNS:
        if pat.search(clean_name):
            if "run_workspace_code" in clean_name or "exec" in clean_name:
                code_snippet = ""
                if isinstance(tool_args, dict) and tool_args.get("code"):
                    code_snippet = str(tool_args["code"])[:100].strip()
                elif isinstance(tool_args, dict) and tool_args.get("file_path"):
                    code_snippet = f"file: {tool_args['file_path']}"
                summary = f"Run workspace script/code ({code_snippet})" if code_snippet else f"Execute workspace code via '{tool_name}'"
                return (
                    "high",
                    "Approval required by default: Sandboxed Python code execution.",
                    summary,
                )
            return (
                "high",
                "Approval required by default: External communication or state mutation.",
                f"Execute external action '{tool_name}' on {conn_name}",
            )

    # 3. Known safe tools
    if clean_name in SAFE_READ_TOOLS:
        return (
            "low",
            "Auto-allowed: Read-only or safe internal deliverable.",
            f"Read/Inspect via '{tool_name}'",
        )

    # 4. Read-only keywords in tool name
    read_keywords = ("read", "get", "list", "search", "query", "fetch", "check", "describe", "inspect", "probe")
    if any(clean_name.startswith(kw) or f"_{kw}_" in clean_name or clean_name.endswith(kw) for kw in read_keywords):
        return (
            "low",
            "Auto-allowed: Information retrieval or read-only query.",
            f"Retrieve data via '{tool_name}' on {conn_name}",
        )

    # 5. Medium risk workspace mutations
    for pat in MEDIUM_RISK_PATTERNS:
        if pat.search(clean_name):
            target = ""
            if isinstance(tool_args, dict) and tool_args.get("file_path"):
                target = f" on {tool_args['file_path']}"
            return (
                "medium",
                "Policy-controlled: File creation or edit in workspace sandbox.",
                f"Modify workspace file{target} via '{tool_name}'",
            )

    # Default fallback for unknown tools
    return (
        "medium",
        "Policy-controlled: General connector action.",
        f"Execute action '{tool_name}' on {conn_name}",
    )


def redact_sensitive_data(value: Any, key: str = "") -> Any:
    """Return a safe copy of tool arguments for approval cards and audit records."""
    secret_key = re.compile(
        r"(?:secret|token|password|passwd|api[_-]?key|access[_-]?key|"
        r"private[_-]?key|credential|authorization)",
        re.IGNORECASE,
    )
    if secret_key.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(k): (
                "[REDACTED]"
                if str(k).lower() in {"headers", "request_headers", "authorization_headers"}
                else redact_sensitive_data(v, str(k))
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item, key) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_data(item, key) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?i)(bearer\\s+)[A-Za-z0-9._~+/=-]+",
            r"\\1[REDACTED]",
            value,
        )
    return value


def evaluate_policy(
    tool_name: str,
    tool_args: Any,
    connector_info: Optional[Dict[str, Any]] = None,
    approved_tools: Optional[Set[str]] = None,
    user_policy: Optional[Dict[str, Any]] = None,
) -> PolicyDecision:
    """Evaluate an action before execution. Critical actions always need fresh approval."""
    approved_tools = approved_tools or set()
    user_policy = user_policy or {}
    conn_name = connector_info.get("name", "Workspace") if connector_info else "Workspace"
    conn_id = connector_info.get("id") if connector_info else None

    tool_perms = (connector_info.get("tool_permissions") or {}) if connector_info else {}
    raw_mcp_name = (connector_info.get("raw_name") if connector_info else None) or tool_name
    configured_perm = (
        tool_perms.get(raw_mcp_name)
        or tool_perms.get(raw_mcp_name.lower())
        or tool_perms.get(tool_name)
        or tool_perms.get(tool_name.lower())
    )
    clean_name = tool_name.lower()
    raw_clean = raw_mcp_name.lower()
    risk_level, policy_rule, action_summary = classify_action_risk(
        raw_mcp_name, tool_args, connector_info
    )
    safe_details = {
        "tool_name": raw_mcp_name,
        "connector_name": conn_name,
        "connector_id": conn_id,
        "risk_level": risk_level,
        "args": redact_sensitive_data(tool_args if isinstance(tool_args, dict) else {}),
    }

    if configured_perm == "block":
        return PolicyDecision(
            requires_approval=False,
            risk_level=risk_level,
            policy_rule="Blocked by user settings.",
            action_summary=f"Action '{raw_mcp_name}' is blocked by connector permissions.",
            reason="This tool has been disabled or blocked in Settings.",
            details=safe_details,
            blocked=True,
        )

    # Payments, credential changes and destructive actions can never be bypassed by
    # session approval or an "allow always" connector setting.
    if risk_level == "critical":
        return PolicyDecision(
            requires_approval=True,
            risk_level=risk_level,
            policy_rule="Fresh explicit approval required for every critical action.",
            action_summary=action_summary,
            reason="This action is consequential and must be approved immediately before execution.",
            details=safe_details,
            options=[
                {
                    "title": "Approve once (Recommended)",
                    "description": f"Authorize this single action on {conn_name}.",
                },
                {
                    "title": "Decline action",
                    "description": "Reject this action and ask Orbit to find a safer alternative.",
                },
            ],
        )

    if clean_name in approved_tools or raw_clean in approved_tools:
        return PolicyDecision(
            requires_approval=False,
            risk_level=risk_level,
            policy_rule=f"{policy_rule} (Pre-approved in active session)",
            action_summary=action_summary,
            reason="User has already granted permission for this action during this session.",
            details=safe_details,
        )

    if configured_perm == "allow":
        return PolicyDecision(
            requires_approval=False,
            risk_level=risk_level,
            policy_rule=f"{policy_rule} (Always allowed in Settings)",
            action_summary=action_summary,
            reason=f"Configured as 'Allow always' on connector '{conn_name}'.",
            details=safe_details,
        )

    if risk_level == "low" and configured_perm != "approval":
        return PolicyDecision(
            requires_approval=False,
            risk_level="low",
            policy_rule=policy_rule,
            action_summary=action_summary,
            reason="Read-only or safe internal operation.",
            details=safe_details,
        )

    if clean_name in ("workspace_files", "create_file", "write_workspace_file", "edit_workspace_file"):
        require_file_approval = bool(user_policy.get("require_file_approval", False))
        if not require_file_approval:
            return PolicyDecision(
                requires_approval=False,
                risk_level="medium",
                policy_rule=policy_rule,
                action_summary=action_summary,
                reason="Workspace sandbox write permitted by default.",
                details=safe_details,
            )

    if connector_info:
        options = [
            {
                "title": "Allow once (Recommended)",
                "description": f"Allow Orbit to run '{raw_mcp_name}' on {conn_name} for this step.",
            },
            {
                "title": "Always allow this tool",
                "description": f"Approve and remember 'Allow always' for '{raw_mcp_name}' in Settings.",
            },
            {
                "title": "Decline action",
                "description": "Reject this tool call and let Orbit adapt.",
            },
        ]
    elif "run_workspace_code" in clean_name:
        options = [
            {
                "title": "Allow execution (Recommended)",
                "description": "Run this script inside the workspace sandbox once.",
            },
            {
                "title": "Decline execution",
                "description": "Do not run the script; ask Orbit to show the code instead.",
            },
        ]
    else:
        options = [
            {
                "title": "Approve action (Recommended)",
                "description": f"Proceed with {action_summary}.",
            },
            {
                "title": "Approve for this session",
                "description": "Allow this action for the remainder of this session.",
            },
            {
                "title": "Decline action",
                "description": "Cancel this step and request an alternative approach.",
            },
        ]

    return PolicyDecision(
        requires_approval=True,
        risk_level=risk_level,
        policy_rule=policy_rule,
        action_summary=action_summary,
        reason=f"Action requires review under policy '{policy_rule}'.",
        details=safe_details,
        options=options,
    )
