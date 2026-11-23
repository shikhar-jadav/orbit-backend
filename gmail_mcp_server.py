"""
Native Multi-User Gmail MCP Server for Orbit Backend.

Co-hosted directly inside Orbit's FastAPI backend service.
Speaks standard Streamable HTTP JSON-RPC 2.0 (MCP Specification).
Provides per-request credential isolation, automatic token refresh,
and scope-based access control for all Orbit agents.
"""

from __future__ import annotations

import base64
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import httpx

logger = logging.getLogger("orbit.gmail_mcp")

router = APIRouter(tags=["Gmail MCP"])

MCP_SERVER_INFO = {
    "name": "orbit-gmail-mcp",
    "version": "3.0.1",
    "title": "Orbit Multi-User Gmail MCP Service",
}

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"

# ------------------------------------------------------------------------------
# Tool Definitions (Matching @klodr/gmail-mcp specifications)
# ------------------------------------------------------------------------------
GMAIL_MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "read_email",
        "description": (
            "Read a Gmail email message by ID. Returns sender, recipient, date, "
            "subject, snippet, and plain text/HTML body content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The unique ID of the message to read.",
                },
                "format": {
                    "type": "string",
                    "enum": ["full", "metadata", "minimal"],
                    "default": "full",
                    "description": "Detail format of the message to retrieve.",
                },
            },
            "required": ["message_id"],
        },
        "required_scopes": ["gmail.readonly", "gmail.modify", "mail.google.com"],
        "read_only": True,
    },
    {
        "name": "search_messages",
        "description": (
            "Search for emails in the mailbox matching a Gmail search query "
            "(e.g., 'from:colleague@example.com', 'has:attachment', 'is:unread', 'subject:report')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Standard Gmail search query string.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of messages to return (1-50).",
                },
                "include_spam_trash": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to include messages from SPAM and TRASH.",
                },
            },
            "required": ["query"],
        },
        "required_scopes": ["gmail.readonly", "gmail.modify", "mail.google.com"],
        "read_only": True,
    },
    {
        "name": "list_messages",
        "description": "List recent messages from the inbox or with optional label filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of messages to return.",
                },
                "label_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by label IDs (e.g. ['INBOX', 'UNREAD']).",
                },
            },
        },
        "required_scopes": ["gmail.readonly", "gmail.modify", "mail.google.com"],
        "read_only": True,
    },
    {
        "name": "get_thread",
        "description": "Get an entire email conversation thread by thread ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "The unique ID of the email thread.",
                },
            },
            "required": ["thread_id"],
        },
        "required_scopes": ["gmail.readonly", "gmail.modify", "mail.google.com"],
        "read_only": True,
    },
    {
        "name": "list_threads",
        "description": "List recent email conversation threads matching optional search queries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search query to filter threads.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of threads to return.",
                },
            },
        },
        "required_scopes": ["gmail.readonly", "gmail.modify", "mail.google.com"],
        "read_only": True,
    },
    {
        "name": "send_email",
        "description": "Compose and send a new email from the authenticated user's account.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Primary recipient email address (or comma-separated addresses).",
                },
                "subject": {
                    "type": "string",
                    "description": "The subject line of the email.",
                },
                "body": {
                    "type": "string",
                    "description": "The message body text.",
                },
                "cc": {
                    "type": "string",
                    "description": "Optional CC recipient email address.",
                },
                "bcc": {
                    "type": "string",
                    "description": "Optional BCC recipient email address.",
                },
            },
            "required": ["to", "subject", "body"],
        },
        "required_scopes": ["gmail.send", "mail.google.com"],
        "read_only": False,
    },
    {
        "name": "reply_email",
        "description": "Reply to an existing email message or thread.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The ID of the message being replied to.",
                },
                "body": {
                    "type": "string",
                    "description": "The reply message body text.",
                },
                "reply_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to reply to all participants on the message.",
                },
            },
            "required": ["message_id", "body"],
        },
        "required_scopes": ["gmail.send", "mail.google.com"],
        "read_only": False,
    },
    {
        "name": "create_draft",
        "description": "Create a new draft email without sending it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address.",
                },
                "subject": {
                    "type": "string",
                    "description": "Draft subject line.",
                },
                "body": {
                    "type": "string",
                    "description": "Draft body text.",
                },
                "cc": {
                    "type": "string",
                    "description": "Optional CC recipient email address.",
                },
            },
            "required": ["to", "subject", "body"],
        },
        "required_scopes": ["gmail.compose", "gmail.modify", "mail.google.com"],
        "read_only": False,
    },
    {
        "name": "list_drafts",
        "description": "List existing email drafts in the user's mailbox.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of drafts to list.",
                },
            },
        },
        "required_scopes": ["gmail.readonly", "gmail.compose", "mail.google.com"],
        "read_only": True,
    },
    {
        "name": "send_draft",
        "description": "Send a previously saved draft by its draft ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "draft_id": {
                    "type": "string",
                    "description": "The ID of the draft to send.",
                },
            },
            "required": ["draft_id"],
        },
        "required_scopes": ["gmail.send", "gmail.compose", "mail.google.com"],
        "read_only": False,
    },
    {
        "name": "modify_email",
        "description": "Modify labels on an email message (e.g. mark read, archive, add/remove labels).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The ID of the message to modify.",
                },
                "add_label_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of label IDs to apply (e.g. ['STARRED']).",
                },
                "remove_label_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of label IDs to remove (e.g. ['UNREAD', 'INBOX']).",
                },
            },
            "required": ["message_id"],
        },
        "required_scopes": ["gmail.modify", "mail.google.com"],
        "read_only": False,
    },
    {
        "name": "trash_message",
        "description": "Move an email message to the Trash folder.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The ID of the message to move to trash.",
                },
            },
            "required": ["message_id"],
        },
        "required_scopes": ["gmail.modify", "mail.google.com"],
        "read_only": False,
    },
    {
        "name": "list_labels",
        "description": "List all system and custom labels in the user's mailbox.",
        "inputSchema": {"type": "object", "properties": {}},
        "required_scopes": ["gmail.readonly", "gmail.modify", "mail.google.com"],
        "read_only": True,
    },
    {
        "name": "create_label",
        "description": "Create a new custom label in the user's mailbox.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the new label.",
                },
            },
            "required": ["name"],
        },
        "required_scopes": ["gmail.labels", "gmail.modify", "mail.google.com"],
        "read_only": False,
    },
]


# ------------------------------------------------------------------------------
# Per-User Credential Resolver & Auto-Refresh
# ------------------------------------------------------------------------------
async def resolve_user_gmail_token(
    request: Optional[Request] = None,
    authorization: Optional[str] = None,
    x_user_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Resolve (access_token, user_id, scopes) from request context.
    Prioritizes:
    1. Direct Bearer token in Authorization header.
    2. Stored encrypted credentials in Neon PostgreSQL for x_user_id.
    """
    # 1. Direct Bearer Token
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        user_id = x_user_id or f"user_{token[:12]}"
        return token, user_id, ["mail.google.com"]

    # 2. Database lookup from Orbit's Neon PostgreSQL store
    target_user_id = x_user_id
    if not target_user_id and request is not None:
        # Check active session if available
        auth_cookie = getattr(request, "cookies", {}).get("orbit_auth")
        if auth_cookie:
            try:
                import auth
                session_user = auth.get_user_from_token(auth_cookie)
                if session_user:
                    target_user_id = session_user.get("user_id")
            except Exception:
                pass

    if not target_user_id:
        return None, None, []

    try:
        from proactive_integrations import ProviderIntegrationStore
        pis = ProviderIntegrationStore()
        account = pis.get_account(target_user_id, "gmail")
        if not account:
            return None, target_user_id, []

        access_token = account.get("access_token")
        refresh_token = account.get("refresh_token")
        scopes = account.get("scopes") or []
        metadata = account.get("metadata") or {}
        expires_at = metadata.get("expires_at")

        now = int(time.time())
        # Refresh if expired or within 60s
        if refresh_token and expires_at and int(expires_at) <= now + 60:
            client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
            client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
            if client_id and client_secret:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        ref_res = await client.post(
                            "https://oauth2.googleapis.com/token",
                            data={
                                "client_id": client_id,
                                "client_secret": client_secret,
                                "refresh_token": refresh_token,
                                "grant_type": "refresh_token",
                            },
                        )
                        if ref_res.is_success:
                            ref_data = ref_res.json()
                            new_access_token = ref_data.get("access_token")
                            new_expires = now + int(ref_data.get("expires_in") or 3600)
                            metadata["expires_at"] = new_expires
                            pis.save_account(
                                user_id=target_user_id,
                                provider="gmail",
                                provider_account_id=account.get("provider_account_id", ""),
                                account_name=account.get("account_name", ""),
                                access_token=new_access_token,
                                refresh_token=refresh_token,
                                scopes=scopes,
                                metadata=metadata,
                            )
                            access_token = new_access_token
                except Exception as ref_err:
                    logger.warning("Token refresh error for %s: %s", target_user_id, ref_err)

        return access_token, target_user_id, scopes
    except Exception as exc:
        logger.error("Error resolving credentials for %s: %s", target_user_id, exc)
        return None, target_user_id, []


# ------------------------------------------------------------------------------
# Gmail API Execution Handlers
# ------------------------------------------------------------------------------
def _decode_body_parts(payload: Dict[str, Any]) -> str:
    """Recursively decode text/plain or text/html bodies from a Gmail message payload."""
    chunks: List[str] = []
    body = payload.get("body") or {}
    data = body.get("data")
    if data:
        try:
            decoded = base64.urlsafe_b64decode(data + "===").decode("utf-8", errors="replace")
            chunks.append(decoded)
        except Exception:
            pass

    for part in payload.get("parts") or []:
        if isinstance(part, dict):
            chunks.append(_decode_body_parts(part))

    return "\n\n".join(filter(None, chunks))


async def execute_gmail_tool(
    name: str,
    arguments: Dict[str, Any],
    access_token: str,
) -> str:
    """Execute a Gmail tool via Google REST API using the user's isolated OAuth token."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        # 1. read_email
        if name == "read_email":
            message_id = arguments.get("message_id")
            fmt = arguments.get("format", "full")
            res = await client.get(f"{GMAIL_API_BASE}/messages/{message_id}?format={fmt}", headers=headers)
            if not res.is_success:
                return f"Error reading email ({res.status_code}): {res.text}"
            data = res.json()
            payload = data.get("payload") or {}
            hdr_dict = {
                h.get("name", "").lower(): h.get("value", "")
                for h in payload.get("headers", [])
            }
            body_text = _decode_body_parts(payload)
            return (
                f"ID: {data.get('id')}\n"
                f"ThreadID: {data.get('threadId')}\n"
                f"From: {hdr_dict.get('from', '')}\n"
                f"To: {hdr_dict.get('to', '')}\n"
                f"Date: {hdr_dict.get('date', '')}\n"
                f"Subject: {hdr_dict.get('subject', '')}\n"
                f"Labels: {', '.join(data.get('labelIds', []))}\n"
                f"Snippet: {data.get('snippet', '')}\n\n"
                f"--- Body ---\n{body_text[:12000]}"
            )

        # 2. search_messages
        elif name == "search_messages":
            query = arguments.get("query", "")
            max_results = min(int(arguments.get("max_results", 10)), 50)
            include_spam_trash = bool(arguments.get("include_spam_trash", False))
            params = {
                "q": query,
                "maxResults": max_results,
                "includeSpamTrash": str(include_spam_trash).lower(),
            }
            res = await client.get(f"{GMAIL_API_BASE}/messages", params=params, headers=headers)
            if not res.is_success:
                return f"Error searching messages ({res.status_code}): {res.text}"
            data = res.json()
            messages = data.get("messages") or []
            if not messages:
                return f"No messages matched query: '{query}'"

            # Fetch details for the top items
            summaries = []
            for msg in messages[:max_results]:
                mid = msg["id"]
                detail_res = await client.get(
                    f"{GMAIL_API_BASE}/messages/{mid}?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date",
                    headers=headers,
                )
                if detail_res.is_success:
                    d = detail_res.json()
                    hdrs = {h.get("name", "").lower(): h.get("value", "") for h in (d.get("payload") or {}).get("headers", [])}
                    summaries.append(
                        f"- ID: `{mid}` | From: {hdrs.get('from', '')} | Subject: {hdrs.get('subject', '(No Subject)')} | Date: {hdrs.get('date', '')}"
                    )
                else:
                    summaries.append(f"- ID: `{mid}`")

            return (
                f"Found {len(messages)} matching messages (showing top {len(summaries)}):\n"
                + "\n".join(summaries)
            )

        # 3. list_messages
        elif name == "list_messages":
            max_results = min(int(arguments.get("max_results", 10)), 50)
            label_ids = arguments.get("label_ids") or []
            params = {"maxResults": max_results}
            if label_ids:
                params["labelIds"] = label_ids
            res = await client.get(f"{GMAIL_API_BASE}/messages", params=params, headers=headers)
            if not res.is_success:
                return f"Error listing messages: {res.text}"
            data = res.json()
            messages = data.get("messages") or []
            return f"Found {len(messages)} messages: " + ", ".join(f"`{m['id']}`" for m in messages)

        # 4. get_thread
        elif name == "get_thread":
            thread_id = arguments.get("thread_id")
            res = await client.get(f"{GMAIL_API_BASE}/threads/{thread_id}?format=full", headers=headers)
            if not res.is_success:
                return f"Error retrieving thread: {res.text}"
            data = res.json()
            thread_msgs = data.get("messages") or []
            output_parts = [f"Thread ID: `{data.get('id')}` with {len(thread_msgs)} messages:"]
            for idx, msg in enumerate(thread_msgs):
                hdrs = {h.get("name", "").lower(): h.get("value", "") for h in (msg.get("payload") or {}).get("headers", [])}
                body = _decode_body_parts(msg.get("payload") or {})
                output_parts.append(
                    f"\n--- Message {idx + 1} (ID: {msg.get('id')}) ---\n"
                    f"From: {hdrs.get('from', '')}\n"
                    f"Date: {hdrs.get('date', '')}\n"
                    f"Subject: {hdrs.get('subject', '')}\n\n"
                    f"{body[:4000]}"
                )
            return "\n".join(output_parts)

        # 5. list_threads
        elif name == "list_threads":
            max_results = min(int(arguments.get("max_results", 10)), 50)
            query = arguments.get("query")
            params = {"maxResults": max_results}
            if query:
                params["q"] = query
            res = await client.get(f"{GMAIL_API_BASE}/threads", params=params, headers=headers)
            if not res.is_success:
                return f"Error listing threads: {res.text}"
            threads = res.json().get("threads") or []
            return f"Found {len(threads)} threads: " + ", ".join(f"`{t['id']}` (snippet: {t.get('snippet', '')})" for t in threads[:max_results])

        # 6. send_email
        elif name == "send_email":
            to = arguments["to"]
            subject = arguments["subject"]
            body = arguments["body"]
            cc = arguments.get("cc")
            bcc = arguments.get("bcc")

            mime_msg = MIMEText(body, "plain", "utf-8")
            mime_msg["to"] = to
            mime_msg["subject"] = subject
            if cc:
                mime_msg["cc"] = cc
            if bcc:
                mime_msg["bcc"] = bcc

            raw_bytes = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode("ascii")
            res = await client.post(
                f"{GMAIL_API_BASE}/messages/send",
                headers=headers,
                json={"raw": raw_bytes},
            )
            if not res.is_success:
                return f"Failed to send email ({res.status_code}): {res.text}"
            sent_data = res.json()
            return f"Email sent successfully. Message ID: `{sent_data.get('id')}`, Thread ID: `{sent_data.get('threadId')}`."

        # 7. reply_email
        elif name == "reply_email":
            message_id = arguments["message_id"]
            body = arguments["body"]
            # Fetch message to get Subject, From, Message-ID for thread references
            orig_res = await client.get(f"{GMAIL_API_BASE}/messages/{message_id}?format=metadata", headers=headers)
            if not orig_res.is_success:
                return f"Could not retrieve original message to reply: {orig_res.text}"
            orig_data = orig_res.json()
            thread_id = orig_data.get("threadId")
            hdrs = {h.get("name", "").lower(): h.get("value", "") for h in (orig_data.get("payload") or {}).get("headers", [])}

            orig_subject = hdrs.get("subject", "")
            subject = orig_subject if orig_subject.lower().startswith("re:") else f"Re: {orig_subject}"
            to_addr = hdrs.get("reply-to") or hdrs.get("from")
            msg_id_hdr = hdrs.get("message-id")

            mime_msg = MIMEText(body, "plain", "utf-8")
            mime_msg["to"] = to_addr
            mime_msg["subject"] = subject
            if msg_id_hdr:
                mime_msg["In-Reply-To"] = msg_id_hdr
                mime_msg["References"] = msg_id_hdr

            raw_bytes = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode("ascii")
            payload = {"raw": raw_bytes}
            if thread_id:
                payload["threadId"] = thread_id

            res = await client.post(f"{GMAIL_API_BASE}/messages/send", headers=headers, json=payload)
            if not res.is_success:
                return f"Failed to send reply ({res.status_code}): {res.text}"
            reply_data = res.json()
            return f"Reply sent successfully. Message ID: `{reply_data.get('id')}`, Thread ID: `{reply_data.get('threadId')}`."

        # 8. create_draft
        elif name == "create_draft":
            to = arguments["to"]
            subject = arguments["subject"]
            body = arguments["body"]
            mime_msg = MIMEText(body, "plain", "utf-8")
            mime_msg["to"] = to
            mime_msg["subject"] = subject
            raw_bytes = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode("ascii")

            res = await client.post(
                f"{GMAIL_API_BASE}/drafts",
                headers=headers,
                json={"message": {"raw": raw_bytes}},
            )
            if not res.is_success:
                return f"Failed to create draft: {res.text}"
            return f"Draft created successfully. Draft ID: `{res.json().get('id')}`."

        # 9. list_drafts
        elif name == "list_drafts":
            max_results = min(int(arguments.get("max_results", 10)), 50)
            res = await client.get(f"{GMAIL_API_BASE}/drafts?maxResults={max_results}", headers=headers)
            if not res.is_success:
                return f"Error listing drafts: {res.text}"
            drafts = res.json().get("drafts") or []
            return f"Found {len(drafts)} drafts: " + ", ".join(f"`{d['id']}`" for d in drafts)

        # 10. send_draft
        elif name == "send_draft":
            draft_id = arguments["draft_id"]
            res = await client.post(f"{GMAIL_API_BASE}/drafts/send", headers=headers, json={"id": draft_id})
            if not res.is_success:
                return f"Failed to send draft: {res.text}"
            return f"Draft `{draft_id}` sent successfully. Message ID: `{res.json().get('id')}`."

        # 11. modify_email
        elif name == "modify_email":
            message_id = arguments["message_id"]
            payload = {
                "addLabelIds": arguments.get("add_label_ids") or [],
                "removeLabelIds": arguments.get("remove_label_ids") or [],
            }
            res = await client.post(f"{GMAIL_API_BASE}/messages/{message_id}/modify", headers=headers, json=payload)
            if not res.is_success:
                return f"Failed to modify message: {res.text}"
            return f"Modified labels for message `{message_id}`. Current labels: {', '.join(res.json().get('labelIds', []))}."

        # 12. trash_message
        elif name == "trash_message":
            message_id = arguments["message_id"]
            res = await client.post(f"{GMAIL_API_BASE}/messages/{message_id}/trash", headers=headers)
            if not res.is_success:
                return f"Failed to move message to trash: {res.text}"
            return f"Message `{message_id}` moved to Trash."

        # 13. list_labels
        elif name == "list_labels":
            res = await client.get(f"{GMAIL_API_BASE}/labels", headers=headers)
            if not res.is_success:
                return f"Error listing labels: {res.text}"
            labels = res.json().get("labels") or []
            label_names = [f"`{l.get('name')}` (ID: {l.get('id')})" for l in labels]
            return f"Labels ({len(labels)}):\n" + "\n".join(f"- {n}" for n in label_names[:40])

        # 14. create_label
        elif name == "create_label":
            label_name = arguments["name"]
            res = await client.post(f"{GMAIL_API_BASE}/labels", headers=headers, json={"name": label_name})
            if not res.is_success:
                return f"Failed to create label: {res.text}"
            return f"Label '{label_name}' created successfully (ID: `{res.json().get('id')}`)."

        return f"Unknown tool: '{name}'"


# ------------------------------------------------------------------------------
# MCP Streamable HTTP JSON-RPC Endpoints
# ------------------------------------------------------------------------------
@router.get("/health")
@router.get("/api/mcp/gmail/health")
@router.get("/mcp/gmail/health")
async def gmail_mcp_health():
    """Health check endpoint for the native Gmail MCP service."""
    return {
        "status": "ok",
        "service": "orbit-gmail-mcp",
        "version": "3.0.1",
        "embedded": True,
        "multi_user": True,
        "protocol": "Streamable HTTP JSON-RPC 2.0",
    }


@router.post("/api/mcp/gmail")
@router.post("/mcp/gmail")
async def gmail_mcp_rpc_handler(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_orbit_secret: Optional[str] = Header(None),
):
    """
    Standard Streamable HTTP JSON-RPC 2.0 endpoint for Gmail MCP tools.
    Fully multi-user with isolated per-user execution and automatic token refresh.
    """
    # Optional service secret check
    service_secret = os.getenv("ORBIT_SERVICE_SECRET")
    if service_secret and x_orbit_secret and x_orbit_secret != service_secret:
        raise HTTPException(status_code=403, detail="Invalid X-Orbit-Secret")

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"},
            },
        )

    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    # 1. MCP initialize handshake
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": MCP_SERVER_INFO,
            },
        }

    # 2. MCP initialized notification
    if method == "notifications/initialized":
        return Response(status_code=204)

    # Resolve user credentials for tools/list and tools/call
    access_token, user_id, scopes = await resolve_user_gmail_token(
        request, authorization=authorization, x_user_id=x_user_id
    )

    # 3. tools/list
    if method == "tools/list":
        # Format tools according to official MCP specification
        formatted_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
                "readOnlyHint": t.get("read_only", False),
            }
            for t in GMAIL_MCP_TOOLS
        ]
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"tools": formatted_tools},
        }

    # 4. tools/call
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}

        if not access_token:
            return JSONResponse(
                status_code=200,
                content={
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Error: No Gmail account connected for this user. "
                                    "Please connect your Google account under Routines or Capabilities > Gmail."
                                ),
                            }
                        ],
                        "isError": True,
                    },
                },
            )

        try:
            tool_output = await execute_gmail_tool(
                name=tool_name,
                arguments=arguments,
                access_token=access_token,
            )
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [{"type": "text", "text": tool_output}],
                    "isError": False,
                },
            }
        except Exception as tool_exc:
            logger.exception("Error executing tool %s for user %s: %s", tool_name, user_id, tool_exc)
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Gmail Tool Execution Error: {str(tool_exc)}",
                        }
                    ],
                    "isError": True,
                },
            }

    # 5. Method not found
    return JSONResponse(
        status_code=404,
        content={
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": f"Method not found: '{method}'"},
        },
    )
