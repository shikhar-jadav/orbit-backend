import datetime
import math
import ast
import json
from typing import Any, Optional, Dict, List, Tuple, Literal
from langchain_core.tools import tool

@tool
def render_a2ui_canvas(title: str, spec: str) -> str:
    """
    Renders an interactive A2UI 1.0 canvas in the workspace using the official component catalog.
    Args:
        title: Title of the canvas (e.g. 'Financial Analysis Dashboard', 'System Status')
        spec: JSON string conforming to A2UI component catalog schema with components:
              - Card: {"type": "card", "title": "...", "description": "...", "badge": "..."}
              - Metric: {"type": "metric", "label": "...", "value": "...", "delta": "...", "status": "positive"|"negative"|"neutral"}
              - BarChart: {"type": "chart", "title": "...", "data": [{"label": "...", "value": 10}]}
              - Table: {"type": "table", "headers": [...], "rows": [[...]]}
              - Button: {"type": "button", "label": "...", "action_id": "...", "variant": "primary"|"secondary"|"outline"|"destructive", "payload": {}}
              - Alert: {"type": "alert", "variant": "info"|"warning"|"success"|"error", "title": "...", "message": "..."}
    """
    try:
        parsed = json.loads(spec) if isinstance(spec, str) else spec
        return json.dumps({"status": "rendered", "title": title, "spec": parsed}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "rendered", "title": title, "spec": {"type": "card", "title": title, "raw": str(spec)}}, ensure_ascii=False)


@tool
def render_generative_ui(title: str, html_content: str, description: str = "") -> str:
    """
    Renders an interactive Generative UI widget, visual diagram, schematic, or illustration inline in the chat.
    Use this tool whenever the user asks for a diagram, visual model, illustration, or schematic (e.g. hydrogen atom, atomic structure, solar system, network topology, architecture flowchart, biology/physics diagrams), or interactive widgets with controls.
    Args:
        title: Title of the diagram or widget (e.g. 'Hydrogen Atom Structure', 'System Architecture Diagram')
        html_content: Complete self-contained HTML/SVG code for the diagram or widget, styled with SVG shapes or Tailwind CSS classes.
        description: Brief description of the diagram or widget capability.
    """
    import json
    return json.dumps({
        "status": "generative_ui_rendered",
        "title": title,
        "description": description,
        "html": html_content
    }, ensure_ascii=False)


@tool
def workspace_files(
    action: Literal["list", "read", "create", "replace", "append", "edit", "copy"],
    path: str = ".",
    file_path: Optional[str] = None,
    source_path: Optional[str] = None,
    destination_path: Optional[str] = None,
    content: str = "",
    start_line: int = 1,
    end_line: int = 120,
    find_text: Optional[str] = None,
    replacement: Optional[str] = None,
    content_encoding: Literal["text", "base64"] = "text",
) -> str:
    """List, read, create, update, or copy files in the current workspace.

    Choose `list` to inspect a directory (use `path`), `read` to read a line range,
    `create`, `replace`, `append`, or `edit` to change a file (use `file_path`),
    or `copy` to copy a file from one path to another (use `source_path` and `destination_path`).
    
    IMPORTANT: The 'inputs/' folder contains all user uploads (attachments, images, documents) and is
    strictly read-only. If you need to edit or modify an uploaded file, use action='copy' with
    source_path='inputs/<filename>' and destination_path='<filename>' to copy it to the main workspace first!

    Args:
        action: Workspace operation to perform ('list', 'read', 'create', 'replace', 'append', 'edit', 'copy').
        path: Directory to list when action='list'.
        file_path: Workspace-relative file to read or change.
        source_path: Source file path when action='copy' (e.g. 'inputs/data.csv').
        destination_path: Destination file path when action='copy' (e.g. 'data.csv').
        content: UTF-8 content, appended text, or complete base64 bytes for create/replace.
        start_line: First line to read (1-indexed).
        end_line: Last line to read (inclusive).
        find_text: Exact text to replace when action='edit'.
        replacement: Replacement text when action='edit'.
        content_encoding: 'text' or 'base64'.
    """
    import json
    from workspace import create_or_update_file, list_files, read_file, copy_file

    if action == "list":
        result = list_files(path)
    elif action == "copy":
        src = source_path or file_path
        if not src or not destination_path:
            result = {"error": "Both source_path (or file_path) and destination_path are required for action 'copy'."}
        else:
            result = copy_file(source_path=src, destination_path=destination_path)
    elif not file_path:
        result = {"error": f"file_path is required for action '{action}'."}
    elif action == "read":
        result = read_file(file_path, start_line=start_line, end_line=end_line)
    else:
        result = create_or_update_file(
            file_path=file_path,
            content=content,
            operation=action,
            find_text=find_text,
            replacement=replacement,
            content_encoding=content_encoding,
        )
    return json.dumps(result, ensure_ascii=False)


@tool
def run_workspace_code(command_or_file: str, timeout_seconds: int = 15) -> str:
    """
    Execute a Python script or command inside the isolated workspace sandbox.
    Captures stdout, stderr, execution duration, and exit codes.
    Args:
        command_or_file: File name (e.g. 'main.py', 'test_script.py') or inline Python expression/code.
        timeout_seconds: Maximum time in seconds before execution is killed (default 15).
    """
    import json
    import asyncio
    import concurrent.futures
    from workspace import run_code

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            res = pool.submit(asyncio.run, run_code(command_or_file, timeout_seconds=timeout_seconds)).result()
    else:
        res = asyncio.run(run_code(command_or_file, timeout_seconds=timeout_seconds))

    return json.dumps(res, ensure_ascii=False)


@tool
def ask_question(questions: list) -> str:
    """
    Ask the user one or more multiple-choice or approval questions directly in chat.
    The user is presented with an interactive card above the input composer to choose from.
    You can ask single or multiple questions at a time (multi-step sequence: < 1 of 3 >, < 2 of 3 >, etc.).
    You can supply ANY contextual options you want (not just allow/deny), each with optional descriptions.
    Questions can be single-choice or multi-select with checkboxes.

    Args:
        questions: List of question objects, each containing:
            - 'question' (str): The question to ask the user.
            - 'options' (list): List of strings (e.g. ['Python', 'TypeScript', 'Rust']) or objects with
              'title' and optional 'description' (e.g. [{'title': 'Ask every time (Recommended)', 'description': 'Pause and wait for explicit approval on each step'}, ...]).
            - 'is_multi_select' (bool, optional): If True, renders checkboxes allowing multiple selections. Default False.
            - 'allow_custom' (bool, optional): If True, provides 'Something else' for custom typed input. Default True.
            - 'can_skip' (bool, optional): If True, shows 'Skip' button. Default True.
    """
    import json
    return json.dumps({
        "status": "question_asked",
        "questions": questions,
    }, ensure_ascii=False)


@tool
def search_available_tools(query: str, limit: int = 5) -> str:
    """
    Search and dynamically load available tools from connected MCP connectors (e.g. Notion, LangChain, Google Drive, GitHub, etc.) and personal skills.
    Use this tool whenever the user asks about external services, workspaces, files, databases, or capabilities not currently in your active tool list.
    Args:
        query: Keywords describing the desired action or service (e.g., 'notion search pages', 'notion databases', 'read notion page').
        limit: Maximum number of relevant tools to match and load (default: 5).
    """
    import json
    return json.dumps({
        "status": "search_tools",
        "query": query,
        "limit": limit,
    }, ensure_ascii=False)


@tool
def manage_memory(
    action: Literal["save", "search", "update", "delete", "list"],
    memory_type: Literal["fact", "rule", "episode", "preference", "all"] = "fact",
    content: str = "",
    fact: str = "",
    key: str = "",
    query: str = "",
    task: str = "",
    solution_steps: str = "",
    tools_used: str = "",
    category: str = "general",
    limit: int = 5,
) -> str:
    """Autonomous tri-layer memory management for the current user.

    Use this tool to autonomously remember, recall, update, or remove:
    1. Facts (`memory_type="fact"`): Important user details, background, preferences.
    2. Rules & Instructions (`memory_type="rule"`): Procedural guidelines and behavioral
       rules learned from user feedback/corrections (e.g. formatting, coding styles, constraints).
    3. Episodes (`memory_type="episode"`): Past successful problem-solving workflows, debugging
       steps, and tool sequences for few-shot learning on similar future tasks.
    4. Preferences (`memory_type="preference"`): Key-value configuration preferences.

    Args:
        action: 'save' to persist, 'search' to find, 'update' to modify, 'delete' to remove, or 'list' to enumerate.
        memory_type: Category of memory ('fact', 'rule', 'episode', 'preference', or 'all').
        content: The text content of the fact, rule, instruction, or preference value (or use 'fact').
        fact: Backwards-compatible alias for 'content' when saving facts.
        key: Specific identifier or tag (optional for save; required for update/delete).
        query: Keywords or natural language query when searching.
        task: When saving an episode, a concise description of the task solved.
        solution_steps: When saving an episode, the resolution strategy or steps taken.
        tools_used: When saving an episode, comma-separated list of tools used.
        category: Category tag for rules (e.g., 'coding', 'tone', 'formatting', 'general').
        limit: Maximum items to return when searching or listing (default 5).
    """
    from memory_store import active_memory_store_var, active_user_id_var
    import json
    store = active_memory_store_var.get()
    user_id = active_user_id_var.get()
    if not store or not user_id:
        return json.dumps({"status": "error", "message": "Memory store not available in current session."})

    text_content = (content or fact).strip()

    if action == "save":
        if memory_type == "rule":
            if not text_content:
                return json.dumps({"status": "error", "message": "content is required when saving a rule."})
            saved_key = store.save_user_instruction(user_id, text_content, category=category, key=key or None)
            return json.dumps({
                "status": "rule_saved",
                "key": saved_key,
                "rule": text_content,
                "category": category,
            }, ensure_ascii=False)

        elif memory_type == "episode":
            task_desc = task.strip() or text_content
            if not task_desc:
                return json.dumps({"status": "error", "message": "task or content is required when saving an episode."})
            tools_list = [t.strip() for t in tools_used.split(",") if t.strip()] if tools_used else []
            saved_key = store.save_episode(
                user_id,
                task=task_desc,
                solution_steps=solution_steps.strip() or "Resolved successfully.",
                tools_used=tools_list,
                key=key or None,
            )
            return json.dumps({
                "status": "episode_saved",
                "key": saved_key,
                "task": task_desc,
            }, ensure_ascii=False)

        elif memory_type == "preference":
            pref_key = key or "general"
            store.save_user_preference(user_id, pref_key, text_content)
            return json.dumps({
                "status": "preference_saved",
                "key": pref_key,
                "value": text_content,
            }, ensure_ascii=False)

        else:  # default "fact"
            if not text_content:
                return json.dumps({"status": "error", "message": "fact or content is required when action='save'."})
            saved_key = store.save_user_memory(user_id, {"fact": text_content}, key=key or None)
            return json.dumps({
                "status": "memory_saved",
                "key": saved_key,
                "fact": text_content,
            }, ensure_ascii=False)

    elif action == "update":
        if not key:
            return json.dumps({"status": "error", "message": "key is required when action='update'."})
        if memory_type == "rule":
            store.save_user_instruction(user_id, text_content, category=category, key=key)
            return json.dumps({"status": "rule_updated", "key": key, "rule": text_content}, ensure_ascii=False)
        elif memory_type == "episode":
            store.save_episode(
                user_id,
                task=task.strip() or text_content,
                solution_steps=solution_steps.strip(),
                key=key,
            )
            return json.dumps({"status": "episode_updated", "key": key}, ensure_ascii=False)
        elif memory_type == "preference":
            store.save_user_preference(user_id, key, text_content)
            return json.dumps({"status": "preference_updated", "key": key, "value": text_content}, ensure_ascii=False)
        else:
            store.save_user_memory(user_id, {"fact": text_content}, key=key)
            return json.dumps({"status": "memory_updated", "key": key, "fact": text_content}, ensure_ascii=False)

    elif action == "delete":
        if not key:
            return json.dumps({"status": "error", "message": "key is required when action='delete'."})
        if memory_type == "rule":
            store.delete_user_instruction(user_id, key)
            return json.dumps({"status": "rule_deleted", "key": key})
        elif memory_type == "episode":
            store.delete_episode(user_id, key)
            return json.dumps({"status": "episode_deleted", "key": key})
        elif memory_type == "preference":
            store.delete((user_id, "preferences"), key)
            return json.dumps({"status": "preference_deleted", "key": key})
        else:
            store.delete_user_memory(user_id, key)
            return json.dumps({"status": "memory_deleted", "key": key})

    elif action == "list":
        if memory_type == "rule":
            rules = store.get_user_instructions(user_id, limit=limit)
            return json.dumps({"status": "rules_listed", "count": len(rules), "rules": rules}, ensure_ascii=False, default=str)
        elif memory_type == "episode":
            episodes = store.get_relevant_episodes(user_id, query="", limit=limit)
            return json.dumps({"status": "episodes_listed", "count": len(episodes), "episodes": episodes}, ensure_ascii=False, default=str)
        elif memory_type == "preference":
            prefs = store.get_user_preferences(user_id)
            return json.dumps({"status": "preferences_listed", "preferences": prefs}, ensure_ascii=False, default=str)
        else:
            mems = store.get_user_memories(user_id, limit=limit)
            return json.dumps({"status": "memories_listed", "count": len(mems), "memories": mems}, ensure_ascii=False, default=str)

    elif action == "search":
        if limit < 1:
            return json.dumps({"status": "error", "message": "limit must be at least 1."})

        if memory_type == "rule":
            results = store.search((user_id, "instructions"), query=query or None, limit=limit)
            items = [{"key": r.key, "value": r.value} for r in results]
            return json.dumps({"status": "rules_retrieved", "count": len(items), "rules": items}, ensure_ascii=False, default=str)
        elif memory_type == "episode":
            episodes = store.get_relevant_episodes(user_id, query=query, limit=limit)
            return json.dumps({"status": "episodes_retrieved", "count": len(episodes), "episodes": episodes}, ensure_ascii=False, default=str)
        elif memory_type == "preference":
            prefs = store.get_user_preferences(user_id)
            return json.dumps({"status": "preferences_retrieved", "preferences": prefs}, ensure_ascii=False, default=str)
        elif memory_type == "all":
            all_mems = store.get_user_memories(user_id, limit=limit)
            all_rules = store.get_user_instructions(user_id, limit=limit)
            all_eps = store.get_relevant_episodes(user_id, query=query, limit=limit)
            return json.dumps({
                "status": "all_memory_retrieved",
                "memories": all_mems,
                "rules": all_rules,
                "episodes": all_eps,
            }, ensure_ascii=False, default=str)
        else:
            if query:
                memories = store.search((user_id, "memories"), query=query, limit=limit)
                items = [{"key": memory.key, "value": memory.value} for memory in memories]
            else:
                items = store.get_user_memories(user_id, limit=limit)
            return json.dumps({
                "status": "memories_retrieved",
                "count": len(items),
                "memories": items,
            }, ensure_ascii=False, default=str)

    return json.dumps({"status": "error", "message": f"Unsupported action '{action}'."})


def _get_proactive_store():
    from proactive_engine import active_proactive_store_var

    store = active_proactive_store_var.get()
    if store is None:
        raise RuntimeError("The PostgreSQL proactive store is not bound to this agent run.")
    return store


def _find_routine_by_id_or_keyword(store, user_id: str, target: str):
    clean = str(target).strip()
    r = store.get_routine(clean)
    if r and (not user_id or r.get("user_id") == user_id or user_id == "default_user"):
        return r
    routines = store.list_routines(user_id=user_id)
    matched = [
        item for item in routines
        if clean.lower() in item.get("name", "").lower()
        or clean.lower() in item.get("id", "").lower()
        or clean.lower() in str(item.get("action_prompt", "")).lower()
    ]
    return matched[0] if matched else None


@tool
def manage_routine(
    action: Literal["schedule", "create", "list", "pause", "resume", "delete", "test"],
    name: Optional[str] = None,
    action_prompt: Optional[str] = None,
    run_at: Optional[str] = None,
    cron_expression: Optional[str] = None,
    routine_id: Optional[str] = None,
    filter_status: str = "all",
    require_approval_for_writes: bool = True,
    routine_type: Optional[Literal["reminder", "task"]] = None,
) -> str:
    """Create or manage reminders and routines using structured schedule values.

    The model resolves the user's request; this tool does not parse natural-language time.
    For `schedule`, provide name, action_prompt, and run_at as an ISO-8601 timestamp WITH timezone.
    For `create`, provide name, action_prompt, and a standard five-field cron_expression evaluated in UTC.
    For `list`, optionally set filter_status to active, paused, or all.
    For pause, resume, delete, or test, provide routine_id (ID or title keyword); delete also accepts 'all'.
    If the date, time, or timezone is ambiguous, ask the user before calling this tool.
    """
    from datetime import datetime, timezone
    from memory_store import active_user_id_var
    from workspace import active_thread_id_var
    from proactive_engine import format_display_time
    import croniter

    user_id = active_user_id_var.get() or "default_user"
    store = _get_proactive_store()

    if action in {"schedule", "create"}:
        if not name or not name.strip():
            return f"Error: name is required to {action} a routine."
        if not action_prompt or not action_prompt.strip():
            return "Error: action_prompt is required."

        if action == "schedule":
            if not run_at or not run_at.strip():
                return "Error: provide run_at as an ISO-8601 timestamp with timezone."
            try:
                target = datetime.fromisoformat(run_at.strip().replace("Z", "+00:00"))
            except ValueError:
                return "Error: run_at must be a valid ISO-8601 timestamp with timezone."
            if target.tzinfo is None or target.utcoffset() is None:
                return "Error: run_at must include a timezone offset; clarify the user's timezone if needed."
            target = target.astimezone(timezone.utc)
            now_utc = datetime.now(timezone.utc)
            if target <= now_utc:
                now_local = now_utc.astimezone()
                return (
                    f"Error: run_at ({run_at.strip()}) must be in the future; "
                    f"current time is {now_local.isoformat(timespec='seconds')} "
                    f"(UTC: {now_utc.isoformat(timespec='seconds')}). "
                    f"Recalculate run_at from this exact current time and retry."
                )
            next_run = target.isoformat(timespec="seconds")
            routine = store.create_routine(
                user_id=user_id,
                name=name.strip(),
                action_prompt=action_prompt.strip(),
                trigger_type="timer",
                routine_type=routine_type or "reminder",
                schedule_expression=next_run,
                schedule_display=f"Once at {next_run}",
                one_shot=True,
                next_run_at=next_run,
                target_thread_id=active_thread_id_var.get(),
            )
            return (
                f"Reminder scheduled: '{routine['name']}'\n"
                f"- ID: `{routine['id']}`\n"
                f"- Time: {format_display_time(next_run)}\n"
                f"- Action: {routine['action_prompt']}"
            )

        if not cron_expression or not cron_expression.strip():
            return "Error: provide a five-field UTC cron_expression."
        expression = cron_expression.strip()
        if len(expression.split()) != 5 or not croniter.croniter.is_valid(expression):
            return "Error: cron_expression must be a valid five-field cron expression evaluated in UTC."
        next_run = croniter.croniter(expression, datetime.now(timezone.utc)).get_next(datetime)
        next_iso = next_run.astimezone(timezone.utc).isoformat(timespec="seconds")
        routine = store.create_routine(
            user_id=user_id,
            name=name.strip(),
            action_prompt=action_prompt.strip(),
            trigger_type="cron",
            routine_type=routine_type or "task",
            schedule_expression=expression,
            schedule_display=f"UTC cron: {expression}",
            one_shot=False,
            next_run_at=next_iso,
            target_thread_id=active_thread_id_var.get(),
            require_approval=require_approval_for_writes,
        )
        return (
            f"Recurring routine created: '{routine['name']}'\n"
            f"- ID: `{routine['id']}`\n"
            f"- UTC cron: `{expression}`\n"
            f"- Next execution: {format_display_time(next_iso)}\n"
            f"- Action: {routine['action_prompt']}"
        )

    if action == "list":
        clean_filter = filter_status.strip().lower()
        if clean_filter not in {"all", "active", "paused"}:
            return "Error: filter_status must be 'all', 'active', or 'paused'."
        routines = store.list_routines(user_id=user_id)
        filtered = [
            routine for routine in routines
            if clean_filter == "all"
            or (clean_filter == "active" and routine["enabled"])
            or (clean_filter == "paused" and not routine["enabled"])
        ]
        if not filtered:
            return "No scheduled reminders or routines found."
        lines = [f"Found {len(filtered)} scheduled reminder(s) / routine(s):"]
        for routine in filtered:
            kind = "One-Time Reminder" if routine["one_shot"] else "Recurring Routine"
            status = "Active" if routine["enabled"] else "Paused"
            lines.append(
                f"- [{status}] `{routine['id']}` | '{routine['name']}' ({kind})\n"
                f"  * Next execution: {format_display_time(routine.get('next_run_at'))}\n"
                f"  * Prompt: {routine['action_prompt']}"
            )
        return "\n".join(lines)

    if not routine_id or not routine_id.strip():
        if action == "delete":
            return "Error: provide routine_id to delete a routine, or explicitly pass 'all' to delete all routines."
        return f"Error: routine_id is required to {action} a routine."
    if action == "delete" and routine_id.strip().lower() in {
        "all", "all_routines", "*", "everything"
    }:
        count = store.delete_all_routines(user_id=user_id)
        return f"Successfully deleted all {count} scheduled routine(s) / reminder(s)."

    routine = _find_routine_by_id_or_keyword(store, user_id, routine_id)
    if not routine:
        return f"Routine matching '{routine_id}' was not found."

    if action == "pause":
        store.update_routine(routine["id"], enabled=False)
        return f"Successfully paused routine '{routine['name']}' (ID: `{routine['id']}`)."
    if action == "resume":
        next_iso = None
        expression = routine.get("schedule_expression")
        if expression and not routine["one_shot"]:
            next_run = croniter.croniter(expression, datetime.now(timezone.utc)).get_next(datetime)
            next_iso = next_run.astimezone(timezone.utc).isoformat(timespec="seconds")
        store.update_routine(routine["id"], enabled=True, next_run_at=next_iso)
        return f"Successfully resumed routine '{routine['name']}'. Next execution: {format_display_time(next_iso)}."
    if action == "delete":
        store.delete_routine(routine["id"])
        return f"Successfully deleted routine '{routine['name']}' (ID: `{routine['id']}`)."
    if action == "test":
        try:
            from server import proactive_engine
        except ImportError:
            proactive_engine = None
        if proactive_engine and hasattr(proactive_engine, "execute_routine_isolated"):
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop:
                loop.create_task(
                    proactive_engine.execute_routine_isolated(
                        routine["id"], trigger_source="manual_test"
                    )
                )
                return f"Test run dispatched for '{routine['name']}' (ID: `{routine['id']}`)."
        run_id = store.create_run(routine["id"], trigger_source="manual_test")
        return f"Test run enqueued for '{routine['name']}' (ID: `{run_id}`)."
    return f"Error: unsupported action '{action}'."


@tool
async def create_event_routine(
    name: str,
    action_prompt: str,
    source: Literal["github", "slack", "gmail", "webhook"],
    event_type: str,
    when_to_run: str,
    filters: Optional[Dict[str, Any]] = None,
    repository: Optional[str] = None,
    channel_name: Optional[str] = None,
) -> str:
    """Create an event-triggered routine from a plain-language request.

    Use this for new GitHub, Slack, or custom webhook triggers. Keep GitHub
    triggers to one owner/repository and Slack triggers to one channel. The
    caller must provide a concise trigger description and deterministic filters.
    """
    import os
    from urllib.parse import urlparse
    from fastapi import HTTPException
    from starlette.requests import Request
    from memory_store import active_user_id_var

    import server

    base_url = os.getenv("ORBIT_PUBLIC_URL", "http://localhost:8000").rstrip("/")
    parsed = urlparse(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    request = Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": parsed.scheme or "http",
        "path": "/api/routines/events",
        "raw_path": b"/api/routines/events",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "server": (parsed.hostname or "localhost", port),
        "client": ("127.0.0.1", 0),
    })
    body = server.EventRoutineCreateRequest(
        name=name,
        instruction=action_prompt,
        source=source,
        event_type=event_type,
        when_to_run=when_to_run,
        filters=filters,
        repository=repository,
        channel_name=channel_name,
    )
    try:
        result = await server.create_event_routine_endpoint(
            body,
            request,
            {"user_id": active_user_id_var.get() or "default_user"},
        )
        routine = result["routine"]
        return (
            f"Event routine created: '{routine['name']}' "
            f"(ID: `{routine['id']}`).\n"
            f"- When to run: {when_to_run}\n"
            f"- Source: {source}\n"
            "Open Routines to review it and test with a safe event."
        )
    except HTTPException as exc:
        return f"Could not create event routine: {exc.detail}"


# Extended tools including A2UI and Generative UI (5 tools for test compatibility)
ALL_AGENT_TOOLS = [
    render_a2ui_canvas,
    render_generative_ui,
]

# Complete suite including Workspace Filesystem, Code Execution, Memory, and Proactive Routines
WORKSPACE_TOOLS = [
    workspace_files,
    run_workspace_code,
    ask_question,
    search_available_tools,
    manage_memory,
    manage_routine,
    create_event_routine,
]

# Complete suite of active agent tools sent to the model:
# (calculate and get_weather removed to reduce context overhead)
WORKSPACE_AGENT_TOOLS = [
    render_a2ui_canvas,
    render_generative_ui,
    *WORKSPACE_TOOLS,
]

@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    import math
    safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_dict.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
    try:
        result = eval(expression, {"__builtins__": None}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_time() -> str:
    """Get the current UTC time formatted nicely."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"Current UTC Time: {now}"


@tool
def get_weather(location: str) -> str:
    """Get weather information for a given city or location."""
    return f"Weather for {location}: Sunny, 22°C (72°F) with light breeze."


AGENT_TOOLS = [calculate, get_current_time, get_weather]
ALL_AGENT_TOOLS = [calculate, get_current_time, get_weather, render_a2ui_canvas, render_generative_ui]
