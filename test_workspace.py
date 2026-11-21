import pytest
import asyncio
from pathlib import Path
from chat_store import PostgresChatStore
from workspace import (
    get_workspace_dir,
    resolve_safe_path,
    write_file,
    read_file,
    edit_file,
    list_files,
    run_code,
    create_or_update_file,
    list_workspace_artifacts,
    snapshot_workspace_artifacts,
    diff_workspace_artifacts,
    WorkspacePollingWatcher,
    subscribe_workspace_changes,
    unsubscribe_workspace_changes,
)


@pytest.fixture
def temp_store(postgres_database):
    return PostgresChatStore(postgres_database)


def test_workspace_security_jail():
    thread_id = "test-thread-jail"
    ws = get_workspace_dir(thread_id)
    assert ws.exists()

    # Traversal attempts should raise ValueError
    with pytest.raises(ValueError):
        resolve_safe_path("../../outside.txt", thread_id)

    with pytest.raises(ValueError):
        resolve_safe_path("foo/../../../etc/passwd", thread_id)

    with pytest.raises(ValueError):
        resolve_safe_path("../test-thread-jail-evil/secret.txt", thread_id)


def test_workspace_file_lifecycle():
    thread_id = "test-thread-ops"

    # 1. Write file
    write_res = write_file("src/hello.py", "print('hello world')\nx = 10\n", thread_id=thread_id)
    assert write_res["status"] == "written"
    assert write_res["lines"] == 2

    # 2. List files
    listing = list_files("src", thread_id=thread_id)
    assert listing["status"] == "ok"
    assert any(item["name"] == "hello.py" for item in listing["items"])

    # 3. Read file with line range
    read_res = read_file("src/hello.py", start_line=1, end_line=1, thread_id=thread_id)
    assert read_res["status"] == "ok"
    assert "hello world" in read_res["content"]
    assert "10" not in read_res["content"]

    # 4. Edit file with exact target matching
    edit_res = edit_file("src/hello.py", "x = 10", "x = 42", thread_id=thread_id)
    assert edit_res["status"] == "edited"
    assert edit_res["occurrences_replaced"] == 1

    # Verify update
    full_read = read_file("src/hello.py", thread_id=thread_id)
    assert "x = 42" in full_read["content"]

    # Target not found error
    fail_edit = edit_file("src/hello.py", "nonexistent_code()", "anything", thread_id=thread_id)
    assert "error" in fail_edit


def test_agent_has_one_workspace_file_tool():
    import tools
    from policy_engine import classify_action_risk

    from tools import WORKSPACE_AGENT_TOOLS

    names = {agent_tool.name for agent_tool in WORKSPACE_AGENT_TOOLS}
    assert "workspace_files" in names
    assert "list_workspace_files" not in names
    assert "read_workspace_file" not in names
    assert "create_file" not in names
    assert "write_workspace_file" not in names
    assert "edit_workspace_file" not in names
    assert "create_artifact" not in names
    assert not hasattr(tools, "list_workspace_files")
    assert not hasattr(tools, "read_workspace_file")
    assert not hasattr(tools, "create_file")
    assert not hasattr(tools, "write_workspace_file")
    assert not hasattr(tools, "edit_workspace_file")
    assert not hasattr(tools, "create_artifact")
    assert classify_action_risk("workspace_files", {"action": "list"})[0] == "low"
    assert classify_action_risk("workspace_files", {"action": "read"})[0] == "low"
    assert classify_action_risk("workspace_files", {"action": "create"})[0] == "medium"


def test_single_workspace_tool_lists_reads_and_creates(tmp_path, monkeypatch):
    import json
    import tools
    import workspace

    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", tmp_path / "workspaces")
    created = json.loads(tools.workspace_files.invoke({
        "action": "create",
        "file_path": "notes.txt",
        "content": "first\nsecond",
    }))
    assert created["status"] == "created"

    read = json.loads(tools.workspace_files.invoke({
        "action": "read",
        "file_path": "notes.txt",
        "start_line": 2,
        "end_line": 2,
    }))
    assert read["status"] == "ok"
    assert read["raw_content"] == "second"

    listing = json.loads(tools.workspace_files.invoke({"action": "list"}))
    assert listing["status"] == "ok"
    assert any(item["name"] == "notes.txt" for item in listing["items"])


def test_unified_file_creation_and_explicit_change_events(tmp_path, monkeypatch):
    import base64
    import uuid
    import workspace

    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", tmp_path / "workspaces")
    thread_id = f"test-thread-unified-files-{uuid.uuid4().hex}"
    events = subscribe_workspace_changes(thread_id)
    try:
        created = create_or_update_file("report.md", "# First", thread_id=thread_id)
        assert created["status"] == "created"
        assert created["file_path"] == "report.md"
        assert events.get(timeout=1)["operation"] == "created"

        updated = create_or_update_file(
            "report.md",
            "First",
            operation="edit",
            find_text="First",
            replacement="Updated",
            thread_id=thread_id,
        )
        assert updated["status"] == "updated"
        assert updated["occurrences_replaced"] == 1
        assert events.get(timeout=1)["operation"] == "modified"
        assert read_file("report.md", thread_id=thread_id)["raw_content"] == "# Updated"

        appended = create_or_update_file(
            "report.md",
            "\nDone",
            operation="append",
            thread_id=thread_id,
        )
        assert appended["status"] == "updated"
        assert read_file("report.md", thread_id=thread_id)["raw_content"] == "# Updated\nDone"

        replaced = create_or_update_file(
            "report.md",
            "# Replaced",
            operation="replace",
            thread_id=thread_id,
        )
        assert replaced["status"] == "updated"
        assert read_file("report.md", thread_id=thread_id)["raw_content"] == "# Replaced"

        binary = create_or_update_file(
            "deck.pdf",
            base64.b64encode(b"%PDF-1.7").decode("ascii"),
            content_encoding="base64",
            thread_id=thread_id,
        )
        assert binary["status"] == "created"
        assert resolve_safe_path("deck.pdf", thread_id).read_bytes() == b"%PDF-1.7"

        duplicate = create_or_update_file("report.md", "replacement", thread_id=thread_id)
        assert "error" in duplicate
        invalid = create_or_update_file(
            "report.md",
            "replacement",
            operation="not-an-operation",
            thread_id=thread_id,
        )
        assert "error" in invalid
    finally:
        unsubscribe_workspace_changes(thread_id, events)


def test_workspace_artifact_discovery_tracks_moves_and_deletions(tmp_path, monkeypatch):
    import workspace

    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", tmp_path / "workspaces")
    thread_id = "artifact-discovery"
    root = get_workspace_dir(thread_id)
    source = root / "report.pdf"
    source.write_bytes(b"pdf-data")
    previous = snapshot_workspace_artifacts(thread_id)
    assert previous["report.pdf"]["artifact_type"] == "pdf"

    renamed = root / "final.pdf"
    source.rename(renamed)
    current = snapshot_workspace_artifacts(thread_id)
    move_changes = diff_workspace_artifacts(thread_id, previous, current)
    assert len(move_changes) == 1
    assert move_changes[0]["operation"] == "moved"
    assert move_changes[0]["old_path"] == "report.pdf"
    assert move_changes[0]["file_path"] == "final.pdf"

    renamed.unlink()
    deleted = snapshot_workspace_artifacts(thread_id)
    delete_changes = diff_workspace_artifacts(thread_id, current, deleted)
    assert len(delete_changes) == 1
    assert delete_changes[0]["operation"] == "deleted"
    assert list_workspace_artifacts(thread_id) == []


def test_workspace_project_discovery_groups_sources_and_ignores_dependencies(tmp_path, monkeypatch):
    import workspace

    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", tmp_path / "workspaces")
    root = get_workspace_dir("project-discovery")
    (root / "package.json").write_text('{"name":"demo"}', encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "main.tsx").write_text("export default function App() {}", encoding="utf-8")
    (root / "src" / "App.tsx").write_text("export function App() {}", encoding="utf-8")
    (root / "node_modules" / "hidden").mkdir(parents=True)
    (root / "node_modules" / "hidden" / "index.js").write_text("internal", encoding="utf-8")
    (root / "scratch.tmp").write_text("temporary", encoding="utf-8")

    artifacts = list_workspace_artifacts("project-discovery")
    assert len(artifacts) == 1
    assert artifacts[0]["artifact_type"] == "application"
    assert artifacts[0]["file_path"] == "src/main.tsx"
    assert artifacts[0]["entry_point"] == "src/main.tsx"
    assert all("node_modules" not in item[0] for item in artifacts[0]["source_signature"])


def test_workspace_watcher_emits_only_stable_changes(tmp_path, monkeypatch):
    import workspace

    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", tmp_path / "workspaces")
    thread_id = "stable-watcher"
    root = get_workspace_dir(thread_id)
    watcher = WorkspacePollingWatcher(stable_scans=2)
    watcher._snapshots[thread_id] = {}
    events = subscribe_workspace_changes(thread_id)

    try:
        target = root / "notes.md"
        target.write_text("first", encoding="utf-8")
        watcher._scan()
        assert events.empty()
        watcher._scan()
        assert events.get(timeout=1)["operation"] == "created"

        target.write_text("updated content", encoding="utf-8")
        watcher._scan()
        assert events.empty()
        watcher._scan()
        assert events.get(timeout=1)["operation"] == "modified"

        target.rename(root / "renamed.md")
        watcher._scan()
        assert events.empty()
        watcher._scan()
        moved = events.get(timeout=1)
        assert moved["operation"] == "moved"
        assert moved["old_path"] == "notes.md"
        assert moved["file_path"] == "renamed.md"

        (root / "renamed.md").unlink()
        watcher._scan()
        assert events.empty()
        watcher._scan()
        assert events.get(timeout=1)["operation"] == "deleted"
    finally:
        unsubscribe_workspace_changes(thread_id, events)


@pytest.mark.asyncio
async def test_workspace_code_execution():
    thread_id = "test-thread-exec"
    write_file("calc.py", "a = 20\nb = 22\nprint(f'RESULT={a+b}')\n", thread_id=thread_id)

    res = await run_code("calc.py", thread_id=thread_id)
    assert res["status"] == "completed"
    assert res["exit_code"] == 0
    assert "RESULT=42" in res["stdout"]


@pytest.mark.asyncio
async def test_workspace_code_execution_timeout():
    thread_id = "test-thread-timeout"
    write_file("hang.py", "import time\ntime.sleep(5)\nprint('Done')\n", thread_id=thread_id)

    res = await run_code("hang.py", timeout_seconds=1, thread_id=thread_id)
    assert res["status"] == "timeout"
    assert res["exit_code"] == -1
    assert "timed out" in res["error"]


def test_artifacts_persistence(temp_store):
    thread = temp_store.create_thread(title="Artifacts Test")
    tid = thread["thread_id"]

    # Add artifact
    art = temp_store.add_artifact(
        thread_id=tid,
        title="Architecture Plan",
        artifact_type="markdown_plan",
        content="# Plan\n- Step 1\n- Step 2",
        language="markdown",
    )
    assert art["artifact_id"].startswith("artifact-")
    assert art["title"] == "Architecture Plan"

    # List artifacts
    all_art = temp_store.list_artifacts(tid)
    assert len(all_art) == 1
    assert all_art[0]["title"] == "Architecture Plan"

    # Get single artifact
    single = temp_store.get_artifact(art["artifact_id"])
    assert single is not None
    assert single["content"] == art["content"]

    # Thread includes artifacts
    thread_details = temp_store.get_thread(tid)
    assert "artifacts" in thread_details
    assert len(thread_details["artifacts"]) == 1

    # Delete artifact
    deleted = temp_store.delete_artifact(art["artifact_id"])
    assert deleted is True
    assert temp_store.get_artifact(art["artifact_id"]) is None


def test_server_workspace_endpoints(tmp_path, monkeypatch, postgres_database):
    from fastapi.testclient import TestClient
    import server
    import workspace

    original_store = server.chat_store
    server.chat_store = PostgresChatStore(postgres_database)
    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", tmp_path / "workspaces")
    client = TestClient(server.app)

    try:
        # 0. Unauthenticated access must be rejected with 401
        unauth_res = client.post("/threads", json={"title": "Blocked"})
        assert unauth_res.status_code == 401

        # Sign up & obtain session cookie
        signup_res = client.post(
            "/auth/signup",
            json={
                "display_name": "Workspace Tester",
                "email": "ws-tester@example.com",
                "password": "Password123!",
            },
        )
        assert signup_res.status_code == 200

        # 1. Create a thread
        t_res = client.post("/threads", json={"title": "Workspace API Test"})
        assert t_res.status_code == 200
        tid = t_res.json()["thread_id"]

        # 2. Write a workspace file via workspace module
        write_file("app.py", "val = 100\nprint(f'SERVER_OUTPUT={val*2}')\n", thread_id=tid)

        # 3. List workspace files via endpoint
        files_res = client.get(f"/threads/{tid}/workspace/files")
        assert files_res.status_code == 200
        files_data = files_res.json()
        assert files_data["status"] == "ok"
        assert any(item["name"] == "app.py" for item in files_data["items"])

        # Workspace files are automatically surfaced as artifact references.
        discovered = client.get(f"/threads/{tid}/artifacts")
        assert discovered.status_code == 200
        assert any(
            item.get("file_path") == "app.py" and item.get("source") == "workspace_file"
            for item in discovered.json()
        )

        # 4. Read workspace file via endpoint
        read_res = client.get(f"/threads/{tid}/workspace/file?path=app.py")
        assert read_res.status_code == 200
        assert "SERVER_OUTPUT" in read_res.json()["content"]

        # 5. Run code via workspace run endpoint
        run_res = client.post(f"/threads/{tid}/workspace/run", json={"command_or_file": "app.py"})
        assert run_res.status_code == 200
        run_data = run_res.json()
        assert run_data["status"] == "completed"
        assert "SERVER_OUTPUT=200" in run_data["stdout"]

        # 6. Create artifact via endpoint
        art_post = client.post(
            f"/threads/{tid}/artifacts",
            json={
                "title": "API Deliverable",
                "artifact_type": "code",
                "content": "def hello(): pass",
                "language": "python",
            },
        )
        assert art_post.status_code == 200
        art_id = art_post.json()["artifact_id"]

        # 7. List artifacts via endpoint
        art_list = client.get(f"/threads/{tid}/artifacts")
        assert art_list.status_code == 200
        assert len(art_list.json()) >= 1

        # 8. Get artifact by ID
        single_art = client.get(f"/threads/{tid}/artifacts/{art_id}")
        assert single_art.status_code == 200
        assert single_art.json()["title"] == "API Deliverable"

        # 9. Delete artifact
        del_res = client.delete(f"/threads/{tid}/artifacts/{art_id}")
        assert del_res.status_code == 200

        # 10. Workspace preview endpoint for text/markdown
        write_file("essay.md", "# AI Essay\n\nDeep learning and NLP overview.", thread_id=tid)
        prev_res = client.get(f"/threads/{tid}/workspace/preview?path=essay.md")
        assert prev_res.status_code == 200
        prev_json = prev_res.json()
        assert prev_json["type"] == "text"
        assert "AI Essay" in prev_json["content"]

        # HTML previews serve the real workspace entry file and its relative assets.
        write_file(
            "site/index.html",
            '<!doctype html><link rel="stylesheet" href="style.css"><h1>Workspace site</h1>',
            thread_id=tid,
        )
        write_file("site/style.css", "h1 { color: red; }", thread_id=tid)
        html_preview = client.get(f"/threads/{tid}/workspace/preview?path=site/index.html")
        assert html_preview.status_code == 200
        assert html_preview.json()["type"] == "html"
        html_page = client.get(f"/threads/{tid}/workspace/serve/site/index.html")
        assert html_page.status_code == 200
        assert "Workspace site" in html_page.text
        assert "sandbox" in html_page.headers.get("content-security-policy", "")
        css_asset = client.get(f"/threads/{tid}/workspace/serve/site/style.css")
        assert css_asset.status_code == 200
        assert "color: red" in css_asset.text
        discovered_after_html = client.get(f"/threads/{tid}/artifacts").json()
        assert any(item.get("file_path") == "site/index.html" for item in discovered_after_html)
        assert not any(item.get("file_path") == "site/style.css" for item in discovered_after_html)

        # 11. Raw file serving endpoint
        raw_res = client.get(f"/threads/{tid}/workspace/raw?path=essay.md")
        assert raw_res.status_code == 200
        assert "AI Essay" in raw_res.text

        # 12. File download endpoint
        dl_res = client.get(f"/threads/{tid}/workspace/download?path=essay.md")
        assert dl_res.status_code == 200
        assert "attachment" in dl_res.headers.get("content-disposition", "")
    finally:
        server.chat_store = original_store


def test_multi_format_previews():
    tid = "test-preview-formats"
    from workspace import get_file_preview_data

    # Markdown / text
    write_file("doc.md", "# Sample\nContent line", thread_id=tid)
    res_md = get_file_preview_data("doc.md", thread_id=tid)
    assert res_md["type"] == "text"
    assert res_md["extension"] == "md"

    # CSV
    write_file("data.csv", "name,age,city\nAlice,30,NYC\nBob,25,LA", thread_id=tid)
    res_csv = get_file_preview_data("data.csv", thread_id=tid)
    assert res_csv["type"] == "csv"
    assert len(res_csv["rows"]) == 3
    assert res_csv["rows"][0] == ["name", "age", "city"]

    # Excel
    import openpyxl
    target_xlsx = resolve_safe_path("sheet.xlsx", thread_id=tid)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Metric", "Value"])
    ws.append(["Users", 500])
    wb.save(target_xlsx)

    res_xlsx = get_file_preview_data("sheet.xlsx", thread_id=tid)
    assert res_xlsx["type"] == "excel"
    assert res_xlsx["sheets"][0]["sheet_name"] == "Summary"
    assert len(res_xlsx["sheets"][0]["rows"]) == 2

    # DOCX
    from docx import Document
    target_docx = resolve_safe_path("doc.docx", thread_id=tid)
    doc = Document()
    doc.add_heading("Title Docx", 0)
    doc.add_paragraph("Paragraph content")
    doc.save(target_docx)

    res_docx = get_file_preview_data("doc.docx", thread_id=tid)
    assert res_docx["type"] == "docx"
    assert "Title Docx" in res_docx["html"]

    # PPTX
    from pptx import Presentation
    target_pptx = resolve_safe_path("deck.pptx", thread_id=tid)
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Presentation Slide 1"
    prs.save(target_pptx)

    res_pptx = get_file_preview_data("deck.pptx", thread_id=tid)
    assert res_pptx["type"] == "pptx"
    assert res_pptx["total_slides"] == 1
    assert res_pptx["slides"][0]["title"] == "Presentation Slide 1"

