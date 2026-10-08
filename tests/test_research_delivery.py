"""Researcher-facing hook contracts: real files, failure preservation and safe paths."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote
import re

import pytest

from academic_helper import research_delivery as delivery

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research_hooks.py"


def task(root, name="20261008-文獻搜尋", purpose="文獻搜尋", status="ready", primary="研究摘要.md", timestamp="2026-10-08T01:00:00+00:00", limitations=None):
    folder = root / "outputs/成果" / name
    folder.mkdir(parents=True)
    if primary:
        (folder / primary).write_text("有來源的研究摘要", encoding="utf-8")
    data = {"schema_version": 1, "purpose": purpose, "status": status,
            "updated_at": timestamp, "primary": primary,
            "artifacts": [primary] if primary else [], "limitations": limitations or []}
    (folder / "delivery.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return folder


def assert_links_exist(path):
    for target in re.findall(r'\]\(<([^>]+)>\)', path.read_text(encoding="utf-8")):
        assert (path.parent / unquote(target)).is_file()


def test_index_entry_and_portable_links(tmp_path):
    task(tmp_path, primary="研究 摘要.md")
    result = delivery.refresh(str(tmp_path))
    assert result["changed"]
    index = tmp_path / "outputs/成果/README.md"
    entry = tmp_path / "研究入口.md"
    assert "2026-10-08T09:00:00+08:00" in index.read_text()
    assert_links_exist(index)
    assert_links_exist(entry)
    assert not delivery.refresh(str(tmp_path))["changed"]


def test_manual_content_preserved_and_no_duplicate_sections(tmp_path):
    task(tmp_path)
    entry = tmp_path / "研究入口.md"
    entry.write_text("# 我的題目\n請保留人工決定。\n", encoding="utf-8")
    index = tmp_path / "outputs/成果/README.md"
    index.write_text("我的人工說明\n", encoding="utf-8")
    delivery.refresh(str(tmp_path))
    entry.write_text(entry.read_text() + "後面的人工筆記\n", encoding="utf-8")
    delivery.refresh(str(tmp_path))
    text = entry.read_text()
    assert text.startswith("# 我的題目\n請保留人工決定。\n")
    assert text.endswith("後面的人工筆記\n") and text.count(delivery.START) == 1
    assert index.read_text().startswith("我的人工說明\n")


def test_failed_new_run_retains_old_delivery_and_shows_failure(tmp_path):
    old = task(tmp_path)
    task(tmp_path, name="20261009-失敗", status="failed", primary=None,
         timestamp="2026-10-09T01:00:00+00:00", limitations=["來源受限"])
    delivery.refresh(str(tmp_path))
    index = tmp_path / "outputs/成果/README.md"
    text = index.read_text()
    assert "保留前次可用成果" in text and "來源受限" in text
    assert old.name in text
    assert_links_exist(index)


def test_partial_is_visible_and_deleted_primary_falls_back(tmp_path):
    task(tmp_path)
    newer = task(tmp_path, name="20261009-部分", status="partial", timestamp="2026-10-09T01:00:00+00:00", limitations=["全文待補"])
    delivery.refresh(str(tmp_path))
    index = tmp_path / "outputs/成果/README.md"
    assert "部分完成；全文待補" in index.read_text()
    (newer / "研究摘要.md").unlink()
    delivery.refresh(str(tmp_path))
    assert "保留前次可用成果" in index.read_text()
    assert_links_exist(index)


def test_deleted_all_primaries_never_links_missing_file(tmp_path):
    folder = task(tmp_path)
    (folder / "研究摘要.md").unlink()
    delivery.refresh(str(tmp_path))
    index = tmp_path / "outputs/成果/README.md"
    assert "尚無可用成果" in index.read_text()
    assert_links_exist(index)


def test_missing_secondary_reported(tmp_path):
    folder = task(tmp_path)
    p = folder / "delivery.json"
    data = json.loads(p.read_text()); data["artifacts"].append("失效.pdf")
    p.write_text(json.dumps(data), encoding="utf-8")
    delivery.refresh(str(tmp_path))
    assert "部分附屬檔案失效" in (tmp_path / "outputs/成果/README.md").read_text()


def test_malformed_managed_markers_do_not_change_either_entry(tmp_path):
    task(tmp_path)
    entry = tmp_path / "研究入口.md"
    entry.write_text("人工段落\n" + delivery.START, encoding="utf-8")
    before = entry.read_bytes()
    with pytest.raises(delivery.DeliveryError): delivery.refresh(str(tmp_path))
    assert entry.read_bytes() == before and not (tmp_path / "outputs/成果/README.md").exists()


@pytest.mark.parametrize("primary", ["../secret.md", "/tmp/secret.md", "C:/secret.md", "folder\\secret.md"])
def test_traversal_record_is_ignored(tmp_path, primary):
    folder = task(tmp_path)
    p = folder / "delivery.json"; data = json.loads(p.read_text())
    data["primary"] = primary; data["artifacts"] = []
    p.write_text(json.dumps(data), encoding="utf-8")
    result = delivery.refresh(str(tmp_path))
    assert not result["changed"] and result["skipped"] == 1
    assert not (tmp_path / "研究入口.md").exists()


def test_symlink_escape_and_entry_link_preserved(tmp_path):
    outside = tmp_path / "outside"; outside.mkdir()
    root = tmp_path / "project"; root.mkdir()
    (root / "outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(delivery.DeliveryError): delivery.refresh(str(root))
    (root / "outputs").unlink()
    folder = task(root)
    secret = outside / "private.md"; secret.write_text("private")
    (folder / "研究摘要.md").unlink(); (folder / "研究摘要.md").symlink_to(secret)
    assert not delivery.refresh(str(root))["changed"]
    (folder / "研究摘要.md").unlink(); (folder / "研究摘要.md").write_text("safe")
    (root / "研究入口.md").symlink_to(secret)
    with pytest.raises(delivery.DeliveryError): delivery.refresh(str(root))
    assert secret.read_text() == "private"


def test_record_validates_files_and_refreshes_without_hooks(tmp_path):
    folder = task(tmp_path); (folder / "delivery.json").unlink()
    result = delivery.record(str(tmp_path), folder.name, "文獻搜尋", "partial", "研究摘要.md", [], ["全文待補"])
    assert result["changed"]
    with pytest.raises(delivery.DeliveryError):
        delivery.record(str(tmp_path), folder.name, "文獻搜尋", "ready", "不存在.md", [], [])
    assert json.loads((folder / "delivery.json").read_text())["status"] == "partial"


def test_no_empty_project_created(tmp_path):
    assert delivery.refresh(str(tmp_path))["changed"] is False
    assert list(tmp_path.iterdir()) == []
    assert delivery.handle_hook({"cwd": str(tmp_path), "hook_event_name": "SessionStart"}) == {}


def test_session_start_only_points_to_context_and_never_echoes_private_text(tmp_path):
    p = tmp_path / "研究入口.md"; p.write_text("SECRET_API_KEY\n我的題目", encoding="utf-8")
    result = delivery.handle_hook({"cwd": str(tmp_path), "hook_event_name": "SessionStart"})
    assert result["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "研究入口.md" in json.dumps(result, ensure_ascii=False)
    assert "SECRET_API_KEY" not in json.dumps(result)
    assert p.read_text().startswith("SECRET_API_KEY")


@pytest.mark.parametrize("response", [{"isError": True}, {"exit_code": 1}, {"interrupted": True}, {"error": "SECRET"}])
def test_failed_tool_hook_does_not_write(tmp_path, response):
    task(tmp_path)
    result = delivery.handle_hook({"cwd": str(tmp_path), "hook_event_name": "PostToolUse", "tool_response": response})
    assert result == {} and not (tmp_path / "研究入口.md").exists()


def test_posttool_for_both_hosts_ignores_extra_fields(tmp_path):
    task(tmp_path)
    common = {"cwd": str(tmp_path), "hook_event_name": "PostToolUse", "tool_response": {"exit_code": 0},
              "tool_name": "apply_patch", "tool_input": {}, "turn_id": "codex-turn", "agent_id": "agent"}
    result = delivery.handle_hook(common)
    assert result["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert delivery.handle_hook(common) == {}


def test_lock_contention_is_nonblocking_and_stale_lock_recovered(tmp_path):
    folder = task(tmp_path).parent
    lock = folder / ".academic-helper-index.lock"; lock.write_text("busy")
    with pytest.raises(delivery.DeliveryError): delivery.refresh(str(tmp_path))
    os.utime(lock, (0, 0))
    assert delivery.refresh(str(tmp_path))["changed"] and not lock.exists()


@pytest.mark.parametrize("payload", ["not json PRIVATE_SECRET", '{"hook_event_name":"PostToolUse"}', '[]'])
def test_cli_hook_failures_are_nonblocking_and_sanitized(tmp_path, payload):
    result = subprocess.run([sys.executable, str(SCRIPT), "hook"], input=payload, text=True,
                            cwd=tmp_path, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert "PRIVATE_SECRET" not in result.stdout + result.stderr
    assert "systemMessage" in data


def test_hook_manifests_select_only_intended_events_and_windows_syntax():
    claude = json.loads((ROOT / "hooks/hooks.json").read_text())
    codex = json.loads((ROOT / "hooks/codex-hooks.json").read_text())
    assert set(claude["hooks"]) == set(codex["hooks"]) == {"SessionStart", "PostToolUse"}
    for event in codex["hooks"].values():
        handler = event[0]["hooks"][0]
        assert "%PLUGIN_ROOT%" in handler["commandWindows"]
        assert "${PLUGIN_ROOT}" in handler["command"]
        assert handler["type"] == "command" and handler["timeout"] == 10
    plugin = json.loads((ROOT / ".codex-plugin/plugin.json").read_text())
    assert plugin["hooks"] == "./hooks/codex-hooks.json"
    assert "${PLUGIN_ROOT}" in plugin["mcpServers"]["academic-helper"]["args"]


def test_cli_outputs_utf8_even_with_ascii_locale(tmp_path):
    (tmp_path / "研究入口.md").write_text("我的研究", encoding="utf-8")
    payload = {"cwd": str(tmp_path), "hook_event_name": "SessionStart"}
    result = subprocess.run([sys.executable, str(SCRIPT), "hook"], input=json.dumps(payload),
                            encoding="utf-8", cwd=tmp_path, capture_output=True,
                            env=dict(os.environ, PYTHONIOENCODING="ascii"), check=True)
    assert "研究入口.md" in json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
