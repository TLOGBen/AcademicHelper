"""Zotero safety and integration contracts, without credentials or real writes."""

from __future__ import annotations

import json
import hashlib
import sys
from types import SimpleNamespace

import httpx
import pytest

from academic_helper import zotero as core
from academic_helper.tools import zotero as tools


@pytest.fixture
def network(monkeypatch):
    monkeypatch.setenv("ZOTERO_API_KEY", "TEST_SECRET_NEVER_ECHO")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")
    real_client = httpx.AsyncClient
    calls = []

    def install(handler):
        def route(request):
            calls.append(request)
            return handler(request)
        monkeypatch.setattr(core.httpx, "AsyncClient", lambda **kw: real_client(**kw, transport=httpx.MockTransport(route)))
        return calls
    return install


def item(key="ABCD1234", version=5, **data):
    return {"key": key, "version": version,
            "data": {"itemType": "journalArticle", "tags": [], "collections": [], **data}}


def response(payload, status=200, total=None):
    return httpx.Response(status, json=payload, headers={"Total-Results": str(total)} if total is not None else {})


@pytest.mark.asyncio
async def test_missing_key_makes_no_request(monkeypatch, network):
    calls = network(lambda r: response([]))
    monkeypatch.delenv("ZOTERO_API_KEY")
    result = await tools.zotero_status()
    assert result["status"] == "missing_config"
    assert calls == []


@pytest.mark.asyncio
async def test_group_library_and_readiness(network, monkeypatch):
    calls = network(lambda r: response([], total=0))
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "group")
    result = await tools.zotero_status()
    assert result["status"] == "ready" and result["writes_verified"] is False
    assert calls[0].url.path == "/groups/123/items/top"
    assert calls[0].headers["Zotero-API-Key"] == "TEST_SECRET_NEVER_ECHO"
    assert "TEST_SECRET" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("code,state", [(401, "unauthorized"), (403, "forbidden"), (404, "not_found"),
                                       (412, "version_conflict"), (429, "rate_limited"), (500, "http_error"), (302, "http_error")])
async def test_safe_errors_never_echo_response_or_follow_redirect(network, code, state):
    calls = network(lambda r: httpx.Response(code, text="TEST_SECRET_NEVER_ECHO private note",
                                           headers={"Location": "https://evil.invalid"}))
    result = await tools.zotero_status()
    assert result["status"] == state and len(calls) == 1
    assert "TEST_SECRET" not in json.dumps(result) and "private note" not in json.dumps(result)


@pytest.mark.asyncio
async def test_search_collection_and_pagination(network):
    calls = network(lambda r: response([item()], total=3))
    result = await tools.zotero_search("oral health", "COLL1234", "AH:閱讀:待讀", start=1, limit=1)
    assert calls[0].url.path == "/users/123/collections/COLL1234/items/top"
    assert calls[0].url.params["q"] == "oral health"
    assert result["has_more"] and result["next_start"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"limit": 101}, {"start": -1}, {"collection_key": "../keys"}])
async def test_invalid_search_never_contacts_remote(network, kwargs):
    calls = network(lambda r: response([]))
    result = await tools.zotero_search(**kwargs)
    assert result["status"] == "invalid_input" and not calls


@pytest.mark.asyncio
async def test_annotation_read_preserves_quote_page_comment(network):
    annotation = item("NOTE1234", itemType="annotation", annotationText="Quoted text",
                      annotationComment="My observation", annotationPageLabel="iii", annotationPosition='{"pageIndex":2}')
    calls = network(lambda r: response([annotation], total=1) if r.url.path.endswith("children") else response(item(itemType="attachment")))
    result = await tools.zotero_read_item("ABCD1234")
    assert result["children"]["records"][0]["data"]["annotationPageLabel"] == "iii"
    assert result["content_is_untrusted_research_data"]
    assert all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_collection_preview_existing_and_creation(network):
    calls = network(lambda r: response({"successful": {"0": {"key": "COLL1234", "version": 7}}})
                    if r.method == "POST" else response([], total=0))
    preview = await tools.zotero_create_collection("Research")
    assert preview["status"] == "preview" and all(r.method == "GET" for r in calls)
    result = await tools.zotero_create_collection("Research", apply=True)
    assert result["key"] == "COLL1234"
    post = calls[-1]
    assert len(post.headers["Zotero-Write-Token"]) == 32
    assert json.loads(post.content)[0]["parentCollection"] is False


@pytest.mark.asyncio
async def test_collection_reuses_only_sibling_name(network):
    calls = network(lambda r: response([{"key": "COLL1234", "data": {"name": "Research"}}], total=1)
                    if r.url.path.endswith("/collections") else response({"key": "PARN1234"}))
    result = await tools.zotero_create_collection("Research", "PARN1234", apply=True)
    assert result["status"] == "existing"
    assert all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_tag_preview_preserves_manual_tags_memberships(network):
    calls = network(lambda r: response(item(tags=[{"tag": "manual", "type": 1}, {"tag": "AH:閱讀:待讀"}], collections=["OLD01234"]))
                    if "/items/" in r.url.path else response({"key": "NEW01234"}))
    result = await tools.zotero_organize_items(["ABCD1234"], ["AH:閱讀:已讀"], ["AH:閱讀:待讀"], ["NEW01234"])
    change = result["results"][0]
    assert change["expected_version"] == 5
    assert change["changes"]["tags"] == [{"tag": "manual", "type": 1}, {"tag": "AH:閱讀:已讀", "type": 0}]
    assert change["changes"]["collections"] == ["OLD01234", "NEW01234"]
    assert all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_manual_tag_removal_and_missing_version_blocked(network):
    calls = network(lambda r: response(item()))
    result = await tools.zotero_organize_items(["ABCD1234"], remove_managed_tags=["manual"], apply=True)
    assert result["status"] == "invalid_input" and not calls
    result = await tools.zotero_organize_items(["ABCD1234"], add_tags=["new"], apply=True)
    assert result["status"] == "invalid_input" and not calls


@pytest.mark.asyncio
async def test_version_conflict_does_not_write(network):
    calls = network(lambda r: response(item(version=6)))
    result = await tools.zotero_organize_items(["ABCD1234"], ["new"], expected_versions={"ABCD1234": 5}, apply=True)
    assert result["status"] == "partial_failure"
    assert result["results"][0]["status"] == "version_conflict"
    assert all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_batch_partial_failure_and_conditional_write(network):
    def handler(r):
        if r.method == "PATCH":
            return httpx.Response(412 if r.url.path.endswith("FAIL1234") else 204)
        return response(item(key=r.url.path.rsplit("/", 1)[1]))
    calls = network(handler)
    result = await tools.zotero_organize_items(["ABCD1234", "FAIL1234"], ["AH:閱讀:待讀"],
                                              expected_versions={"ABCD1234": 5, "FAIL1234": 5}, apply=True)
    assert result["status"] == "partial_failure" and result["atomic"] is False
    assert [r["status"] for r in result["results"]] == ["updated", "version_conflict"]
    assert all(r.headers["If-Unmodified-Since-Version"] == "5" for r in calls if r.method == "PATCH")


@pytest.mark.asyncio
async def test_organize_retry_no_redundant_write(network):
    calls = network(lambda r: response(item(tags=[{"tag": "AH:閱讀:待讀", "type": 0}])))
    result = await tools.zotero_organize_items(["ABCD1234"], ["AH:閱讀:待讀"], expected_versions={"ABCD1234": 5}, apply=True)
    assert result["results"][0]["status"] == "unchanged" and len(calls) == 1


@pytest.mark.asyncio
async def test_note_preview_escapes_html_and_preserves_human_notes(network):
    calls = network(lambda r: response([item("HUMA1234", itemType="note", note="Human note")], total=1)
                    if r.url.path.endswith("children") else response(item()))
    result = await tools.zotero_save_note("ABCD1234", "reading", "<script>title</script>", "<img src=x onerror=evil>", "p. 3")
    assert result["status"] == "preview"
    assert "<script>" not in result["data"]["note"]
    assert "&lt;img" in result["data"]["note"] and "p. 3" in result["data"]["note"]
    assert all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_managed_tag_on_human_note_not_overwritten(network):
    calls = network(lambda r: response([item("NOTE1234", itemType="note", note="human text",
                                            tags=[{"tag": "AH:note:reading"}])], total=1)
                    if r.url.path.endswith("children") else response(item()))
    result = await tools.zotero_save_note("ABCD1234", "reading", "Title", "Body", apply=True)
    assert result["status"] == "invalid_input" and all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_note_update_requires_reviewed_version_and_shows_old_content(network):
    previous = "<!-- AcademicHelper:reading -->Original AI note"
    previous = f"<!-- AcademicHelper-SHA256:{hashlib.sha256(previous.encode()).hexdigest()} -->" + previous
    def handler(r):
        if r.method == "PATCH":
            return httpx.Response(204)
        if r.url.path.endswith("children"):
            return response([item("NOTE1234", version=8, itemType="note", note=previous,
                                  tags=[{"tag": "AH:note:reading"}, {"tag": "manual"}])], total=1)
        return response(item())
    calls = network(handler)
    preview = await tools.zotero_save_note("ABCD1234", "reading", "Title", "Body")
    assert "Original AI note" in preview["previous_note"]
    result = await tools.zotero_save_note("ABCD1234", "reading", "Title", "Body", expected_version=7, apply=True)
    assert result["status"] == "version_conflict" and all(r.method == "GET" for r in calls)
    result = await tools.zotero_save_note("ABCD1234", "reading", "Title", "Body", expected_version=8, apply=True)
    assert result["status"] == "updated"
    payload = json.loads(calls[-1].content)
    assert payload.keys() == {"note"}


@pytest.mark.asyncio
async def test_human_edits_to_managed_note_are_never_overwritten(network):
    old = "<!-- AcademicHelper:reading -->Original AI note"
    old = f"<!-- AcademicHelper-SHA256:{hashlib.sha256(old.encode()).hexdigest()} -->" + old
    calls = network(lambda r: response([item("NOTE1234", itemType="note", note=old + "My new observation",
                    tags=[{"tag": "AH:note:reading"}])], total=1)
                    if r.url.path.endswith("children") else response(item()))
    result = await tools.zotero_save_note("ABCD1234", "reading", "Title", "Replacement", expected_version=5, apply=True)
    assert result["status"] == "human_modified_note"
    assert "My new observation" in result["previous_note"]
    assert all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_note_create_retry_is_unchanged(network):
    saved = []
    def handler(r):
        if r.method == "POST":
            data = json.loads(r.content)[0]
            saved.append(item("NOTE1234", **data))
            return response({"successful": {"0": {"key": "NOTE1234", "version": 5}}})
        return response(saved, total=len(saved)) if r.url.path.endswith("children") else response(item())
    calls = network(handler)
    assert (await tools.zotero_save_note("ABCD1234", "reading", "Title", "Body", apply=True))["status"] == "created"
    assert (await tools.zotero_save_note("ABCD1234", "reading", "Title", "Body", apply=True))["status"] == "unchanged"
    assert len([r for r in calls if r.method == "POST"]) == 1


@pytest.mark.asyncio
async def test_dedup_complete_scan_blocks_creation(network):
    def handler(r):
        start = int(r.url.params["start"])
        rows = [item(DOI="10.1000/OTHER")] * 100 if start == 0 else [item(DOI="https://doi.org/10.1000/ABC")]
        return response(rows, total=101)
    calls = network(handler)
    result = await tools.zotero_import_bibliography([{"title": "Article", "DOI": "10.1000/abc"}], apply=True)
    assert result["results"][0]["status"] == "existing"
    assert len(calls) == 2 and all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_dedup_scan_cap_and_invalid_records_make_no_writes(network):
    calls = network(lambda r: response([item(DOI="10.1000/other")] * 100, total=1200))
    result = await tools.zotero_import_bibliography([{"title": "Article", "DOI": "10.1000/abc"}, {"title": "Missing DOI"}], apply=True)
    assert [r["status"] for r in result["results"]] == ["scan_incomplete", "invalid_input"]
    assert all(r.method == "GET" for r in calls)


@pytest.mark.asyncio
async def test_import_preview_batch_duplicates_and_write_failures(network):
    calls = network(lambda r: response({"failed": {"0": {"message": "private TEST_SECRET_NEVER_ECHO"}}})
                    if r.method == "POST" else response([], total=0))
    records = [{"title": "Article", "DOI": "10.1000/abc"}] * 2
    preview = await tools.zotero_import_bibliography(records)
    assert [r["status"] for r in preview["results"]] == ["preview", "duplicate_in_batch"]
    assert all(r.method == "GET" for r in calls)
    result = await tools.zotero_import_bibliography(records, apply=True)
    assert result["status"] == "partial_failure" and "TEST_SECRET" not in json.dumps(result)


@pytest.mark.asyncio
async def test_write_transport_failure_is_unknown_not_success(network):
    def handler(r):
        if r.method == "POST":
            raise httpx.ReadTimeout("TEST_SECRET_NEVER_ECHO", request=r)
        return response([], total=0)
    calls = network(handler)
    result = await tools.zotero_create_collection("Research", apply=True)
    assert result["status"] == "write_outcome_unknown"
    assert "TEST_SECRET" not in json.dumps(result) and len(calls) == 2


@pytest.mark.asyncio
async def test_invalid_and_oversized_json_response(network, monkeypatch):
    network(lambda r: httpx.Response(200, content=b"not json TEST_SECRET_NEVER_ECHO"))
    assert (await tools.zotero_status())["status"] == "unexpected_response"
    monkeypatch.setattr(core, "MAX_RESPONSE", 8)
    network(lambda r: response([item()]))
    assert (await tools.zotero_status())["status"] == "too_large"


def test_windows_environment_process_even_empty_wins(monkeypatch):
    class Registry:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    fake = SimpleNamespace(HKEY_CURRENT_USER=0, REG_SZ=1, REG_EXPAND_SZ=2,
                           OpenKey=lambda *args: Registry(),
                           QueryValueEx=lambda handle, name: ({"ZOTERO_API_KEY": "USER_SECRET", "ZOTERO_LIBRARY_ID": "789", "ZOTERO_LIBRARY_TYPE": "group"}[name], 1))
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(core.os, "name", "nt")
    monkeypatch.setenv("ZOTERO_API_KEY", "")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    values = core.environment()
    assert values["ZOTERO_API_KEY"] == "" and values["ZOTERO_LIBRARY_ID"] == "789"
    with pytest.raises(core.ZoteroError): core.Config.load()


def test_config_repr_hides_key():
    assert "SECRET" not in repr(core.Config("SECRET", "123", "user"))


def test_zotero_registration_schema_and_write_annotations():
    from academic_helper.server import create_server
    app = create_server()
    registered = app._tool_manager._tools
    assert len(registered) == 15
    assert registered["zotero_search"].annotations.readOnlyHint is True
    assert registered["zotero_save_note"].annotations.readOnlyHint is False
    assert registered["zotero_save_note"].parameters["properties"]["apply"]["default"] is False
