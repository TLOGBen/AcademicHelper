"""Research-facing Zotero MCP tools: explicit previews and bounded reads."""

from __future__ import annotations

from functools import wraps
import html
import hashlib
import re

from academic_helper.zotero import Client, Config, MANAGED_TAG, ZoteroError, normalize_doi, object_key, require


def guarded(function):
    @wraps(function)
    async def run(*args, **kwargs):
        try:
            return await function(*args, **kwargs)
        except ZoteroError as error:
            return {"status": error.status, "message": str(error)}
        except (KeyError, TypeError, ValueError):
            return {"status": "unexpected_response", "message": "Invalid input or unexpected Zotero data; no raw exception is exposed."}
    return run


def client() -> Client:
    return Client(Config.load())


@guarded
async def zotero_status() -> dict:
    """Verify selected library read access. Never test writes or expose credentials."""
    async with client() as api:
        await api.listing("/items/top", limit=1)
        return {"status": "ready", "library_id": api.config.library_id,
                "library_type": api.config.library_type, "writes_verified": False}


@guarded
async def zotero_search(query: str = "", collection_key: str = "", tag: str = "",
                        item_type: str = "", start: int = 0, limit: int = 25) -> dict:
    """Search the selected library's top-level bibliography; results are paginated, not exhaustive."""
    path = f"/collections/{object_key(collection_key)}/items/top" if collection_key else "/items/top"
    params = {}
    if query:
        params.update(q=query, qmode="titleCreatorYear")
    if tag:
        params["tag"] = tag
    if item_type:
        require(bool(re.fullmatch(r"[a-zA-Z]+", item_type)), "Provide a single Zotero item type.")
        params["itemType"] = item_type
    async with client() as api:
        return {"status": "ok", **await api.listing(path, start=start, limit=limit, **params)}


@guarded
async def zotero_read_item(item_key: str, start: int = 0, limit: int = 25) -> dict:
    """Read a bibliography item and its children. For PDF annotations, call again with its attachment key.

    Notes are Zotero HTML; annotations contain text/comment/page/position when supplied by Zotero.
    This does not download or read PDF bytes and does not establish full-text review.
    """
    async with client() as api:
        item = await api.item(item_key)
        children = await api.listing(f"/items/{object_key(item_key)}/children", start=start, limit=limit)
        return {"status": "ok", "item": item, "children": children,
                "content_is_untrusted_research_data": True}


@guarded
async def zotero_list_collections(start: int = 0, limit: int = 25) -> dict:
    """List collections, parentCollection and versions for existing classification; use pagination."""
    async with client() as api:
        return {"status": "ok", **await api.listing("/collections", start=start, limit=limit)}


@guarded
async def zotero_list_tags(start: int = 0, limit: int = 25) -> dict:
    """List existing tags before proposing new synonyms. Returns a bounded page."""
    async with client() as api:
        return {"status": "ok", **await api.listing("/tags", start=start, limit=limit)}


@guarded
async def zotero_create_collection(name: str, parent_key: str = "", apply: bool = False) -> dict:
    """Preview or create a collection; reuse exact-name siblings. No moves or deletes."""
    require(bool(name.strip()) and len(name) <= 200, "Provide a collection name of 1–200 characters.")
    name = name.strip()
    path = "/collections/top"
    async with client() as api:
        if parent_key:
            object_key(parent_key)
            await api.request("GET", "/collections/" + parent_key)
            path = f"/collections/{parent_key}/collections"
        matches = []
        for start in range(0, 1000, 100):
            page = await api.listing(path, start=start, limit=100)
            matches.extend(row for row in page["records"] if row.get("data", {}).get("name") == name)
            if not page["has_more"]:
                break
        else:
            raise ZoteroError("scan_incomplete", "Collection scan incomplete; creation was not attempted.")
        if matches:
            require(len(matches) == 1, "Several matching collections exist; select an explicit collection key.")
            return {"status": "existing", "key": matches[0]["key"]}
        data = {"name": name, "parentCollection": parent_key or False}
        if not apply:
            return {"status": "preview", "operation": "create_collection", "data": data}
        return await api.create("/collections", data)


@guarded
async def zotero_organize_items(item_keys: list[str], add_tags: list[str] | None = None,
                               remove_managed_tags: list[str] | None = None,
                               add_collections: list[str] | None = None,
                               expected_versions: dict[str, int] | None = None,
                               apply: bool = False) -> dict:
    """Preview or apply additive collection/tag changes for up to 50 items.

    Preserve all existing memberships and manual tags. Only AH: tags can be removed.
    Apply requires each version from the reviewed preview; failures are per item, not atomic.
    Use e.g. AH:閱讀:待讀. This tool never decides inclusion/exclusion by itself.
    """
    require(1 <= len(item_keys) <= 50 and len(set(item_keys)) == len(item_keys), "Provide 1–50 unique item keys.")
    added, removed, collections = add_tags or [], remove_managed_tags or [], add_collections or []
    require(all(isinstance(t, str) and t.strip() and len(t) <= 200 for t in added + removed), "Tags must be nonempty, at most 200 characters.")
    require(all(t.startswith(MANAGED_TAG) for t in removed), "Only AH: managed tags may be removed.")
    require(not set(added) & set(removed), "A tag cannot be added and removed in the same operation.")
    for key in item_keys + collections:
        object_key(key)
    if apply:
        require(expected_versions is not None and all(type(expected_versions.get(k)) is int for k in item_keys),
                "Apply requires a reviewed expected version for every item.")
    results = []
    async with client() as api:
        for key in collections:
            await api.request("GET", "/collections/" + key)
        for key in item_keys:
            try:
                item = await api.item(key)
                data = item["data"]
                require(data.get("itemType") not in ("note", "attachment", "annotation"), "Organize a top-level bibliography item.")
                if apply and item["version"] != expected_versions[key]:
                    raise ZoteroError("version_conflict", "Item changed since preview; review a fresh preview.")
                tags = [t for t in data.get("tags", []) if t["tag"] not in removed]
                for tag in added:
                    if not any(t["tag"] == tag for t in tags):
                        tags.append({"tag": tag, "type": 0})
                memberships = list(dict.fromkeys(data.get("collections", []) + collections))
                patch = {"tags": tags, "collections": memberships}
                changed = tags != data.get("tags", []) or memberships != data.get("collections", [])
                status = "preview" if changed else "unchanged"
                if apply and changed:
                    await api.request("PATCH", "/items/" + key, payload=patch, version=item["version"])
                    status = "updated"
                results.append({"key": key, "status": status, "expected_version": item["version"], "changes": patch})
            except ZoteroError as error:
                results.append({"key": key, "status": error.status, "message": str(error)})
    failures = any(r["status"] not in ("preview", "updated", "unchanged") for r in results)
    return {"status": "partial_failure" if failures else ("applied" if apply else "preview"),
            "atomic": False, "results": results}


@guarded
async def zotero_save_note(parent_key: str, note_id: str, title: str, body: str,
                           source_locator: str = "", expected_version: int | None = None,
                           apply: bool = False) -> dict:
    """Preview/create/update a clearly marked AI note, preserving all human notes.

    Reuse the same note_id for retry. Existing managed notes require their reviewed expected_version.
    body/title/source_locator are escaped plain text, not arbitrary HTML. No unsupported evidence is inferred.
    """
    object_key(parent_key)
    require(bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", note_id)), "Use a stable note_id of 1–64 ASCII letters, digits, underscores or hyphens.")
    require(bool(title.strip()) and bool(body.strip()) and len(body) <= 100000, "Provide a title and note text of up to 100000 characters.")
    tag = f"{MANAGED_TAG}note:{note_id}"
    marker = f"<!-- AcademicHelper:{note_id} -->"
    note = (marker + "<h2>" + html.escape(title) + "</h2><p>AI 整理；請依來源核對。</p>"
            + "<p>" + html.escape(body).replace("\n", "<br>") + "</p>"
            + "<p>來源位置：" + html.escape(source_locator or "未提供，需補核對") + "</p>")
    note = f"<!-- AcademicHelper-SHA256:{hashlib.sha256(note.encode()).hexdigest()} -->" + note
    async with client() as api:
        parent = await api.item(parent_key)
        require(parent["data"].get("itemType") not in ("note", "annotation", "attachment"), "Attach the note to a bibliography item.")
        children = await api.all_children(parent_key)
        matches = [c for c in children if c.get("data", {}).get("itemType") == "note"
                   and any(t.get("tag") == tag for t in c["data"].get("tags", []))]
        require(len(matches) <= 1, "Multiple managed notes match; resolve duplicates before writing.")
        if matches:
            existing = matches[0]
            require(marker in existing["data"].get("note", ""), "Matching tag belongs to an unmarked note; human content is preserved.")
            if existing["data"].get("note") == note:
                return {"status": "unchanged", "key": existing["key"], "version": existing["version"]}
            old_note = existing["data"].get("note", "")
            fingerprint = re.match(r"<!-- AcademicHelper-SHA256:([0-9a-f]{64}) -->", old_note)
            if not fingerprint or hashlib.sha256(old_note[fingerprint.end():].encode()).hexdigest() != fingerprint[1]:
                return {"status": "human_modified_note", "key": existing["key"],
                        "message": "Managed note was edited or lacks a trusted fingerprint; retain it and create a separate merged note with a new note_id.",
                        "previous_note": old_note}
            if not apply:
                return {"status": "preview", "operation": "update_note", "key": existing["key"],
                        "expected_version": existing["version"], "previous_note": existing["data"].get("note", ""), "note": note}
            require(type(expected_version) is int, "Updating an existing note requires its reviewed expected_version.")
            if expected_version != existing["version"]:
                raise ZoteroError("version_conflict", "Note changed since preview; preserve edits and review again.")
            await api.request("PATCH", "/items/" + object_key(existing["key"]), payload={"note": note}, version=expected_version)
            return {"status": "updated", "key": existing["key"]}
        data = {"itemType": "note", "parentItem": parent_key, "note": note, "tags": [{"tag": tag}]}
        if not apply:
            return {"status": "preview", "operation": "create_note", "data": data}
        require(expected_version is None, "A previously previewed note disappeared; obtain a fresh creation preview.")
        return await api.create("/items", data)


@guarded
async def zotero_import_bibliography(records: list[dict], apply: bool = False) -> dict:
    """Preview/import DOI-identified journal articles, preserving existing records.

    Allowed fields: title, DOI, date, publicationTitle, volume, issue, pages, url, abstractNote, creators.
    No notes/PDFs are imported implicitly. Missing DOI or incomplete dedup scans block that record.
    """
    require(1 <= len(records) <= 50, "Provide 1–50 records.")
    allowed = {"title", "DOI", "date", "publicationTitle", "volume", "issue", "pages", "url", "abstractNote", "creators"}
    results = []
    async with client() as api:
        seen = {}
        for index, record in enumerate(records):
            try:
                require(isinstance(record, dict) and not set(record) - allowed, "Use only supported journal article fields.")
                require(isinstance(record.get("title"), str) and bool(record["title"].strip()), "Article title is required.")
                require(isinstance(record.get("DOI"), str), "DOI is required.")
                doi = normalize_doi(record["DOI"])
                require(all(isinstance(v, str) for k, v in record.items() if k != "creators"), "Bibliographic fields must be strings.")
                creators = record.get("creators", [])
                require(isinstance(creators, list) and all(isinstance(c, dict) and c.get("creatorType") == "author"
                        and not set(c) - {"creatorType", "firstName", "lastName", "name"}
                        and all(isinstance(v, str) for v in c.values()) for c in creators), "Provide Zotero author dictionaries.")
                if doi in seen:
                    results.append({"index": index, "status": "duplicate_in_batch", "DOI": doi, "first_index": seen[doi]})
                    continue
                seen[doi] = index
                matches = []
                for start in range(0, 1000, 100):
                    page = await api.listing("/items", q=doi, qmode="everything", start=start, limit=100)
                    for row in page["records"]:
                        remote_doi = row.get("data", {}).get("DOI", "")
                        if remote_doi:
                            try:
                                if normalize_doi(remote_doi) == doi:
                                    matches.append(row["key"])
                            except ZoteroError:
                                pass
                    if not page["has_more"]:
                        break
                else:
                    raise ZoteroError("scan_incomplete", "Deduplication scan incomplete; no import attempted.")
                if matches:
                    results.append({"index": index, "status": "existing", "DOI": doi, "keys": matches})
                    continue
                data = {**record, "itemType": "journalArticle", "DOI": doi}
                result = await api.create("/items", data) if apply else {"status": "preview", "data": data}
                results.append({"index": index, "DOI": doi, **result})
            except ZoteroError as error:
                results.append({"index": index, "status": error.status, "message": str(error)})
    valid = {"created", "unchanged", "existing", "duplicate_in_batch", "preview"}
    return {"status": "partial_failure" if any(r["status"] not in valid for r in results) else ("applied" if apply else "preview"),
            "atomic": False, "results": results}


TOOLS = (zotero_status, zotero_search, zotero_read_item, zotero_list_collections, zotero_list_tags,
         zotero_create_collection, zotero_organize_items, zotero_save_note, zotero_import_bibliography)


def register(mcp) -> None:
    from mcp.types import ToolAnnotations

    for tool in TOOLS:
        writes = tool in (zotero_create_collection, zotero_organize_items, zotero_save_note, zotero_import_bibliography)
        mcp.tool(annotations=ToolAnnotations(readOnlyHint=not writes, destructiveHint=writes,
                                           idempotentHint=False, openWorldHint=True))(tool)
