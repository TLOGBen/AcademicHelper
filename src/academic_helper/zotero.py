"""Environment-only Zotero Web API v3 client and safe write helpers.

No local credential files, implicit retries, deletes, annotation writes or file uploads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import httpx

API = "https://api.zotero.org"
MANAGED_TAG = "AH:"
MAX_RESPONSE = 4 * 1024 * 1024
KEY_RE = re.compile(r"[A-Z0-9]{8}\Z")


class ZoteroError(Exception):
    """Fixed, credential-free message and a machine-readable state."""

    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ZoteroError("invalid_input", message)


def object_key(value: str) -> str:
    require(isinstance(value, str) and bool(KEY_RE.fullmatch(value)), "Expected an 8-character Zotero object key.")
    return value


def environment() -> dict[str, str]:
    names = ("ZOTERO_API_KEY", "ZOTERO_LIBRARY_ID", "ZOTERO_LIBRARY_TYPE")
    values = {name: os.environ[name] for name in names if name in os.environ}
    if os.name == "nt" and any(name not in values for name in names):
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry:
                for name in names:
                    if name not in values:
                        try:
                            value, kind = winreg.QueryValueEx(registry, name)
                            if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) and isinstance(value, str):
                                values[name] = value
                        except OSError:
                            pass
        except OSError:
            pass
    return values


@dataclass(frozen=True)
class Config:
    api_key: str = field(repr=False)
    library_id: str
    library_type: str

    @classmethod
    def load(cls) -> Config:
        values = environment()
        secret = values.get("ZOTERO_API_KEY", "")
        library_id = values.get("ZOTERO_LIBRARY_ID", "")
        library_type = values.get("ZOTERO_LIBRARY_TYPE", "user")
        if not secret or not secret.isascii() or any(not 33 <= ord(c) <= 126 for c in secret):
            raise ZoteroError("missing_config", "Provide ZOTERO_API_KEY through a secure environment binding.")
        if not library_id.isascii() or not library_id.isdecimal() or not library_id.strip("0"):
            raise ZoteroError("missing_config", "Set a positive ZOTERO_LIBRARY_ID.")
        if library_type not in ("user", "group"):
            raise ZoteroError("missing_config", "ZOTERO_LIBRARY_TYPE must be user or group.")
        return cls(secret, library_id, library_type)

    @property
    def path(self) -> str:
        prefix = "users" if self.library_type == "user" else "groups"
        return f"/{prefix}/{self.library_id}"


class Client:
    def __init__(self, config: Config):
        self.config = config
        self.http = httpx.AsyncClient(timeout=20, follow_redirects=False)

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *args) -> None:
        await self.http.aclose()

    async def request(self, method: str, path: str, *, params=None, payload=None, version=None):
        headers = {"Zotero-API-Key": self.config.api_key, "Zotero-API-Version": "3"}
        if version is not None:
            require(type(version) is int and version >= 0, "Expected a nonnegative object version.")
            headers["If-Unmodified-Since-Version"] = str(version)
        if method == "POST":
            token = self.config.path + path + json.dumps(payload, sort_keys=True, ensure_ascii=True)
            headers["Zotero-Write-Token"] = hashlib.sha256(token.encode()).hexdigest()[:32]
        try:
            async with self.http.stream(method, API + self.config.path + path,
                                        params=params, json=payload, headers=headers) as response:
                if response.status_code >= 300:
                    state = {401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict",
                             412: "version_conflict", 413: "too_large", 429: "rate_limited"}.get(response.status_code, "http_error")
                    raise ZoteroError(state, f"Zotero returned HTTP {response.status_code}; no automatic retry was performed.")
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_RESPONSE:
                        raise ZoteroError("too_large", "Zotero response exceeded the configured limit.")
                    chunks.append(chunk)
                total = response.headers.get("Total-Results")
                meta = {"total": int(total) if total and total.isdecimal() else None}
                if response.status_code == 204:
                    return None, meta
                try:
                    return json.loads(b"".join(chunks)), meta
                except (ValueError, UnicodeError):
                    raise ZoteroError("unexpected_response", "Zotero returned invalid JSON.") from None
        except httpx.HTTPError:
            state = "write_outcome_unknown" if method != "GET" else "network_or_tls_error"
            raise ZoteroError(state, "Zotero connection failed; check current remote state before retrying any write.") from None

    async def listing(self, path: str, *, start=0, limit=25, **params) -> dict:
        require(type(start) is int and start >= 0 and type(limit) is int and 1 <= limit <= 100,
                "Pagination requires start >= 0 and limit between 1 and 100.")
        rows, meta = await self.request("GET", path, params={"start": start, "limit": limit, **params})
        if not isinstance(rows, list):
            raise ZoteroError("unexpected_response", "Expected a Zotero list.")
        more = start + len(rows) < meta["total"] if meta["total"] is not None else len(rows) == limit
        return {"records": rows, "start": start, "total": meta["total"], "has_more": more,
                "next_start": start + len(rows) if more else None}

    async def item(self, key: str) -> dict:
        result, _ = await self.request("GET", "/items/" + object_key(key))
        if not isinstance(result, dict) or not isinstance(result.get("data"), dict) or type(result.get("version")) is not int:
            raise ZoteroError("unexpected_response", "Expected a versioned Zotero item.")
        return result

    async def all_children(self, key: str) -> list[dict]:
        # Bound work; never create a duplicate note when the scan is incomplete.
        rows = []
        for start in range(0, 1000, 100):
            page = await self.listing(f"/items/{object_key(key)}/children", start=start, limit=100)
            rows.extend(page["records"])
            if not page["has_more"]:
                return rows
        raise ZoteroError("scan_incomplete", "Too many child items to safely complete this write preview.")

    async def create(self, path: str, data: dict) -> dict:
        result, _ = await self.request("POST", path, payload=[data])
        if not isinstance(result, dict):
            raise ZoteroError("unexpected_response", "Expected a Zotero write result.")
        if result.get("failed"):
            # Server failure objects can contain private data; never echo them.
            raise ZoteroError("write_failed", "Zotero rejected the object; no raw response is exposed.")
        success = result.get("successful", {}).get("0")
        if isinstance(success, dict) and success.get("key"):
            return {"status": "created", "key": success["key"], "version": success.get("version")}
        unchanged = result.get("unchanged", {}).get("0")
        if unchanged:
            return {"status": "unchanged", "key": unchanged}
        raise ZoteroError("unexpected_response", "Zotero did not acknowledge the write.")


def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    require(bool(re.fullmatch(r"10\.\d{4,9}/\S+", value)), "A valid DOI is required for safe bibliographic deduplication.")
    return value
