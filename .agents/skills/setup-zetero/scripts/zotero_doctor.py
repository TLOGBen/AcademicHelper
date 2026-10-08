#!/usr/bin/env python3
"""Read-only Zotero checks using credentials from the process environment.

No credentials are read from files, included in output, or sent to any host
other than api.zotero.org.  On Windows, a variable absent from the process
may be supplied by the persistent User environment.  urllib's default
verified TLS and proxy handling are retained.  This helper never creates,
changes, or deletes Zotero items.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import socket
import ssl
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


API_ROOT = "https://api.zotero.org"
REQUEST_TIMEOUT = 20
MAX_RESPONSE_BYTES = 1024 * 1024
LIBRARY_TYPES = {"user": "users", "group": "groups"}
ENV_NAMES = ("ZOTERO_API_KEY", "ZOTERO_LIBRARY_ID", "ZOTERO_LIBRARY_TYPE")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        # A response must not forward credentials to a different destination.
        return None


def _open(request, *, timeout):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def resolve_environment(environ=None, *, platform=None, registry=None) -> dict:
    """Read only Zotero variables, preferring even empty process values.

    Windows User environment values are a fallback for variables absent from
    the current process; no registry values are modified or printed.
    """
    environ = os.environ if environ is None else environ
    resolved = {name: environ[name] for name in ENV_NAMES if name in environ}
    platform = os.name if platform is None else platform
    missing = [name for name in ENV_NAMES if name not in resolved]
    if platform != "nt" or not missing:
        return resolved
    if registry is None:
        try:
            import winreg as registry
        except ImportError:
            return resolved
    try:
        with registry.OpenKey(
            registry.HKEY_CURRENT_USER, "Environment", 0, registry.KEY_READ
        ) as user_environment:
            for name in missing:
                try:
                    value, value_type = registry.QueryValueEx(user_environment, name)
                except OSError:
                    continue
                if value_type in (registry.REG_SZ, registry.REG_EXPAND_SZ) and isinstance(value, str):
                    resolved[name] = value
    except OSError:
        pass
    return resolved


@dataclass(frozen=True)
class Config:
    api_key: str = field(repr=False)
    library_id: str
    library_type: str


class ConfigError(ValueError):
    """Safe errors containing variable names and fixed descriptions only."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _key_format_valid(value: object) -> bool:
    # Zotero keys are opaque ASCII tokens.  Do not impose an undocumented
    # length or try to infer authentication validity from token syntax.
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and all(0x21 <= ord(character) <= 0x7E for character in value)
    )


def _library_id_valid(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.isascii()
        and value.isdecimal()
        and bool(value.strip("0"))
    )


def build_config(environ: Mapping[str, str]) -> Config:
    """Validate raw process variables without consulting local secret files."""
    key = environ.get("ZOTERO_API_KEY", "")
    library_id = environ.get("ZOTERO_LIBRARY_ID", "")
    library_type = environ.get("ZOTERO_LIBRARY_TYPE", "user")
    errors = []
    if not key:
        errors.append("ZOTERO_API_KEY is missing")
    elif not _key_format_valid(key):
        errors.append("ZOTERO_API_KEY must be an ASCII token without whitespace")
    if not library_id:
        errors.append("ZOTERO_LIBRARY_ID is missing")
    elif not _library_id_valid(library_id):
        errors.append("ZOTERO_LIBRARY_ID must be a positive numeric ID")
    if library_type not in LIBRARY_TYPES:
        errors.append("ZOTERO_LIBRARY_TYPE must be user or group")
    if errors:
        raise ConfigError(errors)
    return Config(key, library_id, library_type)


def _metadata(environ: Mapping[str, str]) -> dict:
    library_id = environ.get("ZOTERO_LIBRARY_ID", "")
    library_type = environ.get("ZOTERO_LIBRARY_TYPE", "user")
    return {
        "key_set": bool(environ.get("ZOTERO_API_KEY", "")),
        "library": {
            "id": library_id if _library_id_valid(library_id) else None,
            "type": library_type if library_type in LIBRARY_TYPES else None,
        },
    }


def _connect_blocked(reason: object) -> bool:
    # This text is examined only for classification and is never emitted.
    message = str(reason).lower()
    return "tunnel connection failed" in message and "403" in message


def _error_result(error: Exception) -> dict:
    if isinstance(error, HTTPError):
        status = error.code
        # Common explicit proxy-policy headers distinguish network rejection
        # from an origin server's permission response without reading bodies.
        proxy_rejection = status == 403 and bool(
            error.headers
            and (
                error.headers.get("X-Mitmproxy-Blocked-Reason")
                or error.headers.get("X-Denied-Reason")
            )
        )
        if proxy_rejection or _connect_blocked(error.reason):
            category = "network_policy"
        elif status == 403:
            category = "forbidden"
        elif status == 401:
            category = "unauthorized"
        elif status == 429:
            category = "rate_limited"
        else:
            category = "http_error"
        # Never return the response body, URL, reason, headers, or exception.
        return {"status": category, "http_status": status}
    if isinstance(error, URLError) and _connect_blocked(error.reason):
        return {"status": "network_policy"}
    return {"status": "network_or_tls_error"}


def _get(path: str, config: Config, opener: Callable, expected: str) -> dict:
    request = Request(
        API_ROOT + path,
        headers={
            "Zotero-API-Key": config.api_key,
            "Zotero-API-Version": "3",
            "Accept": "application/json",
            "User-Agent": "AcademicHelper-Zotero-Doctor/1.0",
        },
        method="GET",
    )
    try:
        # Verify bounded JSON to distinguish Zotero responses from proxy HTML.
        # Parsed private metadata and key values are never included in output.
        with opener(request, timeout=REQUEST_TIMEOUT) as response:
            status = response.getcode()
            if status is not None and 200 <= status < 300:
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    return {"status": "unexpected_response", "http_status": status}
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeError, RecursionError):
                    return {"status": "unexpected_response", "http_status": status}
                if expected == "key":
                    valid = isinstance(payload, dict) and isinstance(payload.get("access"), dict)
                else:
                    valid = isinstance(payload, list)
                if not valid:
                    return {"status": "unexpected_response", "http_status": status}
                return {"status": "ok", "http_status": status}
            return {"status": "http_error", "http_status": status}
    except (HTTPError, URLError, OSError, ssl.SSLError, socket.timeout) as error:
        result = _error_result(error)
        if isinstance(error, HTTPError):
            error.close()
        return result


def check_access(config: Config, opener: Callable = _open) -> dict:
    """Authenticate and check one library with two fixed-host GET requests."""
    key_access = _get("/keys/current", config, opener, "key")
    if key_access["status"] != "ok":
        return {
            "status": key_access["status"],
            "key_access": key_access,
            "library_access": {"status": "not_checked"},
        }
    prefix = LIBRARY_TYPES[config.library_type]
    library_access = _get(
        f"/{prefix}/{config.library_id}/items/top?limit=1", config, opener, "library"
    )
    return {
        "status": "ready" if library_access["status"] == "ok" else library_access["status"],
        "key_access": key_access,
        "library_access": library_access,
    }


def main(argv=None, *, environ=None, opener: Callable = _open) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-env", action="store_true", help="validate environment without network requests"
    )
    args = parser.parse_args(argv)
    environ = resolve_environment(environ)
    result = _metadata(environ)
    try:
        config = build_config(environ)
    except ConfigError as error:
        result.update(
            status="missing_config", config_valid=False,
            key_access={"status": "not_checked"},
            library_access={"status": "not_checked"},
            errors=error.errors,
        )
        print(json.dumps(result, ensure_ascii=True))
        return 2
    result["config_valid"] = True
    if args.check_env:
        result.update(
            status="config_ready",
            key_access={"status": "not_checked"},
            library_access={"status": "not_checked"},
        )
    else:
        result.update(check_access(config, opener))
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result["status"] in {"ready", "config_ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
