"""Offline tests: no real credentials or Zotero network access are used.

Run from any checkout with:
    python -m unittest discover -s .agents/skills/setup-zetero/scripts -v
"""

from contextlib import redirect_stdout
from email.message import Message
import io
import json
import ssl
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import zotero_doctor as doctor


FAKE_KEY = "OfflineTestToken_DoNotUseForAuthentication"
PRIVATE_LIBRARY_DATA = "Private bibliography details must not appear in output"
ENV = {"ZOTERO_API_KEY": FAKE_KEY, "ZOTERO_LIBRARY_ID": "123456"}


class JsonResponse:
    def __init__(self, code=200, payload=None, raw=None):
        self.code = code
        self.closed = False
        if payload is None:
            payload = {"key": FAKE_KEY, "access": {"user": {"library": True}}}
        self.data = json.dumps(payload).encode() if raw is None else raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def getcode(self):
        return self.code

    def read(self, size):
        if size != doctor.MAX_RESPONSE_BYTES + 1:
            raise AssertionError("Response reads must be bounded")
        return self.data[:size]


def valid_response(request, **kwargs):
    payload = ([{"data": {"title": PRIVATE_LIBRARY_DATA}}]
               if "/items/top" in request.full_url else None)
    return JsonResponse(payload=payload)


class UnreadableErrorBody(io.BytesIO):
    def read(self, *args):
        raise AssertionError("Error response bodies must not be read")


class DoctorTests(unittest.TestCase):
    def run_main(self, env, argv=None, opener=None):
        output = io.StringIO()
        opener = opener if opener is not None else MagicMock()
        # Tests must never fall back to real Windows User credentials.
        with patch.object(doctor.os, "name", "posix"), redirect_stdout(output):
            code = doctor.main(argv or [], environ=env, opener=opener)
        text = output.getvalue()
        self.assertNotIn(FAKE_KEY, text)
        self.assertNotIn(PRIVATE_LIBRARY_DATA, text)
        return code, json.loads(text), text

    def test_missing_key_does_not_send_any_requests(self):
        opener = MagicMock()
        code, result, _ = self.run_main({"ZOTERO_LIBRARY_ID": "123456"}, opener=opener)
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "missing_config")
        self.assertFalse(result["key_set"])
        self.assertFalse(result["config_valid"])
        opener.assert_not_called()

    def test_check_env_is_successful_without_network(self):
        opener = MagicMock()
        code, result, _ = self.run_main(ENV, ["--check-env"], opener)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "config_ready")
        self.assertEqual(result["key_access"]["status"], "not_checked")
        self.assertEqual(result["library"], {"id": "123456", "type": "user"})
        opener.assert_not_called()

    def test_invalid_config_does_not_make_requests_or_echo_values(self):
        for name, value in (
            ("ZOTERO_API_KEY", "secret\r\nInjected: value"),
            ("ZOTERO_API_KEY", "秘密"),
            ("ZOTERO_LIBRARY_ID", "../keys/current"),
            ("ZOTERO_LIBRARY_ID", "１２３"),
            ("ZOTERO_LIBRARY_ID", "0"),
            ("ZOTERO_LIBRARY_TYPE", "users/../secret"),
        ):
            with self.subTest(name=name, value=value):
                opener = MagicMock()
                code, result, text = self.run_main(dict(ENV, **{name: value}), opener=opener)
                self.assertEqual(code, 2)
                self.assertEqual(result["status"], "missing_config")
                if len(value) > 1:
                    self.assertNotIn(value, text)
                opener.assert_not_called()

    def test_raw_key_is_not_trimmed_or_included_in_config_repr(self):
        with self.assertRaises(doctor.ConfigError) as error:
            doctor.build_config(dict(ENV, ZOTERO_API_KEY=" " + FAKE_KEY))
        self.assertNotIn(FAKE_KEY, str(error.exception))
        self.assertNotIn(FAKE_KEY, repr(doctor.build_config(ENV)))

    def test_printable_ascii_token_punctuation_is_supported(self):
        token = "Opaque!%$+.?=:@[]{}/\\Token"
        config = doctor.build_config(dict(ENV, ZOTERO_API_KEY=token))
        self.assertEqual(config.api_key, token)
        code, result, output = self.run_main(dict(ENV, ZOTERO_API_KEY=token), ["--check-env"])
        self.assertEqual(code, 0)
        self.assertNotIn(token, output)

    def test_user_library_checks_only_fixed_https_get_endpoints(self):
        requests = []
        responses = []

        def opener(request, *, timeout):
            requests.append(request)
            self.assertEqual(timeout, 20)
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.get_header("Zotero-api-key"), FAKE_KEY)
            self.assertEqual(request.get_header("Zotero-api-version"), "3")
            self.assertIsNone(request.data)
            response = valid_response(request)
            responses.append(response)
            return response

        code, result, _ = self.run_main(ENV, opener=opener)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "ready")
        self.assertEqual([request.full_url for request in requests], [
            "https://api.zotero.org/keys/current",
            "https://api.zotero.org/users/123456/items/top?limit=1",
        ])
        self.assertTrue(all(response.closed for response in responses))

    def test_group_library_uses_groups_endpoint(self):
        opener = MagicMock(side_effect=valid_response)
        code, result, _ = self.run_main(dict(ENV, ZOTERO_LIBRARY_TYPE="group"), opener=opener)
        self.assertEqual(code, 0)
        self.assertEqual(result["library"]["type"], "group")
        self.assertEqual(opener.call_args_list[1].args[0].full_url,
                         "https://api.zotero.org/groups/123456/items/top?limit=1")

    def test_http_errors_are_classified_without_echoing_private_details(self):
        for http_status, category in ((302, "http_error"), (401, "unauthorized"), (403, "forbidden"),
                                      (429, "rate_limited"), (500, "http_error")):
            with self.subTest(http_status=http_status):
                body = UnreadableErrorBody(FAKE_KEY.encode())
                error = HTTPError("https://api.zotero.org/" + FAKE_KEY, http_status,
                                  FAKE_KEY, {}, body)
                opener = MagicMock(side_effect=error)
                code, result, _ = self.run_main(ENV, opener=opener)
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], category)
                self.assertEqual(result["key_access"]["http_status"], http_status)
                self.assertEqual(result["library_access"]["status"], "not_checked")
                self.assertEqual(opener.call_count, 1)
                self.assertTrue(body.closed)

    def test_connect_403_is_network_policy_and_never_invalid_key(self):
        opener = MagicMock(side_effect=URLError(
            "Tunnel connection failed: 403 Forbidden " + FAKE_KEY))
        code, result, _ = self.run_main(ENV, opener=opener)
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "network_policy")
        self.assertEqual(result["library_access"]["status"], "not_checked")

    def test_explicit_proxy_denial_is_network_policy(self):
        headers = Message()
        headers["X-Mitmproxy-Blocked-Reason"] = "BLOCKLIST"
        error = HTTPError(doctor.API_ROOT, 403, FAKE_KEY, headers,
                          UnreadableErrorBody(FAKE_KEY.encode()))
        code, result, _ = self.run_main(ENV, opener=MagicMock(side_effect=error))
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "network_policy")

    def test_tls_and_timeout_errors_are_sanitized(self):
        for error in (URLError(ssl.SSLError(FAKE_KEY)), TimeoutError(FAKE_KEY)):
            with self.subTest(error=type(error)):
                code, result, _ = self.run_main(ENV, opener=MagicMock(side_effect=error))
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], "network_or_tls_error")

    def test_library_permission_failure_preserves_successful_key_check(self):
        error = HTTPError(doctor.API_ROOT, 403, FAKE_KEY, {},
                          UnreadableErrorBody(FAKE_KEY.encode()))
        opener = MagicMock(side_effect=[JsonResponse(), error])
        code, result, _ = self.run_main(ENV, opener=opener)
        self.assertEqual(code, 1)
        self.assertEqual(result["key_access"]["status"], "ok")
        self.assertEqual(result["library_access"]["status"], "forbidden")

    def test_redirects_are_disabled_and_default_tls_and_proxy_handlers_retained(self):
        self.assertIsNone(doctor._NoRedirect().redirect_request(
            None, None, 302, "Found", {}, "https://example.invalid/private"))
        with patch.object(doctor, "build_opener") as build:
            request = object()
            doctor._open(request, timeout=20)
            self.assertIsInstance(build.call_args.args[0], doctor._NoRedirect)
            build.return_value.open.assert_called_once_with(request, timeout=20)

    def test_redirect_response_never_forwards_the_authentication_header(self):
        handler = doctor._NoRedirect()
        handler.parent = MagicMock()
        request = Request(doctor.API_ROOT + "/keys/current",
                          headers={"Zotero-API-Key": FAKE_KEY}, method="GET")
        headers = Message()
        headers["Location"] = "https://example.invalid/private"
        result = handler.http_error_302(request, JsonResponse(), 302, "Found", headers)
        self.assertIsNone(result)
        handler.parent.open.assert_not_called()

    def test_unexpected_key_responses_stop_library_checks_and_redact_body(self):
        for response in (
            JsonResponse(raw=("<html>" + FAKE_KEY + "</html>").encode()),
            JsonResponse(payload=[FAKE_KEY]),
            JsonResponse(payload={"access": FAKE_KEY}),
            JsonResponse(raw=b"{" + FAKE_KEY.encode() + b"}"),
            JsonResponse(raw=b"x" * (doctor.MAX_RESPONSE_BYTES + 1)),
        ):
            with self.subTest(body_size=len(response.data)):
                opener = MagicMock(return_value=response)
                code, result, _ = self.run_main(ENV, opener=opener)
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], "unexpected_response")
                self.assertEqual(result["library_access"]["status"], "not_checked")
                self.assertEqual(opener.call_count, 1)
                self.assertTrue(response.closed)

    def test_library_endpoint_requires_json_list(self):
        opener = MagicMock(side_effect=[JsonResponse(), JsonResponse(payload={"data": FAKE_KEY})])
        code, result, _ = self.run_main(ENV, opener=opener)
        self.assertEqual(code, 1)
        self.assertEqual(result["key_access"]["status"], "ok")
        self.assertEqual(result["library_access"]["status"], "unexpected_response")


class WindowsEnvironmentTests(unittest.TestCase):
    def registry(self, values):
        registry = SimpleNamespace(HKEY_CURRENT_USER=1, KEY_READ=2, REG_SZ=3,
                                   REG_EXPAND_SZ=4, OpenKey=MagicMock(), QueryValueEx=MagicMock())
        registry.OpenKey.return_value.__enter__.return_value = object()

        def query(key, name):
            if name not in values:
                raise FileNotFoundError(name)
            return values[name], registry.REG_SZ

        registry.QueryValueEx.side_effect = query
        return registry

    def test_windows_absent_process_variables_use_user_environment(self):
        registry = self.registry(dict(ENV, ZOTERO_LIBRARY_TYPE="group"))
        result = doctor.resolve_environment({}, platform="nt", registry=registry)
        self.assertEqual(result, dict(ENV, ZOTERO_LIBRARY_TYPE="group"))
        registry.OpenKey.assert_called_once_with(1, "Environment", 0, 2)

    def test_process_values_override_user_values_even_when_empty(self):
        registry = self.registry(dict(ENV, ZOTERO_LIBRARY_TYPE="group"))
        process = {"ZOTERO_API_KEY": "", "ZOTERO_LIBRARY_ID": "789012"}
        result = doctor.resolve_environment(process, platform="nt", registry=registry)
        self.assertEqual(result, dict(process, ZOTERO_LIBRARY_TYPE="group"))
        queried_names = [call.args[1] for call in registry.QueryValueEx.call_args_list]
        self.assertEqual(queried_names, ["ZOTERO_LIBRARY_TYPE"])
        with self.assertRaises(doctor.ConfigError):
            doctor.build_config(result)

    def test_posix_does_not_consult_registry_or_unrelated_environment(self):
        registry = self.registry(ENV)
        result = doctor.resolve_environment(dict(ENV, UNRELATED="private"),
                                            platform="posix", registry=registry)
        self.assertEqual(result, ENV)
        registry.OpenKey.assert_not_called()

    def test_unavailable_windows_registry_leaves_config_missing(self):
        registry = self.registry({})
        registry.OpenKey.side_effect = PermissionError(FAKE_KEY)
        result = doctor.resolve_environment({}, platform="nt", registry=registry)
        self.assertEqual(result, {})
        with self.assertRaises(doctor.ConfigError) as error:
            doctor.build_config(result)
        self.assertNotIn(FAKE_KEY, str(error.exception))


if __name__ == "__main__":
    unittest.main()
