"""Tests for app.core.clientip: which address a request is rate-limited under."""

from __future__ import annotations

import pytest
from app.core.clientip import client_address, key_function
from starlette.requests import Request


def request(peer: str | None, forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded is not None else []
    scope = {"type": "http", "headers": headers, "client": (peer, 1234) if peer else None}
    return Request(scope)


def test_without_trusted_proxies_the_header_is_ignored() -> None:
    assert client_address(request("10.0.0.1", "6.6.6.6"), 0) == "10.0.0.1"


def test_with_one_trusted_proxy_the_last_entry_is_the_client() -> None:
    assert client_address(request("10.0.0.1", "203.0.113.5"), 1) == "203.0.113.5"
    assert client_address(request("10.0.0.1", "1.1.1.1, 203.0.113.5"), 1) == "203.0.113.5"


def test_entries_a_client_put_in_front_can_not_change_its_address() -> None:
    honest = client_address(request("10.0.0.1", "203.0.113.5"), 1)
    spoofed = client_address(request("10.0.0.1", "9.9.9.9, 8.8.8.8, 203.0.113.5"), 1)
    assert honest == spoofed == "203.0.113.5"


def test_with_two_trusted_proxies_the_second_from_the_right_is_the_client() -> None:
    assert client_address(request("10.0.0.1", "7.7.7.7, 203.0.113.5, 10.0.0.9"), 2) == "203.0.113.5"


@pytest.mark.parametrize("forwarded", [None, "", " , ", "203.0.113.5"])
def test_a_missing_or_too_short_header_falls_back_to_the_connection(forwarded) -> None:
    assert client_address(request("10.0.0.1", forwarded), 2) == "10.0.0.1"


def test_no_client_at_all_is_one_shared_bucket() -> None:
    assert client_address(request(None), 0) == "unknown"
    assert key_function(1)(request("10.0.0.1", "203.0.113.5")) == "203.0.113.5"
