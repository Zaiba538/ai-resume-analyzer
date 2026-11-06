"""Request ID resolution."""

from __future__ import annotations

import re

import pytest

from backend.core.request_id import get_request_id, new_request_id, resolve_request_id

UUID_HEX = re.compile(r"[0-9a-f]{32}")


def test_new_ids_are_unique_uuid4_hex() -> None:
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(UUID_HEX.fullmatch(i) for i in ids)


@pytest.mark.parametrize("valid", ["abcdefgh", "trace-1234.ABCD_x", "a" * 64])
def test_well_formed_client_id_is_reused(valid: str) -> None:
    assert resolve_request_id(valid.encode()) == valid


@pytest.mark.parametrize(
    "invalid",
    [
        b"short",
        b"a" * 65,
        b"-leading-dash",
        b"has space inside",
        b"inject\r\nX-Evil: 1",
        b'{"json":1}',
        "café-12345".encode(),
    ],
)
def test_malformed_client_id_is_replaced(invalid: bytes) -> None:
    resolved = resolve_request_id(invalid)
    assert UUID_HEX.fullmatch(resolved)


def test_missing_header_generates_id() -> None:
    assert UUID_HEX.fullmatch(resolve_request_id(None))


def test_no_request_id_outside_a_request() -> None:
    assert get_request_id() is None
