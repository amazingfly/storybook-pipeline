#!/usr/bin/env python3
"""Run colab-cli with IPv4 DNS results on hosts with unusable IPv6 routes."""

from __future__ import annotations

import socket


_getaddrinfo = socket.getaddrinfo


def ipv4_getaddrinfo(
    host: str | bytes | None,
    port: str | int | None,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> list[tuple[int, int, int, str, tuple[object, ...]]]:
    if family in (0, socket.AF_UNSPEC):
        family = socket.AF_INET
    return _getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = ipv4_getaddrinfo

from colab_cli.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
