from __future__ import annotations

import ipaddress
import os
import re
import socket
import stat
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from vdp.config import REPO_ROOT, Settings
from vdp.tls import cert_info, default_pair, ensure_cert, generate_cert, local_ipv4s

PUB = serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo


def test_generated_cert_covers_this_host_and_its_key_matches(tmp_path):
    cert, key = default_pair(tmp_path)
    assert cert == tmp_path / "tls" / "cert.pem" and key == tmp_path / "tls" / "key.pem"
    assert ensure_cert(cert, key, ["192.168.7.20", "0.0.0.0", "gpu-box.lan"]) is True
    info = cert_info(cert)
    for name in ("localhost", "127.0.0.1", "::1", "192.168.7.20", "gpu-box.lan", socket.gethostname()):
        assert name in info.names
    assert "0.0.0.0" not in info.names
    assert all(ip in info.names for ip in local_ipv4s())
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600
    c = x509.load_pem_x509_certificate(cert.read_bytes())
    k = serialization.load_pem_private_key(key.read_bytes(), password=None)
    assert c.public_key().public_bytes(*PUB) == k.public_key().public_bytes(*PUB)
    assert (info.not_after - c.not_valid_before_utc).days >= 3 * 365
    assert re.fullmatch(r"([0-9A-F]{2}:){31}[0-9A-F]{2}", info.fingerprint)


def test_ensure_cert_reuses_and_generate_replaces(tmp_path):
    cert, key = default_pair(tmp_path)
    ensure_cert(cert, key)
    first = cert_info(cert).fingerprint
    assert ensure_cert(cert, key) is False and cert_info(cert).fingerprint == first
    generate_cert(cert, key)
    assert cert_info(cert).fingerprint != first


def test_local_ipv4s_are_non_loopback_ipv4():
    for ip in local_ipv4s():
        addr = ipaddress.ip_address(ip)
        assert addr.version == 4 and not addr.is_loopback


def test_server_tls_settings_default_off_and_resolve_relative_paths(tmp_path):
    assert Settings.load(None).server.https is False
    p = tmp_path / "c.yaml"
    p.write_text("server: {https: true, cert: certs/c.pem, key: /abs/k.pem}\n")
    s = Settings.load(p)
    assert s.server.https and s.server.cert == REPO_ROOT / "certs" / "c.pem" and s.server.key == Path("/abs/k.pem")
