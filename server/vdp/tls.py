"""Self-signed HTTPS certificate for `vdp serve --https` (spec: LAN access with a team password and HTTPS).

Teammates' browsers warn once about a self-signed certificate; `vdp serve` and `vdp cert` print its SHA-256
fingerprint so they can check that the warning is about this server.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import ipaddress
import os
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

VALID_DAYS = 3 * 365


def _ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def local_ipv4s() -> list[str]:
    """Non-loopback IPv4 addresses of this host: what the hostname resolves to, plus the address of the default
    route (found by connecting an unbound UDP socket; nothing is sent)."""
    found: set[str] = set()
    with contextlib.suppress(OSError):
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(str(info[4][0]))
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("192.0.2.1", 9))  # TEST-NET-1, never routed; connect() on UDP only chooses the route
        found.add(s.getsockname()[0])
    return sorted(ip for ip in found if not ipaddress.ip_address(ip).is_loopback)


def default_pair(data_root: Path) -> tuple[Path, Path]:
    tls = Path(data_root) / "tls"
    return tls / "cert.pem", tls / "key.pem"


def generate_cert(cert: Path, key: Path, extra_hosts: Sequence[str] = ()) -> None:
    """A new self-signed pair for this host's names and addresses (plus `extra_hosts`, e.g. the `--host` value;
    unspecified addresses like 0.0.0.0 are skipped). Overwrites existing files."""
    hostname = socket.gethostname()
    names = {hostname, "localhost"} | {h for h in extra_hosts if _ip(h) is None}
    ips = {"127.0.0.1", "::1", *local_ipv4s()}
    ips |= {h for h in extra_hosts if (a := _ip(h)) is not None and not a.is_unspecified}
    private = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = dt.datetime.now(dt.UTC)
    san = [x509.DNSName(n) for n in sorted(names)] + [x509.IPAddress(ipaddress.ip_address(i)) for i in sorted(ips)]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(private, hashes.SHA256())
    )
    cert.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(private.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
        ))
    os.chmod(key, 0o600)  # the O_CREAT mode is masked by the umask and ignored for an existing file
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def ensure_cert(cert: Path, key: Path, extra_hosts: Sequence[str] = ()) -> bool:
    """Generate the pair unless both files exist. True when it generated them."""
    if cert.is_file() and key.is_file():
        return False
    generate_cert(cert, key, extra_hosts)
    return True


@dataclass(frozen=True)
class CertInfo:
    path: Path
    names: list[str]
    not_after: dt.datetime
    fingerprint: str  # SHA-256, colon-separated upper-case hex, as browsers show it


def cert_info(cert: Path) -> CertInfo:
    c = x509.load_pem_x509_certificate(cert.read_bytes())
    san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    names = [*san.get_values_for_type(x509.DNSName), *(str(a) for a in san.get_values_for_type(x509.IPAddress))]
    digest = c.fingerprint(hashes.SHA256()).hex().upper()
    return CertInfo(cert, names, c.not_valid_after_utc, ":".join(digest[i:i + 2] for i in range(0, len(digest), 2)))
