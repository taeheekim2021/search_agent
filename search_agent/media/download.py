"""Bounded HTTPS download, DNS pinning and per-hop policy checks. No archive extraction."""

import hashlib
import http.client
import ipaddress
import os
import queue
import socket
import ssl
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlsplit
from urllib.request import getproxies

from .config import MediaSettings
from .domain import MediaEntry, public_https_url


class DownloadError(RuntimeError):
    pass


class UnsafeDownload(DownloadError):
    pass


class URLPolicy:
    def __init__(self, domains: list[str]):
        self.domains = {host.lower() for host in domains}
        if any(not host or "/" in host or "*" in host or ":" in host for host in self.domains):
            raise ValueError("Download allowlist requires exact DNS hostnames, no wildcard or port")

    def check(self, url: str) -> str:
        try:
            public_https_url(url)
            host = urlsplit(url).hostname
            if host is None or host.lower() not in self.domains:
                raise UnsafeDownload("Download hostname is not explicitly allowed")
            # Reject literal IP addresses even if someone puts them in the allowlist.
            try:
                ipaddress.ip_address(host)
            except ValueError:
                pass
            else:
                raise UnsafeDownload("Literal IP addresses are not allowed")
            return host
        except ValueError as exc:
            raise UnsafeDownload("Invalid download URL") from exc

    def resolve(self, url: str, timeout: float = 10) -> tuple[str, str]:
        host = self.check(url)
        result: queue.Queue = queue.Queue(maxsize=1)

        def lookup():
            try:
                result.put(socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM))
            except OSError as exc:
                result.put(exc)

        threading.Thread(target=lookup, daemon=True).start()
        try:
            answer = result.get(timeout=timeout)
        except queue.Empty as exc:
            raise DownloadError("DNS deadline exceeded") from exc
        if isinstance(answer, Exception):
            raise answer
        addresses = {str(item[4][0]) for item in answer}
        if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
            raise UnsafeDownload("Download DNS resolves to non-public address space")
        return host, min(addresses)


class StreamResponse(Protocol):
    status: int

    def getheader(self, name: str, default=None): ...
    def read(self, size: int) -> bytes: ...
    def close(self) -> None: ...
    def set_timeout(self, seconds: float) -> None: ...


class Transport(Protocol):
    def open(self, url: str, policy: URLPolicy, timeout: float) -> StreamResponse: ...


class HTTPSStream:
    def __init__(
        self,
        connection: http.client.HTTPSConnection,
        response: http.client.HTTPResponse,
        connected_socket: ssl.SSLSocket,
        timer: threading.Timer,
    ):
        self.connection, self.response, self.socket = connection, response, connected_socket
        self.status = response.status
        self.timer = timer

    def getheader(self, name: str, default=None):
        return self.response.getheader(name, default)

    def read(self, size: int) -> bytes:
        return self.response.read1(size)

    def set_timeout(self, seconds: float) -> None:
        self.socket.settimeout(seconds)

    def close(self) -> None:
        self.timer.cancel()
        self.response.close()
        self.connection.close()


class PinnedHTTPSTransport:
    def open(self, url: str, policy: URLPolicy, timeout: float) -> StreamResponse:
        # Direct sockets cannot safely inherit a managed HTTP proxy's policy. Fail closed,
        # never silently remove proxy settings or bypass a cloud egress restriction.
        if any(getproxies().get(scheme) for scheme in ("http", "https", "all")):
            raise UnsafeDownload(
                "Managed proxy detected: direct pinned download is disabled; "
                "use an approved environment/egress transport"
            )
        started = time.monotonic()
        deadline = started + timeout
        host, address = policy.resolve(url, timeout)
        timeout -= time.monotonic() - started
        if timeout <= 0:
            raise DownloadError("DNS exhausted download deadline")
        parsed = urlsplit(url)
        connection = http.client.HTTPSConnection(host, timeout=timeout)
        raw = socket.create_connection((address, 443), timeout=timeout)
        timer = None
        try:
            connected = ssl.create_default_context().wrap_socket(
                raw, server_hostname=host, do_handshake_on_connect=False
            )

            connection.sock = connected  # Exact validated IP; TLS validates original hostname.

            def expire():
                try:
                    connected.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connected.close()

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                connected.close()
                raise DownloadError("Connection exhausted download deadline")
            timer = threading.Timer(remaining, expire)
            timer.daemon = True
            timer.start()
            connected.do_handshake()
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            connection.request(
                "GET",
                path,
                headers={"User-Agent": "KidsSearchAgent/0.2", "Accept-Encoding": "identity"},
            )
            return HTTPSStream(connection, connection.getresponse(), connected, timer)
        except Exception:
            if timer:
                timer.cancel()
            raw.close()
            connection.close()
            raise


@dataclass
class Downloaded:
    local_path: str
    checksum_sha256: str
    size_bytes: int
    reused: bool = False


class MediaStorage(Protocol):
    """Object storage can implement this without changing ingestion or OpenSearch."""

    def download(self, entry: MediaEntry, prior: dict | None = None) -> Downloaded: ...


class LocalMediaStore:
    def __init__(self, settings: MediaSettings, transport: Transport | None = None):
        self.settings = settings
        self.policy = URLPolicy(settings.allowed_domains)
        self.transport = transport or PinnedHTTPSTransport()

    def target(self, entry: MediaEntry) -> Path:
        if entry.media_format is None:
            raise ValueError("Downloading requires a known media_format")
        root = self.settings.root.resolve()
        target = root / f"{entry.content_id}.{entry.media_format}"
        if target.is_symlink() or target.resolve().parent != root:
            raise UnsafeDownload("Unsafe storage path or symlink")
        return target

    def download(self, entry: MediaEntry, prior: dict | None = None) -> Downloaded:
        if entry.rights_scope != "media":
            raise ValueError("Metadata licensing does not authorize media acquisition")
        self.policy.check(entry.original_url)
        target = self.target(entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        # A matching retained receipt is mandatory when no source checksum is provided.
        expected = entry.expected_sha256
        if prior and prior.get("original_url") == entry.original_url:
            expected = expected or prior.get("checksum_sha256")
        if target.is_file() and expected and 0 < target.stat().st_size <= self.settings.max_bytes:
            with target.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if digest == expected:
                return Downloaded(target.name, digest, target.stat().st_size, reused=True)
        deadline = time.monotonic() + self.settings.timeout_seconds
        for attempt in range(self.settings.retries + 1):
            try:
                return self._attempt(entry, target, deadline)
            except UnsafeDownload:
                raise
            except (OSError, http.client.HTTPException, DownloadError) as exc:
                if attempt == self.settings.retries or time.monotonic() >= deadline:
                    # No URL/query/credentials are copied to failure receipts.
                    raise DownloadError(f"Download failed ({type(exc).__name__})") from exc
        raise AssertionError("unreachable")

    def _attempt(self, entry: MediaEntry, target: Path, deadline: float) -> Downloaded:
        url = entry.original_url
        response: StreamResponse | None = None
        temporary: Path | None = None
        try:
            for hop in range(self.settings.max_redirects + 1):
                self.policy.check(url)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DownloadError("Download deadline exceeded")
                response = self.transport.open(url, self.policy, remaining)
                if response.status in (301, 302, 303, 307, 308):
                    location = response.getheader("Location")
                    response.close()
                    response = None
                    if not location or hop == self.settings.max_redirects:
                        raise UnsafeDownload("Redirect limit or missing Location")
                    url = urljoin(url, location)
                    self.policy.check(url)
                    continue
                break
            if response is not None and 400 <= response.status < 500:
                raise UnsafeDownload("Media access denied or rate limited; not retrying")
            if response is None or response.status != 200:
                raise DownloadError("Media request did not return HTTP 200")
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise UnsafeDownload("Encoded media response rejected")
            content_type = response.getheader("Content-Type", "").split(";")[0].lower()
            if not content_type.startswith(("video/", "audio/")):
                raise UnsafeDownload("Expected an audio/video Content-Type")
            length = response.getheader("Content-Length")
            if length and (not length.isdigit() or int(length) > self.settings.max_bytes):
                raise UnsafeDownload("Invalid or excessive Content-Length")
            digest, size = hashlib.sha256(), 0
            prefix = b""
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".download-", delete=False
            ) as f:
                temporary = Path(f.name)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise DownloadError("Download deadline exceeded")
                    response.set_timeout(remaining)
                    chunk = response.read(min(65536, self.settings.max_bytes - size + 1))
                    if not chunk:
                        break
                    if len(prefix) < 64:
                        prefix = (prefix + chunk)[:64]
                    size += len(chunk)
                    if size > self.settings.max_bytes:
                        raise UnsafeDownload("Media exceeds maximum bytes")
                    digest.update(chunk)
                    f.write(chunk)
                if not size or (length and size != int(length)):
                    raise DownloadError("Empty or incomplete media")
                if not valid_magic(entry.media_format, prefix):
                    raise UnsafeDownload("Media signature does not match declared format")
                checksum = digest.hexdigest()
                if entry.expected_sha256 and entry.expected_sha256 != checksum:
                    raise UnsafeDownload("Media checksum mismatch")
                f.flush()
                os.fsync(f.fileno())
            # Check again just before replacement, and replace atomically in MEDIA_ROOT.
            if self.target(entry) != target:
                raise UnsafeDownload("Storage destination changed")
            os.replace(temporary, target)
            return Downloaded(target.name, checksum, size)
        finally:
            if response:
                response.close()
            if temporary:
                temporary.unlink(missing_ok=True)


def valid_magic(media_format: str | None, prefix: bytes) -> bool:
    if media_format == "webm":
        return prefix.startswith(b"\x1a\x45\xdf\xa3")
    if media_format in ("ogg", "ogv"):
        return prefix.startswith(b"OggS")
    if media_format == "mp4":
        return prefix[4:8] == b"ftyp"
    if media_format == "wav":
        return prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE"
    if media_format == "flac":
        return prefix.startswith(b"fLaC")
    return media_format == "mp3" and (
        prefix.startswith(b"ID3")
        or len(prefix) >= 2
        and prefix[0] == 255
        and prefix[1] & 224 == 224
    )
