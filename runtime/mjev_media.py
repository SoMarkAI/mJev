"""Bounded media reads for the experimental HTTP runtime (not HF CLI)."""
import base64
import binascii
import http.client
import ipaddress
import mimetypes
import os
from pathlib import Path
import socket
import stat
import urllib.parse
import urllib.request


class MediaTooLarge(ValueError):
    pass


def bounded_read(handle, limit):
    data = handle.read(limit + 1)
    if len(data) > limit:
        raise MediaTooLarge(f'Media exceeds {limit} bytes')
    return data


def local_read(path, roots, limit):
    resolved = path.resolve(strict=True)
    for root in roots:
        root = Path(root).resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        # Walk using directory descriptors so changing a symlink between the
        # path check and open cannot escape the allowlist. Do not follow links.
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if not relative.parts:
                raise ValueError('Media must be a regular file')
            for part in relative.parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            leaf = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            with os.fdopen(leaf, 'rb') as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('Media must be a regular file')
                if info.st_size > limit:
                    raise MediaTooLarge(f'Media exceeds {limit} bytes')
                return bounded_read(handle, limit)
        finally:
            os.close(descriptor)
    raise ValueError('Media path is outside allowed directories')


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Use the validated IP, while verifying TLS for the requested hostname."""
    def __init__(self, host, address):
        super().__init__(host, timeout=30)
        self.address = address

    def connect(self):
        sock = socket.create_connection((self.address, 443), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


def remote_read(parsed, allowed_hosts, limit):
    host = (parsed.hostname or '').lower()
    if parsed.scheme != 'https' or host not in allowed_hosts:
        raise ValueError('Remote media requires an explicitly allowed HTTPS host')
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError('Remote credentials and nonstandard ports are not allowed')
    addresses = {row[4][0] for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError('Remote media must resolve to public addresses')
    connection = PinnedHTTPSConnection(host, sorted(addresses)[0])
    try:
        target = urllib.parse.urlunsplit(('', '', parsed.path or '/', parsed.query, ''))
        connection.request('GET', target, headers={'User-Agent': 'mJev/1'})
        response = connection.getresponse()
        # No redirect can silently change the allowlisted host or destination.
        if response.status != 200:
            raise ValueError(f'Remote media returned HTTP {response.status}; redirects are disabled')
        length = response.getheader('Content-Length')
        if length is not None and int(length) > limit:
            raise MediaTooLarge(f'Media exceeds {limit} bytes')
        return bounded_read(response, limit), response.headers.get_content_type()
    finally:
        connection.close()


def read_media(url, *, limit, roots, allowed_hosts=()):
    if limit < 1:
        raise ValueError('Media size limit must be positive')
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == 'data':
        header, separator, encoded = url.partition(',')
        if not separator or len(header) > 256:
            raise ValueError('Invalid data URL')
        if ';base64' in header:
            if len(encoded) > 4 * ((limit + 2) // 3):
                raise MediaTooLarge(f'Media exceeds {limit} bytes')
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError('Invalid base64 media') from exc
        else:
            if len(encoded) > 3 * limit:
                raise MediaTooLarge(f'Media exceeds {limit} bytes')
            data = urllib.parse.unquote_to_bytes(encoded)
        if len(data) > limit:
            raise MediaTooLarge(f'Media exceeds {limit} bytes')
        return data, header[5:].split(';', 1)[0] or 'application/octet-stream', False
    if parsed.scheme in ('http', 'https'):
        data, mime = remote_read(parsed, set(allowed_hosts), limit)
        return data, mime, True
    if parsed.scheme == 'file':
        if parsed.netloc not in ('', 'localhost') or parsed.query or parsed.fragment:
            raise ValueError('Invalid local file URL')
        path = Path(urllib.request.url2pathname(parsed.path))
    elif not parsed.scheme:
        path = Path(url)
    else:
        raise ValueError(f'Unsupported media URL scheme: {parsed.scheme}')
    return local_read(path, roots, limit), mimetypes.guess_type(path.name)[0] or 'application/octet-stream', False
