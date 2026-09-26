"""Selective downloads with atomic files and strict HTTP Range ZIP access."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import zipfile
import requests


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def url(repo, revision, filename):
    return f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'


def transport_url(url_):
    endpoint = os.environ.get('MJEV_HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    if not endpoint.startswith('https://'):
        raise ValueError('Mirror transport must use HTTPS')
    return url_.replace('https://huggingface.co/', endpoint+'/', 1)


def download(url_, target, expected_sha=None, expected_size=None):
    """No automatic retries. Completed files have size/hash receipts; failures retain .part."""
    target = Path(target); receipt = target.with_suffix(target.suffix+'.receipt.json')
    if target.is_file() and receipt.is_file():
        old = json.loads(receipt.read_text())
        if old['url'] == url_ and target.stat().st_size == old['bytes'] and sha256(target) == old['sha256']:
            if (not expected_sha or expected_sha == old['sha256']) and (not expected_size or expected_size == old['bytes']):
                return old
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix+'.part')
    digest = hashlib.sha256(); count = 0
    if part.exists():
        with part.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                digest.update(block); count += len(block)
    offset = count
    headers = {'Accept-Encoding':'identity'}
    if offset: headers['Range'] = f'bytes={offset}-'
    with requests.get(transport_url(url_), headers=headers, stream=True, timeout=(20,90)) as response:
        response.raise_for_status()
        if offset:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
            if response.status_code != 206 or not match or int(match[1]) != offset:
                raise ValueError('Resume requires exact HTTP Range response')
            expected_total = int(match[3])
        else:
            if response.status_code != 200: raise ValueError('Expected complete file response')
            length=response.headers.get('Content-Length')
            expected_total=int(length) if length is not None else None
        with part.open('ab' if offset else 'wb') as f:
            for block in response.iter_content(4*1024*1024):
                count += len(block); digest.update(block); f.write(block)
        if expected_total is not None and count != expected_total:
            raise ValueError('Truncated download')
    if expected_size is not None and count != expected_size:
        raise ValueError('Source size mismatch')
    hexdigest = digest.hexdigest()
    if expected_sha and hexdigest != expected_sha:
        raise ValueError('Source SHA256 mismatch')
    part.replace(target)
    record = {'url': url_, 'bytes': count, 'sha256': hexdigest}
    atomic_json(receipt, record)
    return record


class HTTPRangeFile(io.RawIOBase):
    """Seekable HTTP object: refuses servers that return entire ZIPs for ranges.

    No speculative prefetch. zipfile requests only ZIP framing and selected
    members; unselected MP4 payloads are never downloaded in bulk.
    """
    def __init__(self, url_):
        self.url = url_; self.pos = 0; self.transferred = 0
        self.session = requests.Session()
        # Use a range GET rather than HEAD because some signed CDNs disallow HEAD.
        with self.session.get(transport_url(url_), headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},
                              stream=True, timeout=(20,90)) as r:
            r.raise_for_status()
            match = re.fullmatch(r'bytes 0-0/(\d+)', r.headers.get('Content-Range',''))
            if r.status_code != 206 or not match:
                raise ValueError('Server must support exact byte ranges; full download refused')
            self.size = int(match[1]); self.transferred += len(r.content)
        self._cache = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        new = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if new < 0: raise ValueError('Negative seek')
        self.pos = new; return new

    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if n <= 0: return b''
        start, end = self.pos, self.pos+n-1
        key = (start,end)
        if key in self._cache:
            data = self._cache[key]
        else:
            # Range is also in query to avoid intermediaries caching the wrong range.
            sep = '&' if '?' in self.url else '?'
            request_url = self.url + sep + f'jev_range={start}-{end}'
            with self.session.get(transport_url(request_url),
                                  headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                                  stream=True, timeout=(20,90)) as r:
                r.raise_for_status()
                if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                    raise ValueError('Incorrect HTTP Range response; full download refused')
                data = r.content
                if len(data) != n: raise ValueError('Truncated byte range')
            self.transferred += len(data)
            if n < 1024*1024: self._cache[key] = data
        self.pos += len(data)
        return data

    def close(self):
        self.session.close(); super().close()


def extract_selected_zip(url_, targets):
    """targets maps member basename to local path. zipfile verifies member CRC."""
    outputs = []
    pending = {}
    for name, target in targets.items():
        target = Path(target)
        receipt = target.with_suffix(target.suffix+'.receipt.json')
        if target.is_file() and receipt.is_file():
            old = json.loads(receipt.read_text())
            if (old.get('url') == url_ and old.get('member_basename') == name
                and target.stat().st_size == old['bytes'] and sha256(target) == old['sha256']):
                outputs.append(old); continue
        pending[name] = target
    if not pending: return outputs
    with HTTPRangeFile(url_) as remote, zipfile.ZipFile(remote) as archive:
        lookup = {}
        for info in archive.infolist():
            name = Path(info.filename).name
            if name in pending:
                if name in lookup: raise ValueError('Ambiguous ZIP basename')
                lookup[name] = info
        if set(lookup) != set(pending): raise ValueError('Selected video missing from ZIP')
        for name, target in pending.items():
            info = lookup[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_suffix(target.suffix+'.part')
            before = remote.transferred
            with archive.open(info) as source, part.open('wb') as f:
                shutil.copyfileobj(source, f, length=32*1024*1024)
            if part.stat().st_size != info.file_size: raise ValueError('ZIP member size mismatch')
            record = {'url':url_, 'member_basename':name, 'member':info.filename,
                      'bytes':info.file_size, 'sha256':sha256(part), 'crc32':info.CRC,
                      'compressed_bytes': info.compress_size,
                      'range_bytes':remote.transferred-before}
            part.replace(target); atomic_json(target.with_suffix(target.suffix+'.receipt.json'),record)
            outputs.append(record)
    return outputs
