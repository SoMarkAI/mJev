"""HTTP media boundary tests require neither vLLM nor a GPU/network."""
import base64
from io import BytesIO
import socket
import pytest
import mjev_media as media


def test_local_paths_symlinks_and_file_hosts(tmp_path):
    root=tmp_path/'allowed';root.mkdir()
    (root/'ok.png').write_bytes(b'ok')
    outside=tmp_path/'secret';outside.write_bytes(b'outside')
    (root/'link').symlink_to(outside)
    assert media.read_media(str(root/'ok.png'),limit=8,roots=[root])[0]==b'ok'
    for url in [str(outside),str(root/'../secret'),str(root/'link'),'file://remote'+str(root/'ok.png')]:
        with pytest.raises(ValueError):media.read_media(url,limit=8,roots=[root])


def test_local_size_checked_before_read_and_special_files_rejected(tmp_path,monkeypatch):
    p=tmp_path/'large';p.write_bytes(b'x'*100)
    monkeypatch.setattr(media,'bounded_read',lambda *args:pytest.fail('Oversized file must not be read'))
    with pytest.raises(media.MediaTooLarge):media.read_media(str(p),limit=8,roots=[tmp_path])
    import os
    fifo=tmp_path/'pipe';os.mkfifo(fifo)
    with pytest.raises(ValueError,match='regular file'):media.read_media(str(fifo),limit=8,roots=[tmp_path])


def test_data_limit_precedes_decoding(monkeypatch):
    assert media.read_media('data:image/png;base64,'+base64.b64encode(b'abcd').decode(),limit=4,roots=[])[0]==b'abcd'
    assert media.read_media('data:,%61%62',limit=2,roots=[])[0]==b'ab'
    monkeypatch.setattr(media.base64,'b64decode',lambda *a,**k:pytest.fail('Must reject before decoding'))
    with pytest.raises(media.MediaTooLarge):media.read_media('data:;base64,'+'A'*100,limit=4,roots=[])
    with pytest.raises(media.MediaTooLarge):media.read_media('data:,'+'%41'*100,limit=4,roots=[])


def test_remote_default_denied_and_private_dns_blocked(monkeypatch):
    with pytest.raises(ValueError,match='allowed HTTPS'):media.read_media('https://example.org/x',limit=4,roots=[])
    with pytest.raises(ValueError):media.read_media('http://example.org/x',limit=4,roots=[],allowed_hosts=['example.org'])
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1',443))])
    with pytest.raises(ValueError,match='public addresses'):media.read_media('https://example.org/x',limit=4,roots=[],allowed_hosts=['example.org'])


@pytest.mark.parametrize('status,length',[(302,None),(200,'100')])
def test_remote_redirect_and_declared_size_rejected_before_read(monkeypatch,status,length):
    from email.message import Message
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))])
    class Response:
        headers=Message()
        def getheader(self,key):return length
        def read(self,*a):pytest.fail('Must not read body')
    Response.status=status
    class Connection:
        def __init__(self,host,address):assert address=='8.8.8.8'
        def request(self,*a,**k):pass
        def getresponse(self):return Response()
        def close(self):pass
    monkeypatch.setattr(media,'PinnedHTTPSConnection',Connection)
    with pytest.raises(ValueError):media.read_media('https://example.org/x',limit=4,roots=[],allowed_hosts=['example.org'])


def test_unknown_length_read_is_bounded():
    stream=BytesIO(b'123456789')
    with pytest.raises(media.MediaTooLarge):media.bounded_read(stream,4)
    assert stream.tell()==5


def test_outside_path_is_rejected_before_open(tmp_path,monkeypatch):
    root=tmp_path/'allowed';root.mkdir()
    outside=tmp_path/'outside';outside.write_bytes(b'secret')
    monkeypatch.setattr(media.os,'open',lambda *a,**k:pytest.fail('Disallowed path must not be opened'))
    with pytest.raises(ValueError,match='outside allowed'):media.read_media(str(outside),limit=100,roots=[root])
