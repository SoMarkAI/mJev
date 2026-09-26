"""CPU API integration checks in the optional pinned vLLM environment."""
import asyncio
import pytest
pytest.importorskip('vllm',reason='experimental server imports optional vLLM')
from fastapi import FastAPI
from fastapi.testclient import TestClient
import mjev_server as server
import mjev_vllm_patch as patch


def test_routes_and_hook_readiness(monkeypatch):
    ready=[True]
    def readiness():
        if not ready[0]:raise RuntimeError('missing hooks')
        return {'all':True}
    monkeypatch.setattr(patch,'require_installed',readiness)
    app=FastAPI();server.attach_router(app)
    assert sum(r.path=='/v1/mjev/multi_question' for r in app.routes)==1
    with TestClient(app) as client:
        assert client.get('/v1/mjev/health').status_code==200
        ready[0]=False
        assert client.get('/v1/mjev/health').status_code==503


def test_normalizer_passes_validated_bytes_not_local_path(tmp_path,monkeypatch):
    import base64
    p=tmp_path/'image.png';p.write_bytes(b'owned media')
    monkeypatch.setenv('MJEV_ALLOWED_MEDIA_ROOTS',str(tmp_path))
    parts,stats=asyncio.run(server.normalize_media([server.MediaItem(type='image_url',url=str(p))]))
    url=parts[0]['image_url']['url']
    assert url.startswith('data:image/png;base64,')
    assert base64.b64decode(url.split(',',1)[1])==b'owned media'
    assert stats['bytes']==len(b'owned media')
