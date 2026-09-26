import os
if os.environ.get('MJEV_ENABLE') == '1':
    # Python normally swallows Exception from sitecustomize and continues.
    # An opted-in worker must not start with absent or partial attention hooks.
    try:
        from mjev.patch import install
        install()
    except Exception as exc:
        raise SystemExit('mJev hook startup failed; refusing to start an unpatched process') from exc
