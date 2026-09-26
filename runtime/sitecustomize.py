"""Install the opt-in mJev vLLM worker hooks in every subprocess."""

import os

if os.environ.get("MJEV_ENABLE_PATCHES") == "1":
    try:
        from mjev_vllm_patch import install
        install()
    except Exception as exc:
        raise SystemExit('mJev runtime hook startup failed; refusing to start an unpatched process') from exc
