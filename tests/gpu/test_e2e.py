"""Full official-model vLLM test; explicitly selected, never a CPU smoke test."""
import os
from pathlib import Path
import runpy


def test_official_model_e2e():
    assert os.environ.get('MJEV_MODEL'), 'Set MJEV_MODEL to official local weights'
    assert os.environ.get('MJEV_ENABLE')=='1', 'Use scripts/test_gpu.sh for hook bootstrap'
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'e2e.py'),run_name='__main__')
