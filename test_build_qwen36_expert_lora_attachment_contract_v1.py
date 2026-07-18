from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def test_builder_import_does_not_import_cuda_capable_model_stack():
    code = """
import sys
import build_qwen36_expert_lora_attachment_contract_v1 as builder
for name in ('torch', 'transformers', 'peft', 'qwen36_expert_lora_v1'):
    assert name not in sys.modules, name
assert builder.ARCHITECTURE_CONTRACT.name == 'qwen36_architecture_contract_v1.json'
"""
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
