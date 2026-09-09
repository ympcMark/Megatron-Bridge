#!/usr/bin/env python3
"""Launch MAGI-2 without importing unrelated eager Bridge registries."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType


repo = Path(__file__).resolve().parents[3]
src = repo / "src"
sys.path.insert(0, str(repo / "3rdparty" / "Megatron-LM"))
sys.path.insert(0, str(src))

for package, path in (
    ("megatron.bridge", src / "megatron" / "bridge"),
    ("megatron.bridge.models", src / "megatron" / "bridge" / "models"),
    (
        "megatron.bridge.models.hf_pretrained",
        src / "megatron" / "bridge" / "models" / "hf_pretrained",
    ),
    (
        "megatron.bridge.models.conversion",
        src / "megatron" / "bridge" / "models" / "conversion",
    ),
    ("megatron.bridge.data", src / "megatron" / "bridge" / "data"),
    ("megatron.bridge.diffusion", src / "megatron" / "bridge" / "diffusion"),
    (
        "megatron.bridge.diffusion.models",
        src / "megatron" / "bridge" / "diffusion" / "models",
    ),
    ("megatron.bridge.recipes", src / "megatron" / "bridge" / "recipes"),
):
    module = ModuleType(package)
    module.__path__ = [str(path)]
    sys.modules[package] = module

# ConfigContainer imports these names for legacy typing. Importing the complete
# model registry would load optional Transformers architectures that are not part
# of MAGI-2 and are absent from the pinned NeMo container.
models_package = sys.modules["megatron.bridge.models"]
models_package.GPTModelProvider = type("GPTModelProvider", (), {})
models_package.T5ModelProvider = type("T5ModelProvider", (), {})

if __name__ == "__main__":
    entrypoint = runpy.run_path(
        str(Path(__file__).with_name("pretrain_magi2.py")),
        run_name="_magi2_pretrain_entrypoint",
    )
    entrypoint["main"]()
