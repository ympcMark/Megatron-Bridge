# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""H100 MAGI-2 training recipes."""

from megatron.bridge.recipes.magi2.h100.magi2 import (
    magi2_114b_pretrain_64gpu_h100_bf16_config,
)


__all__ = ["magi2_114b_pretrain_64gpu_h100_bf16_config"]
