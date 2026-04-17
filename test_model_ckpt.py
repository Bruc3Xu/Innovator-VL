
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from transformers import (
     AutoConfig, AutoProcessor,
     Qwen2Tokenizer
)
from transformers import logging
from huggingface_hub import hf_hub_download, snapshot_download


logging.set_verbosity_info()
logger = logging.get_logger(__name__)
CUDA_DEVICE = 0

from ds.merge_hybrid_model import  load_rice_vit_weights, load_siglip2_weights, load_dinov3_weights


if __name__ == "__main__":
    # load_siglip2_weights(None, "../models/siglip2-so400m-patch14-384")
    load_dinov3_weights(None, "../models/dinov3-vitl16-pretrain-lvd1689m")
    # load_rice_vit_weights(None, "../models/rice-vit-large-patch14-560")