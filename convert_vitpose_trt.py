#!/usr/bin/env python3
"""
convert_vitpose_trt.py

Compile a ViTPose checkpoint into a TensorRT-backed TorchScript module.
Later you can reload it with:

    model = torch.jit.load("models/vitpose_h_wholebody.ts").eval().cuda()
"""

import argparse
import os
import torch
import torch_tensorrt as trt

from easy_ViTPose.vit_models.model import ViTPose
from easy_ViTPose.vit_utils.util import infer_dataset_by_path, dyn_model_import


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True,
                   help="Path to the .pth file to compile")
    p.add_argument("--model-name", required=True, choices=["s", "b", "l", "h"],
                   help="[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]")
    p.add_argument("--output", default="models/trt",
                   help="Directory for the compiled file")
    p.add_argument("--dataset", default=None,
                   help='Dataset tag; leave blank to infer from filename')
    p.add_argument("--fp16", action="store_true",
                   help="Compile for FP16 inference (GPU must support it)")
    return p.parse_args()


def load_vitpose(ckpt_path: str, model_name: str, dataset: str) -> ViTPose:
    model_cfg = dyn_model_import(dataset, model_name)
    model = ViTPose(model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(ckpt, strict=False)

    model.eval().cuda()
    return model


def main():
    args = parse_args()

    dataset = args.dataset or infer_dataset_by_path(args.model_ckpt)
    valid = ["mpii", "coco", "coco_25", "wholebody", "aic", "ap10k", "apt36k"]
    if dataset not in valid:
        raise ValueError(f"dataset must be one of {valid}")

    model = load_vitpose(args.model_ckpt, args.model_name, dataset)

    example = torch.randn(1, 3, 256, 192, device="cuda")
    precisions = {torch.float16} if args.fp16 else {torch.float32}

    trt_mod = trt.compile(
        model,
        ir="dynamo",                           # Dynamo front-end
        inputs=[trt.Input(example.shape, dtype=list(precisions)[0])],
        enabled_precisions=precisions,
        workspace_size=1 << 28
    )

    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(
        args.output,
        os.path.basename(args.model_ckpt).replace(".pth", ".ts")
    )

    # trt.save(trt_mod, out_file,
    #          input_samples=[example],
    #          output_format="torchscript")
    
    trt.save(trt_mod, out_file, inputs=[example],
    output_format="torchscript")

    print(f"compiled module written to: {out_file}")


if __name__ == "__main__":
    main()
