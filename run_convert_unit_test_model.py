from mdl_converter import *
from trainers.trainer_simple import AutoGeneratorTrainer

import torch
import os
import sys


os.makedirs("./output/", exist_ok=True)

def get_network_definition():
    return AutoGeneratorTrainer().model

print(f"---- {sys.argv[0]} ---- ")

if (len(sys.argv)) > 1:
    torch_model_path = sys.argv[1]
    torch_model = get_network_definition()
    onnx_model_path = torch_model_path.split(".pth")[0] + ".onnx"

    print(f"converting input model {torch_model_path} to {onnx_model_path} ...")
    torch_model.load_state_dict(torch.load(torch_model_path))
    export_torch_to_onnx(torch_model, onnx_model_path, shape=(1,), conversion_check_enabled=False)
else:
    print("error: no input model provided for convertion!")
    exit(0)

print(f"{sys.argv[0]} completed.")


