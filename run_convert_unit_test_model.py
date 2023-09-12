from mdl_converter import *
from trainers.trainer_simple import *

import torch
import os
import sys


os.makedirs("./output/", exist_ok=True)

def get_model(torch_model_path):
    """
    get_model create network definition and loads weights into it

    NOTE: for now tested and validated only with the unit test model 1d logistic regression
    """

    # TODO: input dimenions and shape should be dynamically retreived from the model structure
    input_dims = 1

    model = LogisticRegressionNetwork(input_dims)
    input_shape = next(model.parameters()).shape
    assert(input_shape[0]==input_dims)

    model.load_state_dict(torch.load(torch_model_path))
    print(model)
    return model, input_shape

print(f"---- {sys.argv[0]} ---- ")

if (len(sys.argv)) > 1:
    torch_model_path = sys.argv[1]
    onnx_model_path = torch_model_path.split(".pth")[0] + ".onnx"

    torch_model, input_shape = get_model(torch_model_path)

    print(f"converting input model {torch_model_path} to {onnx_model_path} ...")
    export_torch_to_onnx(torch_model, onnx_model_path, shape=input_shape, conversion_check_enabled=True)
else:
    print("error: no input model provided for convertion!")
    exit(0)

print(f"{sys.argv[0]} completed.")


