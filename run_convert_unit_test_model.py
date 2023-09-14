from mdl_converter import *
from trainers.trainer_simple import *

import torch
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_model', '-i', help='input torch model path')
parser.add_argument('--output_model', '-o', help='output onnx model path')
parser.add_argument('--compress', '-c', action='store_true', help='enable model compression')


os.makedirs("./output/", exist_ok=True)

def get_model(torch_model_path):
    """Create model from network definition and load weights into it

    Args:
        torch_model_path (str): path to the model weights

    Returns:
        model (nn.Module): network model loaded with the weights
        input_shape (tuple): input shape of the model

    """
    model = LogisticRegressionNetwork(input_dims=1)

    # Get input shape from the model
    input_shape = next(model.parameters()).shape
    assert(input_shape[0]==1)

    model.load_state_dict(torch.load(torch_model_path))
    print(model)
    return model, input_shape

print(f"---- {sys.argv[0]} ---- ")


if (len(sys.argv)) > 1:
    # get torch model path from command line argument --input_model
    args = parser.parse_args()
    torch_model_path = args.input_model

    torch_model, input_shape = get_model(torch_model_path)
    if parser.parse_args().compress:
        print("converting pytorch model to quantized model ...")
        torch_model = torch.quantization.convert(torch_model, inplace=True)
        onnx_model_path = torch_model_path.split(".pth")[0] + ".quant.onnx"
    else:
        onnx_model_path = torch_model_path.split(".pth")[0] + ".onnx"
    torch_model.eval()

    print(f"converting input model {torch_model_path} to {onnx_model_path} ...")
    export_torch_to_onnx(torch_model, onnx_model_path, shape=input_shape, conversion_check_enabled=True)
else:
    print("error: no input model provided for convertion!")
    exit(0)

print(f"{sys.argv[0]} completed.")


