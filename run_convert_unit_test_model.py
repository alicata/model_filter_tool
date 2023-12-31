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
    qtype = "bf16" # bf16 only supported, unsupported "fbgemm" or "qnnpack"
    quantization_enabled = parser.parse_args().compress


    torch_model, input_shape = get_model(torch_model_path)
    if quantization_enabled:
        print("--------- model compression enabled -----------")
        print(f"converting pytorch model to quantized format {qtype} ...")
        if qtype == "fbgemm":
            # quantize compress model to CPU-side int8
            torch_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(torch_model, inplace=True)
            torch.quantization.convert(torch_model, inplace=True)
        elif qtype == "qnnpack":
            # quantize compress model to CPU-side int8
            torch_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            torch.quantization.prepare(torch_model, inplace=True)
            torch.quantization.convert(torch_model, inplace=True)
        elif qtype == "bf16":
            # quantize compress model to GPU-side bf16
            torch_model = torch_model.to('cuda').to(dtype=torch.bfloat16)
            input_data = torch.randn(*input_shape).to('cuda').to(dtype=torch.bfloat16)
            torch.save(torch_model.state_dict(), torch_model_path.split(".pth")[0] + ".{qtype}.pth")
            with torch.no_grad():
                output = torch_model(input_data)    
        onnx_model_path = torch_model_path.split(".pth")[0] + "." + str(qtype) + ".quant.onnx"
    else:
        onnx_model_path = torch_model_path.split(".pth")[0] + ".onnx"
    torch_model.eval()

    # report device and model parameters
    device = next(torch_model.parameters()).device
    print(f"----------{device.type},{device}---------------------")

    for name, param in torch_model.named_parameters():
        print(f"Parameter: {name}, dtype: {param.dtype} bytes: {param.element_size() * param.numel()}")
    print("Parameters layers: ", sum(p.numel() for p in torch_model.parameters() if p.requires_grad))      
 
    print("-------------------------------------------------")
    print(f"converting input model {torch_model_path} to {onnx_model_path} ...")
    print("-------------------------------------------------")
    export_torch_to_onnx(torch_model, onnx_model_path, shape=input_shape, conversion_check_enabled=True)
else:
    print("error: no input model provided for convertion!")
    exit(0)

print("-------------------------------------------------")
print(f"completed creating model {onnx_model_path}!")


