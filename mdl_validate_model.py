import onnx
import sys

if len(sys.argv) > 1:
    model_name = sys.argv[1]
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)
else:
    print("please provide model name")

