import numpy as np
import torch.onnx
import onnx
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_onnx(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model

def get_model_quantization_type(torch_model):
    return next(torch_model.parameters()).dtype

def validate_model_inference(onnx_path, torch_out, x):
    print(f"val: load onnx model {onnx_path} into memory ...")
    try:
        ort_session = onnxruntime.InferenceSession(onnx_path)
        print("val: prepare input for ONNXRuntime inference")
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x.to('cuda').to(dtype=torch.float32))}
        ort_outs = ort_session.run(None, ort_inputs)
        print("val: ort inputs : ", ort_inputs)
        print("val: ort outputs: ", ort_outs)
        try:
            np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
            print("val: Exported model tested ONNXRuntime: prediction results all close to original. OK")
        except AssertionError as e:
            print("val: Exported model tested ONNXRuntime: prediction results not close to original. FAIL")
            print(e)
    except Exception as e:
        print("     [VALIDATION ERROR] : Exported but onnx file was not validated because ORT cannot complete.")
        print("    ", e)


def export_torch_to_onnx(torch_model, onnx_model_path, shape=(3, 224, 224), conversion_check_enabled=False):
    # Input to the model, output a prediction
    print("export: switch model to eval mode (disable dropout and batchnorm)")

    # Generate a batch with 1 random sample, and given shape
    x = torch.randn(1, *shape, requires_grad=True)
    device = next(torch_model.parameters()).device

    if device.type == 'cuda':
        print("export: model is on GPU, convert input to cuda tensor")
        x = x.to('cuda')
        if get_model_quantization_type(torch_model) == torch.bfloat16: 
            x = x.to(dtype=torch.bfloat16)  

    print("export: convert the model in onnx format using tracing method: ", onnx_model_path)
    torch.onnx.export(torch_model,             # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    onnx_model_path,           # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}) # batch size can change

    if conversion_check_enabled:
        print(f"---- exported model validation ----")
        torch_out = torch_model(x)
        print("val: original model output: ", torch_out)
        # compare ONNX Runtime and PyTorch results
        validate_model_inference(onnx_model_path, torch_out, x)
        print("val: model validation completed.")
