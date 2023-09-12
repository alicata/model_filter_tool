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

def validate_model_inference(onnx_path, torch_out):
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("input : ", ort_inputs)
    print("output: ", ort_outs)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def export_torch_to_onnx(torch_model, onnx_model_path, shape=(3, 224, 224), conversion_check_enabled=False):
    # Input to the model, output a prediction
    print("switch model to eval mode (disable dropout and batchnorm)")
    torch_model.eval()

    # Generate a batch with 1 random sample, and given shape
    #x = torch.randn(1, shape[0], shape[1], shape[2], requires_grad=True)
    x = torch.randn(1, *shape, requires_grad=True)

    print("export the model in onnx format using tracing method: ", onnx_model_path)
    torch.onnx.export(torch_model,             # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    onnx_model_path,           # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}) # batch size can change

    if conversion_check_enabled:
        torch_out = torch_model(x)
        print("original model output: ", torch_out)
        # compare ONNX Runtime and PyTorch results
        validate_model_inference(onnx_model_path, torch_out)    

