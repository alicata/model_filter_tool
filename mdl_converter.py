import numpy as np
import torch.onnx
import onnx
import onnxruntime
import shutil

def test_inference(onnx_path):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("input : ", ort_inputs)
    print("output: ", ort_outs)
    return ort_outs

def load_onnx(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model

def export_torch_to_onnx(torch_model, onnx_model_path, shape=(1, 224, 224), conversion_check_enabled=False):
    # Input to the model, output a prediction
    print("eval mode: disable dropout and batchnorm")
    torch_model.eval()

    batch_size = 1    # just a random number
    x = torch.randn(batch_size, shape[0], shape[1], shape[2], requires_grad=True)
    torch_out = torch_model(x)

    print("export the model in onnx format using tracing method: ", onnx_model_path)
    torch.onnx.export(torch_model,             # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    onnx_model_path,           # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    if conversion_check_enabled:
        print("original model output: ", torch_out)
        # compare ONNX Runtime and PyTorch results
        ort_outs = test_inference(onnx_model_path)    
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")


"""
ModelSyncher: synchronize a model stored in cache / drive, load at start, save during updates
"""
class ModelSyncher:
    def __init__(self, drive_subfolder='data'):
       self.drive_folder = f"/content/drive/MyDrive/{drive_subfolder}"

    def start(self, model_path):
        self.model_path = model_path

    def update(self, model):   
        shutil.copy(self.model_path,  self.drive_folder + self.model_path)

    def stop(self):
        pass



