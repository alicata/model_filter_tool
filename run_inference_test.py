from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import onnx
import onnxruntime
import os

os.makedirs("./output/", exist_ok=True)


def load_image_as_tensor():
    img = Image.open("./data/cat_224x224.jpg")
    resize = transforms.Resize([224, 224])
    img = resize(img)
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)
    return img_y, img_cb, img_cr


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def scale_color_and_merge_super_gray(img_out_y, img_cb, img_cr):
    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")
    return final_img


print("ONNX load model ...")
onnx_model = onnx.load("./models/model.super_resolution.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("./models/model.super_resolution.onnx")

img_y, img_cb, img_cr = load_image_as_tensor()

print("ONNX Runtime Inference: superresolution scaling of grayscale y channel")
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

merged_image = scale_color_and_merge_super_gray(img_out_y, img_cb, img_cr)

# Save the image, we will compare this with the output image from mobile device
merged_image.save("./output/output_cat_superres_with_ort.jpg")




