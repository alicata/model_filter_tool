import torch
#import torchsummary
#from torchviz import make_dot
#import pickletools
import numpy as np

# Define the path to the saved .pth file
PATH = "d:/data/model/sam/sam_vit_b_01ec64.pth"
MODELFILE = 'sam_vit_b_01ec64.pth'
print("loading ", PATH)
model = torch.load(PATH)

input_size = model[list(model.keys())[0]].shape


# Print the model name
print("----------------------")
print("filepath            : ", MODELFILE)
print("----------------------")
print("number of layers : ", len(model.keys()))
print("input size       : ", input_size)
print("----------------------")
[print(l, t.shape) for l, t in model.items()]

def sum(v):
    #print("shape : ", v.shape)
    if (v is None):
        return 0
    return torch.sum(v.flatten())

# compute total checksum of all weights/biases

def checksum(m):
    s = [sum(v)  for k, v in m.items()]
    print("\ntotal checksum: ", np.sum(s))

checksum(model)

th = 10e-2 #300
print("filter out small weights: th=", th)

for k, v in model.items():
    model[k] = (torch.abs(v) > th)*v

checksum(model)


