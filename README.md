# Model filtering tool
ML utility to analyze, filter and cleanup models checkpoints. 

The `mdl_inspect` tool loads a model scan its structure and identifies and filters out weights to match a given treshold requirement. 

## Setup

Clone the repo, and install dependencies within the repo folder:

```bash
git clone https://github.com/alicata/model_filter_tool.git 
cd model_filter_tool
pip install -r requirements
```

## Usage
| Command          | Description   | Implemented |
| ---------------- |:-------------| :-----:|
| ./mdl_inspect.py model.pth | Inspect structure of the model checkpoint | Y |
| ./mdl_inspect.py -f 0.01 model.pth | supress all weightes below 0.01 in the model checkpoint | N |


## Limitations
Currently the tool supports only the .pth format. More formats are planned. 

