# MDL - A Model filtering tool
ML utility to analyze, filter and cleanup models checkpoints. 

The `mdl_inspect` tool loads a model scan its structure and identifies and filters out weights to match a given treshold requirement. 

## Setup

Clone the repo, and install dependencies within the repo folder:

```bash
git clone https://github.com/alicata/model_filter_tool.git 
cd model_filter_tool
pip install -r requirements
```

## MDL Usage
| Command          | Description   | Implemented |
| ---------------- |:-------------| :-----:|
| ./mdl_inspect.py model.pth | Inspect structure of the model checkpoint | Y |
| ./mdl_inspect.py -f 0.01 model.pth | supress all weightes below 0.01 in the model checkpoint | N |


## Utilities
### Scripts
| Script          | Description   | Implemented |
| ---------------- |:-------------| :-----:|
| ./run_generate_unit_test_model.py| Generate simple 1d logistic regression model to aid unit testing | Y |
| ./run_inference_test.py | Exercise ONNX RT inference on the super resolution model | Y |

### Jupyter Notebook
Synchronize model in nb to/from Drive, for persistent caching of model.
This utility is useful when tracking experimenting with various filtering modifications to the model.
```
!pip install https://github.com/alicata/model_filter_tool/master

sync = ModelSyncher()
sync.start('./model_under_study.pth')
...
# train or change model
...
sync.update()

```

## Limitations
Currently the tool supports only the .pth format. More formats are planned. 

