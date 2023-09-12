# MDL - A Model filtering tool
ML utility to analyze, filter, structurally validate and cleanup DL models checkpoints. 

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
### Scripts To Support Validation Tasks
| Script          | Description   | Netron model visualization   | Implemented |
| ---------------- |:-------------|:-------------| :-----:|
| ./run_generate_unit_test_model.py| Generate simple [1d logistic regression model](https://github.com/alicata/model_filter_tool/blob/main/trainers/trainer_simple.py#L8) to aid unit testing | https://netron.app/?url=https://github.com/alicata/model_filter_tool/blob/main/models/unit_test_model.pth | Y |
| ./run_inference_test.py | Exercise ONNX RT inference on the super resolution model | https://netron.app/?url=https://github.com/alicata/model_filter_tool/blob/main/models/model.super_resolution.onnx |Y |

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

# Model Cards
| Model Name       | Identifier  | Description   |  Genesis | Netron model visualization   |
| ---------------- | :------------- | :------------- | :------------- | :-----: |
| unit_test_model.pth | [LogisticRegression](https://github.com/alicata/model_filter_tool/blob/main/trainers/trainer_simple.py#L8) | 1D logistic regression model binary classifier | trained with synthetic data by AutoGeneratorTrainer | https://netron.app/?url=https://github.com/alicata/model_filter_tool/blob/main/models/unit_test_model.pth |


# Limitations
Currently the tool supports only the .pth format. More formats are planned. 


## Validation Workflow 1: Convert and Validate Auto Generated Unit Test Model. 
| Step        | Description   | Expected Result |
| ---------------- |:-------------| :-----:|
| ./run_generate_unit_test_model.py | Generate PyTorch simple 1d input, one layer, unit test model | unit_test_model.pth saved in output folder |
| ./run_convert_unit_test_model.py | Convert and validate unit test model from PyTorch to ONNX graph format | unit_test_model.onnx saved in output folder|
| Visually inspect graph structure | Visualize ONNX graph structure with Netron web app | generated and reference unit test models show same [ONNX graph in Netron](https://netron.app/?url=https://github.com/alicata/model_filter_tool/blob/main/models/unit_test_model.onnx)|
