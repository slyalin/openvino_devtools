# ov2py
Python tool that converts OpenVINO Model into a pretty-printed Python code that recreates the original model.
The resulting code could be used for easier (in comparison to `XML`) model exploration and modification.


## Installation

```console
pip install git+https://github.com/slyalin/openvino_devtools.git
```

## Usage

`ov2py` can be used in two ways: as a command line utility `ov2py` or as a Pytnon module `openvino_devtools.ov2py`.

### Command Line

When used in the command line, it accepst a path to OpenVINO IR `XML` file and prints the resulting Python code to stdout:

```console
$ ov2py your_openvino_model.xml
```

will print something similar to:

```python
def build_model(model):
    import openvino as ov
    from openvino_devtools.builder import OpFactory, outputs_to_nodes
    opset = OpFactory('opset13')

    t0 = opset.Parameter({'shape': [-1, -1], 'element_type': 'f32'}, node_name="a", output_names=[{'a'}])  #  -> f32[?,?]
    t1 = opset.Add([t0, t0], {'auto_broadcast': 'numpy'}, node_name="aten::add/Add", output_names=[{'8'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t2 = opset.Multiply([t0, t0], {'auto_broadcast': 'numpy'}, node_name="aten::mul/Multiply", output_names=[{'3'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t3 = opset.Constant(model, 3, node_name="Constant_1730_compressed")  #  -> f16[1,1]([[2.0]])
    t4 = opset.Convert([t3], {'destination_type': 'f32'}, node_name="Constant_1730")  # f16[1,1] -> f32[1,1]
    t5 = opset.Add([t2, t4], {'auto_broadcast': 'numpy'}, node_name="aten::add/Add_1", output_names=[{'6', 'b'}])  # f32[?,?], f32[1,1] -> f32[?,?]
    t6 = opset.Subtract([t0, t5], {'auto_broadcast': 'numpy'}, node_name="aten::sub/Subtract", output_names=[{'10'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t7 = opset.Multiply([t1, t6], {'auto_broadcast': 'numpy'}, node_name="aten::mul/Multiply_1", output_names=[{'11'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t8 = opset.Divide([t7, t0], {'auto_broadcast': 'numpy', 'm_pythondiv': True}, node_name="aten::div/Divide", output_names=[{'12', 'input'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t9 = opset.Constant(model, 9, node_name="self.linear.weight_compressed", output_names=[{'self.linear.weight'}])  #  -> f16[128,1024]
    t10 = opset.Convert([t9], {'destination_type': 'f32'}, node_name="self.linear.weight")  # f16[128,1024] -> f32[128,1024]
    t11 = opset.MatMul([t8, t10], {'transpose_a': False, 'transpose_b': True}, node_name="__module.linear/aten::linear/MatMul")  # f32[?,?], f32[128,1024] -> f32[?,128]
    t12 = opset.Constant(model, 12, node_name="Constant_1731_compressed")  #  -> f16[1,128]
    t13 = opset.Convert([t12], {'destination_type': 'f32'}, node_name="Constant_1731")  # f16[1,128] -> f32[1,128]
    t14 = opset.Add([t11, t13], {'auto_broadcast': 'numpy'}, node_name="__module.linear/aten::linear/Add")  # f32[?,128], f32[1,128] -> f32[?,128]
    t15 = opset.Result([t14], {}, node_name="Result_44")        # f32[?,128] -> f32[?,128]

    parameters = [t0]
    results = [t15]
    sinks = []
    return ov.Model(outputs_to_nodes(results), outputs_to_nodes(sinks), outputs_to_nodes(parameters), name=model.get_friendly_name())


if __name__ == '__main__':
    from openvino_devtools.builder import regenerate_model
    regenerate_model(build_model, path_to_source_model='your_openvino_model.xml')
```

The resulting code can be forwarded to a file and then executed:

```console
$ ov2py your_openvino_model.xml > generator.py
$ python generator.py name_of_destination.xml
```

The result is OpenVINO IR saved into `name_of_destination.xml` and `name_of_destination.bin` files with re-generated original model.
You can easily modify `generator.py` before running the script as it is a regular Python code.

### Python Module

Using the module in Python is useful for debugging, especially in Jupyter Notebooks when you can generate the model as Python text,
copy a full output or a part of it and insert in the next cell to generate a desired part as ov.Model.

```python
from openvino_devtools.ov2py import ov2py
# ov_model is an instance of ov.Model, usually the result of ov.Core.read_model/ov.convert_model
print(ov2py(ov_model))
```

which will print the code similar to what `ov2py` prints but without entry point and without names for nodes to reduce the amount of output:

```python
def build_model(model):
    import openvino as ov
    from openvino_devtools.builder import OpFactory, outputs_to_nodes
    opset = OpFactory('opset13')

    t0 = opset.Parameter({'shape': [-1, -1], 'element_type': 'f32'}, output_names=[{'a'}])  #  -> f32[?,?]
    t1 = opset.Add([t0, t0], {'auto_broadcast': 'numpy'}, output_names=[{'8'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t2 = opset.Multiply([t0, t0], {'auto_broadcast': 'numpy'}, output_names=[{'3'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t3 = opset.Constant(model, 3)                               #  -> f32[1,1]([[2.0]])
    t4 = opset.Add([t2, t3], {'auto_broadcast': 'numpy'}, output_names=[{'b', '6'}])  # f32[?,?], f32[1,1] -> f32[?,?]
    t5 = opset.Subtract([t0, t4], {'auto_broadcast': 'numpy'}, output_names=[{'10'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t6 = opset.Multiply([t1, t5], {'auto_broadcast': 'numpy'}, output_names=[{'11'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t7 = opset.Divide([t6, t0], {'auto_broadcast': 'numpy', 'm_pythondiv': True}, output_names=[{'12', 'input'}])  # f32[?,?], f32[?,?] -> f32[?,?]
    t8 = opset.Constant(model, 8, output_names=[{'self.linear.weight'}])  #  -> f32[1,10]
    t9 = opset.MatMul([t7, t8], {'transpose_a': False, 'transpose_b': True})  # f32[?,?], f32[1,10] -> f32[?,1]
    t10 = opset.Constant(model, 10)                             #  -> f32[1,1]([[0.009999999776482582]])
    t11 = opset.Add([t9, t10], {'auto_broadcast': 'numpy'})     # f32[?,1], f32[1,1] -> f32[?,1]
    t12 = opset.Result([t11], {})                               # f32[?,1] -> f32[?,1]

    parameters = [t0]
    results = [t12]
    sinks = []
    return ov.Model(outputs_to_nodes(results), outputs_to_nodes(sinks), outputs_to_nodes(parameters), name=model.get_friendly_name())
```

Both node names and entry point can be printed as well by providing additional arguments:

```python
print(ov2py(ov_model, with_node_names=True, entry_point=True))
```

## Limitations
The following features of OpenVINO model are not supported:
- Custom operations.
- Arbitrary standard opsets (now it is fixed to `opset13`).
- Run-time info of any kind: in at a node level and at a model level (meta information).
- Control flow operations.
- Original model is always required to run `build_model` for `Constant` ops content (when the re-generated OpenVINO model is saved to IR, the orignal model is no longer needed).
