# Tools

This repository includes several tools:
- ov2py
- ovdiff
- ov2svg
- ovstat

They all are available as command line scripts after installation. See usage details for each tool below.

## Installation

You need to have OpenVINO installed, including OpenVINO Python API. The installation script of the tool doesn't install it.

```console
pip install git+https://github.com/slyalin/openvino_devtools.git
```

# ov2py
Python tool that converts [OpenVINO](https://github.com/openvinotoolkit/openvino) Model into a pretty-printed Python code that recreates the original model.
The resulting code can be used for easier (in comparison to `XML`) model exploration and modification.


## Usage

`ov2py` can be used in two ways: as a command line utility `ov2py` or as a Python module `openvino_devtools.ov2py`.

### Command Line

When used in the command line, it accepts a path to OpenVINO IR `XML` file and prints the resulting Python code to stdout:

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

Note that the resulting Python code uses `OpFactory` from `openvino_devtools.builder` instead of the original [OpenVINO Python API](https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.runtime.opset13.html) to create operations.
Using `OpFactory` as a wrapper for OpenVINO Python API unlocks an easy way to override the original operation-building functionality by
replacing `OpFactory` with a custom factory class.
When modifying the generated code, you can mix the original Python API with `OpFactory` calls.

While the main purpose of `ov2py` is to print out OpenVINO models, as a bonus functionality it accepts all [supported in OpenVINO model file formats](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html)
that can be loaded with `ov.Core.read_model`. For example, having ONNX model you can run:

```console
$ ov2py your_onnx_model.onnx
```

and `ov2py` converts the model from a source format to OpenVINO model with `ov.Core.read_model` and print it out.
This functionality is convenient to explore a model compute graph in a uniform way using OpenVINO operation set without the knowledge of the original model format.

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

You can compare two models by running `diff` on the listings generated by `ov2py`.
The autogenerated output names (i.e., the `output_names=...` part) can make such comparisons harder.
The `with_tensor_names` option in `ov2py` can be used to omit the generation of output names.
To print program listings most suitable for side-by-side comparison use the following:

```python
print(ov2py(ov_model, with_tensor_names=False))
```

## Limitations
The following features of OpenVINO model are not supported:
- Custom operations.
- Arbitrary standard opsets (now it is fixed to `opset13`).
- Run-time info of any kind: at the node level and at the model level (meta information).
- Control flow operations.
- Original model is always required to run `build_model` for `Constant` ops content (when the re-generated OpenVINO model is saved to IR, the orignal model is no longer needed).

# ovdiff
Compares XML files in two directories or two individual XML files and reports differences in <layer> tags.
It scans the directories, collects operation counts from the XML files, and prints any differences in the operations count between the reference and target files.
The script also includes an option to filter out files containing 'tokenizer' in their names.

Usage:
```console
$ ovdiff [--filter-tokenizer | --no-filter-tokenizer] <reference_directory> <target_directory>
```

Output example:
```console
$ ovdiff /path/to/reference/folder /path/to/target/folder

Reference: /path/to/reference/folder
Target: /path/to/target/folder
Filter tokenizer files: True
====================================================================================================
Diff: openvino_model.xml
Op difference detected
op_name                     ref   target    r-t
-------------------------------------------------
Add                           228     197      31
Const                        2851    2628     223
Gather                        361     296      65
ShapeOf                       230     166      64
Slice                         162     131      31
-------------------------------------------------
Total                        7040    6626     414
.bin file sizes are equal: 8035958816 bytes
====================================================================================================
```


# ov2svg

Generates visual SVG representation of a given OpenVINO model. Requires `graphviz` `dot` command line utility to produce an SVG file.
Leverages builtin OpenVINO capabilities implemented in `VisualizeTree` transformation.
Enables additional annotations like tensor shapes, types and symbols.

Usage:
```console
$ ov2svg <model.xml> [ <output_file.svg> ]
```

If `<output_file.svg>` is not provided then `model.svg` will be used as output name, where `model` is a part of input file name.
If `<output_file.svg>` has no `.svg` extension, it will be added.
The tool overrides existing output file without a warning.


# ovstat

Gives the aggregated statistics on operations used in the model alongside with input and output tensor types and attribute values.
The aggregation group together all static dimensions and denote it as `S`.
Similarly, it aggregates all dynamic dimensions and denote them as `?`.
Shape-like operation attributes are aggregated in the same way.
Attribute `variable_id` values in operations `ReadValue` and `Assign` are aggregated as well and denoted as `S`.
Number of operations in each group is printed at the end of the line, after the tabulation symbol.

Usage:

```console
$ ovstat <model.xml>
```

Sample output:

```console
Add { auto_broadcast = numpy } : i64[S], i64[S] -> i64[S]       33
Add { auto_broadcast = numpy } : f32[?,?,S], f32[?,?,S] -> f32[?,?,S]   64
Add { auto_broadcast = numpy } : i32[], i32[S] -> i32[S]        32
Assign { variable_id = S } : f32[?,S,?,S] -> f32[?,S,?,S]       64
Broadcast { mode = numpy } : f32[], i64[S] -> f32[?,S,S,S]      64
Broadcast { mode = bidirectional } : f32[S,S,S], i64[S] -> f32[?,S,S]   1
Concat { axis = 0 } : i64[S], i64[S], i64[S], i64[S] -> i64[S]  66
Concat { axis = 0 } : i64[S], i64[S] -> i64[S]  2
Constant { element_type = f32, shape = [] } :  -> f32[] 70
Constant { element_type = i64, shape = [S] } :  -> i64[S]       1358
Constant { element_type = i64, shape = [] } :  -> i64[] 116
Parameter { shape = [?], element_type = i32 } :  -> i32[?]      1
Parameter { shape = [?,?], element_type = i64 } :  -> i64[?,?]  3
ReadValue { variable_id = S, variable_type = f32, variable_shape = [?,S,?,S] } : f32[?,S,S,S] -> f32[?,S,?,S]   64
Result {  } : f32[?,?,S] -> f32[?,?,S]  1
```