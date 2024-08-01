import openvino as ov
from openvino_devtools import builder
import sys

def ov2py(model, with_node_names=False, path_to_source_model=None, entry_point=False):
    printer = builder.ModelPrinter()
    return printer.model_to_python(model, with_node_names=with_node_names, path_to_source_model=path_to_source_model, entry_point=entry_point)

def main():
    model_path = sys.argv[1]
    core = ov.Core()
    model = core.read_model(model_path)
    print(ov2py(model, with_node_names=True, path_to_source_model=model_path, entry_point=True), '\n')

if __name__ == "__main__":
    main()