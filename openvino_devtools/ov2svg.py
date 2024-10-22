import sys
import openvino as ov
import os

def set_graph_dumper_env(enable=True):
    import os
    #TODO: Make it configurable by arguments
    envs_to_set = [
        'OV_VISUALIZE_TREE_OUTPUT_SHAPES',
        'OV_VISUALIZE_TREE_OUTPUT_TYPES',
        'OV_VISUALIZE_TREE_EDGE_LABELS',
        'OV_VISUALIZE_PARTIAL_VALUES_AND_LABELS']
    for e in envs_to_set:
        os.environ[e] = '1' if enable else '0'

def serialize_model_svg (model, output_name='serialized_model.svg'):
    #TODO: Keep original values of envs and restore them in the end
    from openvino.runtime.passes import VisualizeTree
    set_graph_dumper_env(True)
    VisualizeTree(output_name).run_on_model(model)
    set_graph_dumper_env(False)

def set_symbols(model):
    for parameter in model.get_parameters():
        shape = parameter.get_output_partial_shape(0)
        new_dims = []
        has_dynamic = False
        for dim in shape:
            if dim.is_dynamic:
                dim.set_symbol(ov.Symbol())
                has_dynamic = True
            new_dims.append(dim)
        if has_dynamic:
            parameter.set_shape(ov.PartialShape(new_dims))

def ov2svg (input_name, output_name=None):
    core = ov.Core()
    extension = '.svg'
    model = core.read_model(input_name)
    set_symbols(model)
    model.validate_nodes_and_infer_types()  # to propagate shape in stateful models
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(input_name))[0] + extension
    elif not output_name.endswith(extension):
        output_name += extension
    serialize_model_svg(model, output_name)

def main():
    model_path = sys.argv[1]
    svg_path = sys.argv[2] if len(sys.argv) >= 3 else None
    ov2svg(model_path, svg_path)

if __name__ == "__main__":
    main()