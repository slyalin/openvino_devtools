import sys
import openvino as ov
import os

def enable_graph_dumper(enable=True):
    import os
    #TODO: Make it configurable by arguments
    envs_to_set = [
        'OV_ENABLE_VISUALIZE_TRACING',
        'OV_VISUALIZE_TREE_OUTPUT_SHAPES',
        'OV_VISUALIZE_TREE_OUTPUT_TYPES',
        'OV_VISUALIZE_TREE_EDGE_LABELS',
        'OV_VISUALIZE_PARTIAL_VALUES_AND_LABELS']
    for e in envs_to_set:
        os.environ[e] = '1' if enable else '0'

def serialize_model_svg (model, output_name='serialized_model_svg'):
    #TODO: Keep original values of envs and restore them in the end

    from openvino.runtime.passes import Manager, GraphRewrite
    enable_graph_dumper(True)
    manager = Manager()
    manager.set_per_pass_validation(False)
    anchor = manager.register_pass(GraphRewrite())
    anchor.set_name(output_name)
    manager.run_passes(model)
    enable_graph_dumper(False)

def ov2svg (input_name, output_name=None):
    core = ov.Core()
    model = core.read_model(input_name)
    model.validate_nodes_and_infer_types()  # to propagate shape in stateful models
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(input_name))[0]
    serialize_model_svg(model, output_name)

def main():
    model_path = sys.argv[1]
    svg_path = sys.argv[2] if len(sys.argv) >= 3 else None
    ov2svg(model_path, svg_path)

if __name__ == "__main__":
    main()