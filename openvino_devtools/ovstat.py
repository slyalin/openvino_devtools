import argparse
from collections import defaultdict
from pathlib import Path

import openvino as ov


def obfuscate_shape(shape, obfuscate):
    if shape.rank.is_dynamic:
        return '[...]'
    result = []
    for d in shape:
        result.append('?' if d.is_dynamic else 'S' if obfuscate else str(d))
    return '[' + ','.join(result) + ']'


def obfuscated_signature(op, obfuscate):
    def port_representation(port):
        return port.get_element_type().get_type_name() + obfuscate_shape(port.get_partial_shape(), obfuscate)

    input_types = ', '.join(map(port_representation, op.inputs()))
    output_types = ', '.join(map(port_representation, op.outputs()))
    return input_types + ' -> ' + output_types


def obfuscate_attributes(op, obfuscate):
    result = []
    for k, v in op.get_attributes().items():
        if isinstance(v, ov.PartialShape):
            v = obfuscate_shape(v, obfuscate)
        elif isinstance(v, list):
            if k in ['shape', 'variable_shape']:
                v = '[' + ','.join('?' if d == -1 else 'S' if obfuscate else str(d) for d in v) + ']'
            else:
                v = '[' + ','.join(['S'] * len(v)) + ']'
        elif k == 'variable_id' and op.get_type_info().name in ['ReadValue', 'Assign']:
            v = 'S'
        result.append((k, v))
    return '{ ' + ', '.join([k + ' = ' + str(v) for k, v in result]) + ' }'


def op_statistics(model, stat, obfuscate):
    ops = model.get_ordered_ops()
    local_ops = defaultdict(int)
    for op in ops:
        key = (op.get_type_info().name, obfuscated_signature(op, obfuscate), str(obfuscate_attributes(op, obfuscate)))
        # stat[key] += 1
        local_ops[key] += 1
    for key in local_ops:
        stat[key][0] += 1
        stat[key][1] += local_ops[key]
    return stat


def ovstat(path: str, obfuscate: bool):
    core = ov.Core()
    stat = defaultdict(lambda: [0, 0])
    model = core.read_model(path)
    op_statistics(model, stat, obfuscate)
    keys = list(stat)
    keys.sort(key=lambda x: x[0] + x[2] + x[1])
    format_len = len(str(max(stat.values(), key=lambda x: x[1])[1]))
    for key in keys:
        opstat = stat[key]
        print(f'{opstat[1]:{format_len}} {key[0]} {key[2]} : {key[1]}')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Print model statistics")
    parser.add_argument('model', type=Path, help="Path to the model.")
    parser.add_argument('--obfuscate', action=argparse.BooleanOptionalAction, default=True,
                        help="Obfuscate static dimensions (default: True). Use --no-obfuscate to get actual values.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    ovstat(str(args.model), args.obfuscate)


if __name__ == "__main__":
    main()
