import sys
import openvino as ov
from collections import defaultdict


def obfuscate_shape(shape):
    if shape.rank.is_dynamic:
        return '[...]'
    result = []
    for d in shape:
        result.append('?' if d.is_dynamic else 'S')
    return '[' + ','.join(result) + ']'

def obfuscated_signature(op):
    input_types = ', '.join([port.get_element_type().get_type_name() + str(obfuscate_shape(port.get_partial_shape())) for port in op.inputs()])
    output_types = ', '.join([port.get_element_type().get_type_name() + str(obfuscate_shape(port.get_partial_shape())) for port in op.outputs()])
    return input_types + ' -> ' + output_types

def obfuscate_attributes(op):
    result = []
    for k, v in op.get_attributes().items():
        if isinstance(v, ov.PartialShape):
            v = obfuscate_shape(v)
        elif isinstance(v, list):
            if k in ['shape', 'variable_shape']:
                v = '[' + ','.join('?' if d == -1 else 'S' for d in v) + ']'
            else:
                v = '[' + ','.join(['S']*len(v)) + ']'
        elif k == 'variable_id' and op.get_type_info().name in ['ReadValue', 'Assign']:
            v = 'S'
        result.append((k, v))
    return '{ ' + ', '.join([k + ' = ' + str(v) for k, v in result]) + ' }'

def op_statistics(model, stat):
    ops = model.get_ordered_ops()
    local_ops = defaultdict(int)
    for op in ops:
        key = (op.get_type_info().name, obfuscated_signature(op), str(obfuscate_attributes(op)))
        #stat[key] += 1
        local_ops[key] += 1
    for key in local_ops:
        stat[key][0] += 1
        stat[key][1] += local_ops[key]
    return stat

def ovstat(path):
    core = ov.Core()
    stat = defaultdict(lambda: [0, 0])
    model = core.read_model(path)
    op_statistics(model, stat)
    keys = list(stat)
    keys.sort(key=lambda x: x[0])
    for key in keys:
        opstat = stat[key]
        print(key[0] + ' ' + key[2] + ' : ' + key[1] + '\t' + str(opstat[1]))

def main():
    model_path = sys.argv[1]
    ovstat(model_path)

if __name__ == "__main__":
    main()