from .expr import ExprNode, ExprLeaf


def dump_tree(obj, level=0):
    if isinstance(obj, ExprLeaf):
        return '  ' * level + repr(obj)
    elif isinstance(obj, ExprNode):
        s = '  ' * level + type(obj).__name__ + '(\n'
        for i, arg in enumerate(obj.args):
            s += (
                dump_tree(arg, level=level + 1)
                + (',\n' if i != len(obj.args) - 1 else ''))
        s += ')'
        return s

    raise ValueError
