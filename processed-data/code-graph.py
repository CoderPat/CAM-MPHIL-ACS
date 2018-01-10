from ast import *
from collections import defaultdict

import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt
import json
import numpy as np

AST_EDGE = 0
NEXT_TOKEN_EDGE = 1

print_random_example = False

"""
    codegen
    ~~~~~~~

    Extension to ast that allow ast -> python code generation.

    :copyright: Copyright 2008 by Armin Ronacher.
    :license: BSD.
"""

BOOLOP_SYMBOLS = {
    And:        'and',
    Or:         'or'
}

BINOP_SYMBOLS = {
    Add:        '+',
    Sub:        '-',
    Mult:       '*',
    Div:        '/',
    FloorDiv:   '//',
    Mod:        '%',
    LShift:     '<<',
    RShift:     '>>',
    BitOr:      '|',
    BitAnd:     '&',
    BitXor:     '^',
    Pow:        '**'
}

CMPOP_SYMBOLS = {
    Eq:         '==',
    Gt:         '>',
    GtE:        '>=',
    In:         'in',
    Is:         'is',
    IsNot:      'is not',
    Lt:         '<',
    LtE:        '<=',
    NotEq:      '!=',
    NotIn:      'not in'
}

UNARYOP_SYMBOLS = {
    Invert:     '~',
    Not:        'not',
    UAdd:       '+',
    USub:       '-'
}

ALL_SYMBOLS = {}
ALL_SYMBOLS.update(BOOLOP_SYMBOLS)
ALL_SYMBOLS.update(BINOP_SYMBOLS)
ALL_SYMBOLS.update(CMPOP_SYMBOLS)
ALL_SYMBOLS.update(UNARYOP_SYMBOLS)

class AstGraphGenerator(NodeVisitor):
    def __init__(self):
        self.node_id = 0
        self.graph = defaultdict(lambda: [])
        self.node_label = {}
        self.representations = []
        self.parent = None
        self.previous_token = None


    def __add_edge(self, node, label=None, repr=None):
        if repr is None:
            repr = np.zeros(50)

        self.representations.append(repr.tolist())
        nid = self.__generate_id()
        self.node_label[nid] = label
        if self.parent is not None:
            self.graph[(self.parent, nid)].append(AST_EDGE)
        return nid

        

    def __generate_id(self):
        self.node_id += 1
        return self.node_id - 1

    def terminal(self, label):
        nid = self.__add_edge(self.parent, label)

        if self.previous_token is not None:
            self.graph[(self.previous_token, nid)].append(NEXT_TOKEN_EDGE)
        self.previous_token = nid

    def non_terminal(self, node, idx):
        initial_rep = np.zeros(50)
        initial_rep[idx] = 1
        nid = self.__add_edge(self.parent, node.__class__.__name__, initial_rep)
        return nid

    def body(self, statements):
        self.new_line = True
        for stmt in statements:
            self.visit(stmt)

    def body_or_else(self, node):
        self.body(node.body)
        if node.orelse:
            self.terminal('else:')
            self.body(node.orelse)

    def list_nodes(self, lnodes):
        for idx, nodes in enumerate(lnodes):
            self.terminal(', ' if idx else '')
            self.visit(nodes)

    def signature(self, node):
        want_comma = []
        def write_comma():
            if want_comma:
                self.terminal(', ')
            else:
                want_comma.append(True)

        padding = [None] * (len(node.args) - len(node.defaults))
        for arg, default in zip(node.args, padding + node.defaults):
            write_comma()
            self.visit(arg)
            if default is not None:
                self.terminal('=')
                self.visit(default)
        if node.vararg is not None:
            write_comma()
            self.terminal('*' + node.vararg.arg)

    def decorators(self, node):
        for decorator in node.decorator_list:
            self.terminal('@')
            self.visit(decorator)

    def visit_Assign(self, node):
        nid = self.non_terminal(node, 0)
        gparent, self.parent = self.parent, nid
        
        for idx, target in enumerate(node.targets):
            if idx:
                self.terminal(', ')
            self.visit(target)
        self.terminal(' = ')
        self.visit(node.value)
        self.parent = gparent

    def visit_AugAssign(self, node):
        nid = self.non_terminal(node, 1)
        gparent, self.parent = self.parent, nid
        
        self.visit(node.target)
        self.terminal(BINOP_SYMBOLS[type(node.op)] + '=')
        self.visit(node.value)
        self.parent = gparent

    def visit_ImportFrom(self, node):
        nid = self.non_terminal(node, 2)
        gparent, self.parent = self.parent, nid
        
        self.terminal('from %s%s import ' % ('.' * node.level, node.module))
        for idx, item in enumerate(node.names):
            if idx:
                self.terminal(', ')
            self.visit(item)
        self.parent = gparent

    def visit_Import(self, node):
        nid = self.non_terminal(node, 3)
        gparent, self.parent = self.parent, nid
        
        for item in node.names:
            self.terminal('import ')
            self.visit(item)
        self.parent = gparent

    def visit_Expr(self, node):
        nid = self.non_terminal(node, 4)
        gparent, self.parent = self.parent, nid
        
        self.generic_visit(node)
        self.parent = gparent

    def visit_FunctionDef(self, node):
        nid = self.non_terminal(node, 5)
        gparent, self.parent = self.parent, nid

        self.decorators(node)
        self.terminal('def')
        self.terminal(node.name)
        self.terminal('(')
        self.signature(node.args)
        self.terminal('):')
        self.body(node.body)
        
        self.parent = gparent

    def visit_ClassDef(self, node):
        nid = self.non_terminal(node, 6)
        gparent, self.parent = self.parent, nid
        have_args = []
        def paren_or_comma():
            if have_args:
                self.terminal(', ')
            else:
                have_args.append(True)
                self.terminal('(')

        self.decorators(node)
        
        self.terminal('class %s' % node.name)
        for base in node.bases:
            paren_or_comma()
            self.visit(base)
        # XXX: the if here is used to keep this module compatible
        #      with python 2.6.
        if hasattr(node, 'keywords'):
            for keyword in node.keywords:
                paren_or_comma()
                self.terminal(keyword.arg + '=')
                self.visit(keyword.value)
        self.terminal(have_args and '):' or ':')
        self.body(node.body)
        self.parent = gparent

    def visit_If(self, node):
        nid = self.non_terminal(node, 7)
        gparent, self.parent = self.parent, nid
        
        self.terminal('if ')
        self.visit(node.test)
        self.terminal(':')
        self.body(node.body)
        while True:
            else_ = node.orelse
            if len(else_) == 1 and isinstance(else_[0], If):
                node = else_[0]
                self.terminal('elif ')
                self.visit(node.test)
                self.terminal(':')
                self.body(node.body)
            else:
                if len(else_) > 0:
                    self.terminal('else:')
                    self.body(else_)
                break
        self.parent = gparent

    def visit_For(self, node):
        nid = self.non_terminal(node, 8)
        gparent, self.parent = self.parent, nid
        
        self.terminal('for ')
        self.visit(node.target)
        self.terminal(' in ')
        self.visit(node.iter)
        self.terminal(':')
        self.body_or_else(node)
        self.parent = gparent

    def visit_While(self, node):
        nid = self.non_terminal(node, 9)
        gparent, self.parent = self.parent, nid
        
        self.terminal('while ')
        self.visit(node.test)
        self.terminal(':')
        self.body_or_else(node)
        self.parent = gparent

    def visit_With(self, node):
        nid = self.non_terminal(node, 10)
        gparent, self.parent = self.parent, nid
        
        self.terminal('with ')
        self.list_nodes(node.items)
            
        self.terminal(':')
        self.body(node.body)
        self.parent = gparent

    def visit_Pass(self, node):
        nid = self.non_terminal(node, 11)
        gparent, self.parent = self.parent, nid
        
        self.terminal('pass')
        self.parent = gparent

    def visit_Print(self, node):
        nid = self.non_terminal(node, 12)
        gparent, self.parent = self.parent, nid
        # XXX: python 2.6 only
        
        self.terminal('print ')
        want_comma = False
        if node.dest is not None:
            self.terminal(' >> ')
            self.visit(node.dest)
            want_comma = True
        for value in node.values:
            if want_comma:
                self.terminal(', ')
            self.visit(value)
            want_comma = True
        if not node.nl:
            self.terminal(',')
        self.parent = gparent

    def visit_Delete(self, node):
        nid = self.non_terminal(node, 13)
        gparent, self.parent = self.parent, nid
        
        self.terminal('del ')
        for idx, target in enumerate(node.targets):
            if idx:
                self.terminal(', ')
            self.visit(target)
        self.parent = gparent

    def visit_TryExcept(self, node):
        nid = self.non_terminal(node, 14)
        gparent, self.parent = self.parent, nid
        
        self.terminal('try:')
        self.body(node.body)
        for handler in node.handlers:
            self.visit(handler)
        self.parent = gparent

    def visit_TryFinally(self, node):
        nid = self.non_terminal(node, 15)
        gparent, self.parent = self.parent, nid
        
        self.terminal('try:')
        self.body(node.body)
        
        self.terminal('finally:')
        self.body(node.finalbody)
        self.parent = gparent

    def visit_Global(self, node):
        nid = self.non_terminal(node, 16)
        gparent, self.parent = self.parent, nid
        
        self.terminal('global ' + ', '.join(node.names))
        self.parent = gparent

    def visit_Nonlocal(self, node):
        nid = self.non_terminal(node, 17)
        gparent, self.parent = self.parent, nid
        
        self.terminal('nonlocal ' + ', '.join(node.names))
        self.parent = gparent

    def visit_Return(self, node):
        nid = self.non_terminal(node, 18)
        gparent, self.parent = self.parent, nid
        
        if node.value:
            self.terminal('return ')
            self.visit(node.value)
        else:
            self.terminal('return')
        self.parent = gparent

    def visit_Break(self, node):
        nid = self.non_terminal(node, 19)
        gparent, self.parent = self.parent, nid
        
        self.terminal('break')
        self.parent = gparent

    def visit_Continue(self, node):
        nid = self.non_terminal(node, 20)
        gparent, self.parent = self.parent, nid
        
        self.terminal('continue')
        self.parent = gparent

    def visit_Raise(self, node):
        nid = self.non_terminal(node, 21)
        gparent, self.parent = self.parent, nid
        # XXX: Python 2.6 / 3.0 compatibility
        
        self.terminal('raise')
        if hasattr(node, 'exc') and node.exc is not None:
            self.terminal(' ')
            self.visit(node.exc)
            if node.cause is not None:
                self.terminal(' from ')
                self.visit(node.cause)
        elif hasattr(node, 'type') and node.type is not None:
            self.visit(node.type)
            if node.inst is not None:
                self.terminal(', ')
                self.visit(node.inst)
            if node.tback is not None:
                self.terminal(', ')
                self.visit(node.tback)

    # Expressions
        self.parent = gparent

    def visit_Attribute(self, node):
        nid = self.non_terminal(node, 22)
        gparent, self.parent = self.parent, nid
        self.visit(node.value)
        self.terminal('.')
        self.terminal(node.attr)
        self.parent = gparent

    def visit_Call(self, node):
        nid = self.non_terminal(node, 23)
        gparent, self.parent = self.parent, nid
        want_comma = []
        def write_comma():
            if want_comma:
                self.terminal(', ')
            else:
                want_comma.append(True)

        self.visit(node.func)
        self.terminal('(')
        for arg in node.args:
            write_comma()
            self.visit(arg)
        for keyword in node.keywords:
            write_comma()
            arg = keyword.arg or ''
            self.terminal(arg + '=' if arg else '**')
            self.visit(keyword.value)
        self.terminal(')')
        self.parent = gparent

    def visit_Name(self, node):
        nid = self.non_terminal(node, 24)
        gparent, self.parent = self.parent, nid
        self.terminal(node.id)
        self.parent = gparent

    def visit_NameConstant(self, node):
        nid = self.non_terminal(node, 25)
        gparent, self.parent = self.parent, nid
        self.terminal(str(node.value))
        self.parent = gparent

    def visit_Str(self, node):
        nid = self.non_terminal(node, 26)
        gparent, self.parent = self.parent, nid
        self.terminal(repr(node.s))
        self.parent = gparent

    def visit_Bytes(self, node):
        nid = self.non_terminal(node, 27)
        gparent, self.parent = self.parent, nid
        self.terminal(repr(node.s))
        self.parent = gparent

    def visit_Num(self, node):
        nid = self.non_terminal(node, 28)
        gparent, self.parent = self.parent, nid
        self.terminal(repr(node.n))
        self.parent = gparent

    def visit_Tuple(self, node):
        nid = self.non_terminal(node, 29)
        gparent, self.parent = self.parent, nid
        self.terminal('(')
        idx = -1
        for idx, item in enumerate(node.elts):
            if idx:
                self.terminal(', ')
            self.visit(item)
        self.terminal(idx and ')' or ',)')

    def sequence_visit(left, right):
        def visit(self, node):
            nid = self.non_terminal(node, 30)
            gparent, self.parent = self.parent, nid
            self.terminal(left)
            for idx, item in enumerate(node.elts):
                if idx:
                    self.terminal(', ')
                self.visit(item)
            self.terminal(right)
            self.parent = gparent
        return visit

    visit_List = sequence_visit('[', ']')
    visit_Set = sequence_visit('{', '}')
    del sequence_visit

    def visit_Dict(self, node):
        nid = self.non_terminal(node, 31)
        gparent, self.parent = self.parent, nid
        self.terminal('{')
        for idx, (key, value) in enumerate(zip(node.keys, node.values)):
            if idx:
                self.terminal(', ')
            self.visit(key)
            self.terminal(': ')
            self.visit(value)
        self.terminal('}')
        self.parent = gparent

    def visit_BinOp(self, node):
        nid = self.non_terminal(node, 32)
        gparent, self.parent = self.parent, nid
        self.visit(node.left)
        self.terminal(' %s ' % BINOP_SYMBOLS[type(node.op)])
        self.visit(node.right)
        self.parent = gparent

    def visit_BoolOp(self, node):
        nid = self.non_terminal(node, 33)
        gparent, self.parent = self.parent, nid
        self.terminal('(')
        for idx, value in enumerate(node.values):
            if idx:
                self.terminal(' %s ' % BOOLOP_SYMBOLS[type(node.op)])
            self.visit(value)
        self.terminal(')')
        self.parent = gparent

    def visit_Compare(self, node):
        nid = self.non_terminal(node, 34)
        gparent, self.parent = self.parent, nid
        self.terminal('(')
        self.visit(node.left)
        for op, right in zip(node.ops, node.comparators):
            self.terminal(' %s ' % CMPOP_SYMBOLS[type(op)])
            self.visit(right)
        self.terminal(')')
        self.parent = gparent

    def visit_UnaryOp(self, node):
        nid = self.non_terminal(node, 35)
        gparent, self.parent = self.parent, nid
        self.terminal('(')
        op = UNARYOP_SYMBOLS[type(node.op)]
        self.terminal(op)
        if op == 'not':
            self.terminal(' ')
        self.visit(node.operand)
        self.terminal(')')
        self.parent = gparent

    def visit_Subscript(self, node):
        nid = self.non_terminal(node, 36)
        gparent, self.parent = self.parent, nid
        self.visit(node.value)
        self.terminal('[')
        self.visit(node.slice)
        self.terminal(']')
        self.parent = gparent

    def visit_Slice(self, node):
        nid = self.non_terminal(node, 37)
        gparent, self.parent = self.parent, nid
        if node.lower is not None:
            self.visit(node.lower)
        self.terminal(':')
        if node.upper is not None:
            self.visit(node.upper)
        if node.step is not None:
            self.terminal(':')
            if not (isinstance(node.step, Name) and node.step.id == 'None'):
                self.visit(node.step)
        self.parent = gparent

    def visit_ExtSlice(self, node):
        nid = self.non_terminal(node, 38)
        gparent, self.parent = self.parent, nid
        for idx, item in node.dims:
            if idx:
                self.terminal(', ')
            self.visit(item)
        self.parent = gparent

    def visit_Yield(self, node):
        nid = self.non_terminal(node, 39)
        gparent, self.parent = self.parent, nid
        self.terminal('yield ')
        self.visit(node.value)
        self.parent = gparent

    def visit_Lambda(self, node):
        nid = self.non_terminal(node, 40)
        gparent, self.parent = self.parent, nid
        self.terminal('lambda ')
        self.signature(node.args)
        self.terminal(': ')
        self.visit(node.body)
        self.parent = gparent

    def visit_Ellipsis(self, node):
        nid = self.non_terminal(node, 41)
        gparent, self.parent = self.parent, nid
        self.terminal('Ellipsis')

    def generator_visit(left, right):
        def visit(self, node):
            nid = self.non_terminal(node, 42)
            gparent, self.parent = self.parent, nid
            self.terminal(left)
            self.visit(node.elt)
            for comprehension in node.generators:
                self.visit(comprehension)
            self.terminal(right)
            self.parent = gparent
        return visit

    visit_ListComp = generator_visit('[', ']')
    visit_GeneratorExp = generator_visit('(', ')')
    visit_SetComp = generator_visit('{', '}')
    del generator_visit

    def visit_DictComp(self, node):
        nid = self.non_terminal(node, 43)
        gparent, self.parent = self.parent, nid
        self.terminal('{')
        self.visit(node.key)
        self.terminal(': ')
        self.visit(node.value)
        for comprehension in node.generators:
            self.visit(comprehension)
        self.terminal('}')
        self.parent = gparent

    def visit_IfExp(self, node):
        nid = self.non_terminal(node, 44)
        gparent, self.parent = self.parent, nid
        self.visit(node.body)
        self.terminal(' if ')
        self.visit(node.test)
        self.terminal(' else ')
        self.visit(node.orelse)
        self.parent = gparent

    def visit_Starred(self, node):
        nid = self.non_terminal(node, 45)
        gparent, self.parent = self.parent, nid
        self.terminal('*')
        self.visit(node.value)
        self.parent = gparent

    def visit_Repr(self, node):
        nid = self.non_terminal(node, 46)
        gparent, self.parent = self.parent, nid
        # XXX: python 2.6 only
        self.terminal('`')
        self.visit(node.value)
        self.terminal('`')
        self.parent = gparent

    # Helper Nodes
    def visit_arg(self, node):
        self.terminal(node.arg)

    def visit_alias(self, node):
        nid = self.non_terminal(node, 47)
        gparent, self.parent = self.parent, nid
        self.terminal(node.name)
        if node.asname is not None:
            self.terminal(' as ' + node.asname)
        self.parent = gparent

    def visit_comprehension(self, node):
        nid = self.non_terminal(node, 48)
        gparent, self.parent = self.parent, nid
        self.terminal(' for ')
        self.visit(node.target)
        self.terminal(' in ')
        self.visit(node.iter)
        if node.ifs:
            for if_ in node.ifs:
                self.terminal(' if ')
                self.visit(if_)
        self.parent = gparent

    def visit_excepthandler(self, node):
        nid = self.non_terminal(node, 49)
        gparent, self.parent = self.parent, nid
        
        self.terminal('except')
        if node.type is not None:
            self.terminal(' ')
            self.visit(node.type)
            if node.name is not None:
                self.terminal(' as ')
                self.visit(node.name)
        self.terminal(':')
        self.body(node.body)
        self.parent = gparent
    
        


if __name__ == "__main__":
    
    with open("./data/V2/repo_split/repo_split.parallel_methods_bodies.valid") as f:
        bodies = f.readlines()

    with open("./data/V2/repo_split/repo_split.parallel_methods_decl.valid") as f:
        decls = f.readlines()

    bodies = [body.replace("DCNL ", "\n").replace("DCSP ", "\t") for body in bodies]
    bodies = ["\n".join([line[1:] for line in body.split("\n")]) for body in bodies]
    errors = 0

    data = []
    num_inits = 0
    for idx, (body, decl) in enumerate(zip(bodies, decls)):
        try:
            visitor = AstGraphGenerator() 
            visitor.visit(parse(body))
            edge_list = [(origin, t, destination) for (origin, destination), edges in visitor.graph.items() for t in edges]
            label = "__init__" in decl
            num_inits += 1 if label else 0
            node_features = visitor.representations
            data.append({"graph":edge_list, "node_features":node_features, "label":label})
        except:
            errors += 1

    new_data = []
    for graph_data in data:
        if graph_data['label'] == 0:
            if num_inits <= 0:
                continue
            new_data.append(graph_data)
            num_inits -= 0.75
        else:
            new_data.append(graph_data)

    print(len(new_data))
    with open("graphs.json", "w") as f:
        json.dump(new_data, f)

    print("Generated %d graphs out of %d snippets" % (len(bodies) - errors, len(bodies)))
    
    if print_random_example:
        print(bodies[412])
        visitor = AstGraphGenerator() 
        visitor.visit(parse(bodies[412]))
        graph = nx.parse_edgelist(["%d %d {'type': %d}" 
                    % (origin, destination, t) for (origin, destination), edges in visitor.graph.items() for t in edges], 
                                    nodetype = int,
                                    create_using = nx.DiGraph())
        pos=nx.nx_agraph.graphviz_layout(graph, prog='dot')

        
        edges = graph.edges()
        colors = ['green' if graph[source][destination]['type'] else 'blue' for source, destination in edges]
        nx.draw(graph, pos, with_labels=False, arrows=True, edges=edges, edge_color=colors)
        nx.draw_networkx_labels(graph, pos, labels=visitor.node_label)
        plt.draw()
        plt.show()
