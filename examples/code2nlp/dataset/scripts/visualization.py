from pygraphviz import *
from ast_graph_generator import *

def plot_code_graph(snippet):
    print(snippet)

    color_dict = {
        0 : 'blue',
        1 : 'yellow',
        2 : 'black',
        3 : 'red',
        4 : 'green',
        5 : 'brown',
    }
    visitor = AstGraphGenerator() 
    visitor.visit(parse(snippet))

    A = AGraph(strict=False, directed=True)

    for (u, v), edges in visitor.graph.items():
        for edge in edges:
            A.add_edge(u, v, color=color_dict[edge])
    
    for i, (nid, label) in enumerate(visitor.node_label.items()):
        A.get_node(i).attr['label'] = "%s (%d)" % (label, nid)
    

    A.layout('dot')                                                                 
    A.draw('multi.png')   