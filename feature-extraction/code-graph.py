import ast
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt

with open("data/V2/repo_split/repo_split.parallel_methods_declbodies.valid") as f:
    bodies = f.readlines()


class AstGraphVisitor(ast.NodeVisitor):
    def __init__(self, source):
        self.graph = defaultdict(lambda: [])
        self.parent = None
        self.node_id = 0
        self.node_label = {}
        
    def __generate_id(self):
        self.node_id += 1
        return self.node_id - 1

    def __add_edge(self, node, label=None):
        nid = self.__generate_id()
        self.node_label[nid] = label
        if self.parent is not None:
            source = self.parent
            destination = nid
            self.graph[source].append(destination)
        return nid
        
    def generic_visit(self, node, nid=None):
        if nid is None:
            nid = self.__add_edge(node, node.__class__.__name__)

        grandp = self.parent
        self.parent = nid
        for value in ast.iter_child_nodes(node):
            if isinstance(value, list):
                for item in value:
                    self.visit(item)

            else :
                self.visit(value)
             
        self.parent = grandp

    def terminal(self, node, label):
        nid = self.__add_edge(node, label)
        self.generic_visit(node, nid)

    visit_Num = lambda self, node : self.terminal(node, node.n)
    visit_Str = lambda self, node : self.terminal(node, node.s)


if __name__ == "__main__":
    bodies = [body.replace("DCNL ", "\n").replace("DCSP ", "\t") for body in bodies]
    visitor = AstGraphVisitor(bodies[0].split("\n")) 
    visitor.generic_visit(ast.parse(bodies[0]))


    graph = nx.DiGraph(visitor.graph)
    pos=nx.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_labels=False, arrows=True)
    nx.draw(graph, pos, labels=visitor.node_label)
    plt.draw()
    plt.show()
