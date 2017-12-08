import ast

with open("data/V2/repo_split/repo_split.parallel_methods_declbodies.valid") as f:
    bodies = f.readlines()

bodies = [body.replace("DCNL ", "\n").replace("DCSP ", "\t") for body in bodies]
print(bodies[0])
print(ast.parse(bodies[0]))