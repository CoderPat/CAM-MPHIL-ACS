A simple implementation of the pagerank-nible algorithm described in https://arxiv.org/abs/1304.8132, along with two experimental setting for it.

#Libraries
Besides the standard python libraries. This project makes use of the Numpy Library for vector and numerical computation, NetworkX for graph structures and Matplotlib for plotting the experiments.
This project also contains a file with an external implementation (as in not made by me) of a PriorityQueue structure. For full details, view the file implementation/pqdict.py

#How to run the experiments
In order for the experiments to correctly import the implementation files, the experiment files need to be run from the project base folder (ie python3 experiments/experiment_1.py)
Note that this will save the plots and graphs then to the project basefolder rather then displaying them on screen.
Finally note that the experiments are not instantaneos, and may take from a few minutes to close to one hour