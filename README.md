# MAE 247 Final Project - Formation Control
Welcome to the final report project for MAE 247 at UCSD done by Shane Benetz, Spring 2023. In this project, you will find an implimentation of the algorithm proposed by Jorge Cortez in his paper "Global and Robust Formation-Shape Stabilization of Relative Sensing Networks." Currently, this implementation only supports the 2D case of the algorithm, but in the future hopefully we will be able to support 3D and more complex problems. For the detailed report, please see the PDF in repository 

## How to use
After cloning the repository, open up the "params.py" file in your editor. This is where you will find all the parameters tha control what kind of experiment you are going to run. More information on the parameters can be found in the later section. After setting the parameters, all you must do is call `./main.py` from the command line while in the main repository folder. This will run the experiment and create 3 separate graphs pop up, onyl having one open at a time. This first will show a gif of the first step of the algorithm which is where the agents all align their reference frames. This second will be the actual control algorithm which is where the agents apply the control algorithm to land them in the desired formation specified by the parameters. This third graph will show the error over time of the distance between the agents and the corresponding formation locations.


### Parameters
SAVE - whether to save the results to a gif file in (casuses error graph to break)

DIM - unused for now, assume 2 for 2D

DEPTH - unused for now, assume None since 2D

HEIGHT - height of output graph

WIDTH -width of output graph

Alignment_change - how far for the agents to move in the alignment step

H - convergence parameter for the algorithm (0 to 1)

MAX_TIME - how many time steps to wait for the algorithm to finish

DELAY - whether or not to run the delay experiment (full node failures)

MIXED_DELAY - whether or not to run the mixed delay experiment (random link failures)

B - length of delay for above experiments

FAIL_RATE - liklihood of failure for the above experiments


AGENTS - number of agents to simulate

GRAPH_NAME - name for experiment graph

A - numpy matrix for the adjacency matrix

Z = set of (x,y) points that is the length of the number of agents


To run a new experiment with a different graph structure, simply comment out or delete the current example experiment and add back in parameters AGENTS, GRAPH_NAME, A, and Z


### Required Python Packages
- networkx
- numpy
- matplotlib


