import numpy as np
import networkx as nx

SAVE = False # save the gifs

# Parameters for experiment
DIM = 2 # unused for now, assume 2D
DEPTH = None # unused for now, assume 2D

HEIGHT = 300
WIDTH = 300

# algorithm params
Alignment_change = int(HEIGHT/10)
H = 0.4
MAX_TIME = 100

DELAY = False
MIXED_DELAY = False
B = 10
FAIL_RATE = 0.25

# set up the graph structure
# random Graph
GRAPH_NAME = "random"
AGENTS = 20
connection_rate = 0.15
graph = nx.gnp_random_graph(AGENTS,connection_rate , seed=123, directed=False)
A = nx.to_numpy_array(graph)
for n in graph:
    if len(graph[n]) == 0:
        random_connect = np.random.randint(0,AGENTS)
        A[n,random_connect] = 1
        A[random_connect,n] = 1
# circle formation 
step = np.pi*2/AGENTS
radius = max(HEIGHT,WIDTH)/3
center = [WIDTH/2,HEIGHT/2]
Z_points = []
for n in range(AGENTS):
    point = np.array(center) + np.array([radius*np.cos(n*step),radius*np.sin(n*step)])
    Z_points.append(point)
Z = np.array(Z_points)

# #circle graph
# GRAPH_NAME = "circle"
# AGENTS = 6
# # adjacency matrix
# A = np.array([[0,1,0,0,0,1],
#              [1,0,1,0,0,0],
#              [0,1,0,1,0,0],
#              [0,0,1,0,1,0],
#              [0,0,0,1,0,1],
#              [1,0,0,0,1,0]]
#              )
# # desired formation
# Z = np.array([[150,250],
#               [100,175],
#               [150,100],
#               [200,100],
#               [250,175],
#               [200,250]])

    

