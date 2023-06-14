import numpy as np
from SimObjs import Agents
def plot_graph_lines(ax,agents:Agents,adj):
    agent_pos_matrix = agents.pos_matrix
    lines_x = []; lines_y = []
    for i in range(agents.N):
        for j in range(i, agents.N):
            if i==j: continue
            if adj[i,j] >0 :
                lines_x.append(np.array([agent_pos_matrix[i,0],agent_pos_matrix[j,0]]).T)
                lines_y.append(np.array([agent_pos_matrix[i,1],agent_pos_matrix[j,1]]).T)
    lines=[]
    for l in range(len(lines_x)):
        lines.append(ax.plot(lines_x[l],lines_y[l],color="lightgrey",alpha=0.6,linewidth=0.9)[0])
    return lines
def update_graph_lines(lines_graph_obj,offsets,adj):
    lines_x = []
    lines_y = []
    for i in range(offsets.shape[0]):
        for j in range(i,offsets.shape[0]):
            if i==j: continue
            if adj[i,j] >0 :
                lines_x.append(np.array([offsets[i,0],offsets[j,0]]).T)
                lines_y.append(np.array([offsets[i,1],offsets[j,1]]).T) 
    for ind, line in enumerate(lines_graph_obj):
        line.set_data(lines_x[ind],lines_y[ind])  
         
def plot_orientations(ax,agents:Agents):
    X_0 = agents.pos_matrix[:,0]
    Y_0 = agents.pos_matrix[:,1]
    X_vecs = ax.quiver(X_0, Y_0, agents.ori_matrix[0,0,:], agents.ori_matrix[0,1,:],color="g")
    Y_vecs = ax.quiver(X_0, Y_0, agents.ori_matrix[1,0,:], agents.ori_matrix[1,1,:],color="r")
    return X_vecs,Y_vecs