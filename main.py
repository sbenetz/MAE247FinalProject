import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from SimObjs import Environment, RandomAgent, Agents
from plotting_utils import *
from params import *

# setup 
ENV = Environment(HEIGHT,WIDTH,DEPTH)
EXPERIMENT_NAME = GRAPH_NAME + "_"+str(AGENTS)+"_agents"
if DELAY: EXPERIMENT_NAME += "_delay_"+str(B)+"_fail_rate_"+ str(FAIL_RATE)
elif MIXED_DELAY: EXPERIMENT_NAME += "_mixed_delay_"+str(B)+"_fail_rate_"+ str(FAIL_RATE)
## Setup Initial State
agents = Agents()

# random positions and orientations for agents
for i in range(AGENTS):
    agents.add_agent(RandomAgent(ENV))

# Plot the initial graph
fig, ax = plt.subplots()
graph_lines = plot_graph_lines(ax,agents,A)

X_0 = agents.pos_matrix[:,0]
Y_0 = agents.pos_matrix[:,1]
X_vecs,Y_vecs = plot_orientations(ax,agents)

plt.xlim(0,ENV.width)
plt.ylim(0,ENV.height)
plt.title("Orientation Alignment")
# select one node to start moving in x dir
rand_agent = np.random.randint(0,AGENTS) # must be reachable from all nodes
Aligned_agents = [rand_agent]
alignment_dir = agents.ori_matrix[0,:,rand_agent]
orig_positions = agents.pos_matrix.copy()

def alignment_step(agents:Agents, aligned_a:list):
    # align agents with the one they observe
    aligned = []
    for i in range(AGENTS):
        if i in aligned_a: 
            continue
        for a in aligned_a:
            if A[i,a] > 0: # if neighbors
                agents.ori_matrix[:,:,i] = agents.ori_matrix[:,:,a]
                aligned.append(i)
                break  
    aligned_a.extend(aligned)
        
def update_orientations(num, X_vecs,Y_vecs,line_segs):
    """Animation function for aligning reference frames at the beginning"""
    offsets = X_vecs.get_offsets()
    original = np.allclose(offsets,orig_positions)
    if len(Aligned_agents) == AGENTS and original: return
    if not original: offsets = orig_positions
    else:
        for a in Aligned_agents:
            offsets[a,:] += Alignment_change*alignment_dir
            
    update_graph_lines(line_segs,offsets,A)
    X_vecs.set_offsets(offsets)
    Y_vecs.set_offsets(offsets)   
    
    if not original: 
        # align agents with the one they observe
        alignment_step(agents,Aligned_agents)
        X_vecs.set_UVC(agents.ori_matrix[0,0,:],agents.ori_matrix[0,1,:]) 
        Y_vecs.set_UVC(agents.ori_matrix[1,0,:],agents.ori_matrix[1,1,:])            
    return X_vecs,Y_vecs, line_segs

#animate frame alignment
anim = animation.FuncAnimation(fig, update_orientations, fargs=(X_vecs,Y_vecs,graph_lines),
                               interval=1000, blit=False,frames=AGENTS*2, repeat=False)
if SAVE: anim.save('./result_gifs/'+EXPERIMENT_NAME+ '_ALIGN.gif')
# Show plot with grid

plt.grid()
plt.show()

# run the actual algorithm
L_G = np.diag(np.sum(A,1))-A
b = L_G@Z

errors = np.zeros((AGENTS,MAX_TIME))
total_iters = [0] 
fig2, ax2 = plt.subplots()  
# plot the formation
ax2.plot(Z[:,0],Z[:,1],color="blue",alpha=0.6)
ax2.plot([Z[0,0],Z[-1,0]],[Z[0,1],Z[-1,1]],color="blue",alpha=0.6)
plt.xlim(0,ENV.width)
plt.ylim(0,ENV.height)
plt.title("Motion Control, H="+str(H))
agents.pos_matrix = orig_positions
Xvecs = ax2.quiver(X_0,Y_0,X_vecs.U,X_vecs.V,color="green")
Yvecs = ax2.quiver(X_0,Y_0,Y_vecs.U,Y_vecs.V,color="red")
graphlines = plot_graph_lines(ax2,agents,A)

 
def algorithm_update(num,X_vecs,Y_vecs,graphlines):
    curr_pos = X_vecs.get_offsets()
    errors[:,num] = np.linalg.norm(curr_pos-Z,axis=1)
    if DELAY:
        agents.delayed_step(b)
    elif MIXED_DELAY:
        agents.mixed_fails(b)
    else: agents.step_algorithm(b)
    X_vecs.set_offsets(agents.pos_matrix)
    Y_vecs.set_offsets(agents.pos_matrix)
    update_graph_lines(graphlines,agents.pos_matrix,A)  
    total_iters[0] = num
    return X_vecs, Y_vecs, graphlines

anim = animation.FuncAnimation(fig2, algorithm_update, fargs=(Xvecs,Yvecs,graphlines),
                               interval=700, blit=False,frames=MAX_TIME, repeat=False)  
if SAVE: anim.save('./result_gifs/'+EXPERIMENT_NAME+ '_CONTROL.gif') 

copy_agents = Agents()
copy_agents.pos_matrix = agents.pos_matrix.copy()
copy_agents.N = agents.N
# run the algorithm without animation to get end results
for t in range(MAX_TIME):
    if DELAY:
        copy_agents.delayed_step(b)
    elif MIXED_DELAY:
        copy_agents.mixed_fails(b)
    else: copy_agents.step_algorithm(b)

ax2.set_xlim((min(np.min(copy_agents.pos_matrix[:,0])-ENV.width*0.1, 0),max(np.max(copy_agents.pos_matrix[:,0])+ENV.width*0.1, ENV.width)))
ax2.set_ylim((min(np.min(copy_agents.pos_matrix[:,1])-ENV.height*0.1, 0),max(np.max(copy_agents.pos_matrix[:,1])+ENV.height*0.1, ENV.height)))
ax2.grid(True)
plt.show()   

# plot the error over time
plt.figure()
plt.plot(errors.T[:total_iters[0]-1])
plt.xlabel('Time')
plt.ylabel("Error")
plt.show() 
