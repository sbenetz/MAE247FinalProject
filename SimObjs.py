import numpy as np
from params import *

def get_orthogonal(vector):
    x = np.random.randn(vector.shape[0])  # take a random vector
    if x.size == 2:
        x[0] = vector[1]
        x[1] = -vector[0]
        return x
    x -= x.dot(vector) * vector       # make it orthogonal to k
    x /= np.linalg.norm(vector)  # normalize it
    return x

class Environment():
    def __init__(self,height,width,depth=None) -> None:
        self.height = height
        self.width = width
        self.depth = depth
        self.dims = [self.height,self.width,self.depth] 
    def __getitem__(self,dim):
        return self.dims[dim]
    
class Agent():
    def __init__(self,orientation,point):
        self.dir_vectors = orientation
        self.point = point
    def orientation(self,dim=None):
        if dim:
            return self.dir_vectors[dim,:]
        return self.dir_vectors
    def position(self):
        return self.point
    
class RandomAgent(Agent):
    def __init__(self,env:Environment):
        ori = np.random.rand(DIM)-0.5
        ori /= np.linalg.norm(ori)
        orient = np.zeros((DIM,DIM))
        orient[0,:] = ori
        orient[1,:] = get_orthogonal(ori)
        if DIM == 3:
            orient[2,:] = y = np.cross(ori, orient[1,:]) 
        pos = np.random.rand(DIM)
        for dim in range(pos.size):
            pos[dim] = pos[dim]*(env[dim]*0.8)+(env[dim]*0.1)
        super().__init__(orient,pos)
        
        
class Agents():
    def __init__(self):
        self.pos_matrix = None
        self.ori_matrix = None
        self.N = 0
        self.delayed_pos_matrix = None
        self.delayed_diff = 0
    def add_agent(self,agent:Agent):
        if self.N > 0:
            self.pos_matrix = np.vstack((self.pos_matrix,agent.position()))
            self.ori_matrix = np.dstack((self.ori_matrix,agent.orientation()))
        else:
            self.pos_matrix = agent.position()
            self.ori_matrix = agent.orientation()
        self.N+=1
    def step_algorithm(self,b):
        update_pos_mat = self.pos_matrix.copy()
        for i in range(self.N):
            di = np.sum(A[i,:])
            if di == 0: update_pos_mat[i,:] = self.pos_matrix[i,:]; continue
            update_pos_mat[i,:] = (1-H)*self.pos_matrix[i,:]+(H/di)*(A[i,:]@self.pos_matrix + b[i,:])
            
        self.pos_matrix = update_pos_mat
    def delayed_step(self,b):
        if self.delayed_diff > 0:
            if self.delayed_diff > B:
                self.delayed_pos_matrix = None
                self.delayed_diff = 0
                self.step_algorithm(b)
            else:
                self.delayed_pos_matrix = np.dstack((self.delayed_pos_matrix,self.pos_matrix))   
                self.delayed_diff+=1
                delayed_mat = self.delayed_pos_matrix[:,:,0]
                update_pos_mat = np.zeros(self.pos_matrix.shape)
                
                for i in range(self.N):
                    di = np.sum(A[i,:])
                    if di == 0: update_pos_mat[i,:] = self.pos_matrix[i,:]; continue
                    if i in self.random_fails:
                        update_pos_mat[i,:] = (1-H)*delayed_mat[i,:]+(H/di)*(A[i,:]@delayed_mat + b[i,:])
                    else:
                        update_pos_mat[i,:] = (1-H)*self.pos_matrix[i,:]+(H/di)*(A[i,:]@self.pos_matrix + b[i,:])
                self.pos_matrix = update_pos_mat
        else:
            random_fails = []
            for n in range(AGENTS):
                if np.random.rand() < FAIL_RATE: random_fails.append(n)
            self.random_fails = random_fails
            self.delayed_pos_matrix = self.pos_matrix
            self.delayed_diff+=1
            self.step_algorithm(b)
            
    def mixed_fails(self,b):
        if self.delayed_diff > 0:
            if self.delayed_diff > B:
                self.delayed_pos_matrix = None
                self.delayed_diff = 0
                self.step_algorithm(b)
            else:
                self.delayed_pos_matrix = np.dstack((self.delayed_pos_matrix,self.pos_matrix))   
                self.delayed_diff+=1
                update_pos_mat = np.zeros(self.pos_matrix.shape)
                delayed_mat = self.delayed_pos_matrix[:,:,0]
                for i in range(self.N):
                    di = np.sum(A[i,:])
                    mixed_matrix = self.pos_matrix.copy()
                    fails = list(np.nonzero(self.delayed_connections[i,:]))
                    mixed_matrix[fails,:] = delayed_mat[fails,:]
                    update_pos_mat[i,:] = (1-H)*mixed_matrix[i,:]+(H/di)*(A[i,:]@mixed_matrix + b[i,:])
                self.pos_matrix = update_pos_mat
        else:
            self.delayed_connections = np.random.randint(2, size=(AGENTS,AGENTS))
            self.delayed_pos_matrix = self.pos_matrix
            self.delayed_diff+=1
            self.step_algorithm(b)
            
    
