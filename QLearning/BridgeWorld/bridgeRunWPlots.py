# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:09:14 2017

@author: rajag038
"""

#==============================================================================
# Running Bridge World agents with plots
#==============================================================================
#==============================================================================


# =============================================================================
# Import statements
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

from bridgeModel import WorldModel
from mesa.batchrunner import BatchRunner

# =============================================================================



#%%

# =============================================================================
# Create and run a 10 agent world
# =============================================================================
height = 11
width = 11
model = WorldModel(22, width, height)
obstacleMap = model.obstacleMap

evenAgents = []
oddAgents = []

for agent in model.schedule.agents:
    if(agent.unique_id % 2 == 0):
        evenAgents.append(agent.pos)        
    else:
        oddAgents.append(agent.pos)
        
        
evenAgents = np.matrix(evenAgents)
oddAgents = np.matrix(oddAgents)       
 



#%%

#fig, ax = plt.subplots()
#yticks = np.arange(-0.5, height+0.5, 1) 
#xticks = np.arange(-0.5, width+0.5, 1) 


for i in range(4):
    model.step()
    model.render()
    
    
agent = model.schedule.agents[10]

print(agent.pos)

print(agent.getCommState())




'''
    plt.clf()
    
    evenAgents = []
    oddAgents = []
    
    for agent in model.schedule.agents:
        if(agent.unique_id % 2 == 0):
            evenAgents.append(agent.pos)        
        else:
            oddAgents.append(agent.pos)
            
            
    evenAgents = np.matrix(evenAgents)
    oddAgents = np.matrix(oddAgents)      
    
    # Scatter plot of agents and obstacles
    area = np.pi*(7)**2  # 15 point radii
    if(np.size(evenAgents,0)>0):
        plt.scatter(np.array(evenAgents[:,0]), np.array(evenAgents[:,1]),s=area, c='g', alpha=0.5)
    if(np.size(oddAgents,0)>0):
        plt.scatter(np.array(oddAgents[:,0]), np.array(oddAgents[:,1]),s=area, c='r', alpha=0.5)
    
    
    
    plt.axes().set_yticks(yticks, minor=True)
    plt.axes().set_xticks(xticks, minor=True)
    plt.grid(which='minor')
    plt.ylim(-0.5,height-0.5)
    plt.xlim(-0.5,width-0.5)  
    
    plt.imshow(~model.obstacleMap.T,cmap='gray')        
    
    plt.pause(0.001)

    
    '''


#%%
    
'''
# Scatter plot of agents and obstacles
plt.figure()  
fig, ax = plt.subplots()
area = np.pi*(7)**2  # 15 point radii
plt.scatter(np.array(evenAgents[:,0]), np.array(evenAgents[:,1]),s=area, c='g', alpha=0.5)
plt.scatter(np.array(oddAgents[:,0]), np.array(oddAgents[:,1]),s=area, c='r', alpha=0.5)

plt.imshow(~obstacleMap.T,cmap='gray')

yticks = np.arange(-0.5, height+0.5, 1) 
xticks = np.arange(-0.5, width+0.5, 1) 
ax.set_yticks(yticks, minor=True)
ax.set_xticks(xticks, minor=True)
plt.grid(which='minor')
plt.ylim(-0.5,height-0.5)
plt.xlim(-0.5,width-0.5)
plt.savefig('BridgeWorldStarting.eps',bbox_inches='tight')
plt.savefig('BridgeWorldStarting.png',bbox_inches='tight')


for i in range(2000):
    model.step()

evenAgents = []
oddAgents = []

for agent in model.schedule.agents:
    if(agent.unique_id % 2 == 0):
        evenAgents.append(agent.pos)        
    else:
        oddAgents.append(agent.pos)
        
        
evenAgents = np.matrix(evenAgents)
oddAgents = np.matrix(oddAgents)      

# Scatter plot of agents and obstacles
plt.figure()  
fig, ax = plt.subplots()
area = np.pi*(7)**2  # 15 point radii
plt.scatter(np.array(evenAgents[:,0]), np.array(evenAgents[:,1]),s=area, c='g', alpha=0.5)
plt.scatter(np.array(oddAgents[:,0]), np.array(oddAgents[:,1]),s=area, c='r', alpha=0.5)

plt.imshow(~obstacleMap.T,cmap='gray')

yticks = np.arange(-0.5, height+0.5, 1) 
xticks = np.arange(-0.5, width+0.5, 1) 
ax.set_yticks(yticks, minor=True)
ax.set_xticks(xticks, minor=True)
plt.grid(which='minor')
plt.ylim(-0.5,height-0.5)
plt.xlim(-0.5,width-0.5)
plt.savefig('BridgeWorldRandom.eps', bbox_inches='tight')
plt.savefig('BridgeWorldRandom.png', bbox_inches='tight')

'''