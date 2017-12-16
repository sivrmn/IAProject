# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:23:01 2017

@author: rajag038
"""


#==============================================================================
# Bridge World and Robot v1
#==============================================================================
# A bridge world with colliding robots




#==============================================================================

#==============================================================================
# Import statements
#==============================================================================
import numpy as np
import random
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

#==============================================================================

#==============================================================================
# Class for grid objects (obstacles, targets etc)
#==============================================================================
class GridMap():

    #--------------------------------------------------------------------------        
    # Initilize with agentGrid and add obstacles, targets etc
    #--------------------------------------------------------------------------        
    def __init__(self, agentGrid):

       self.agentGrid = agentGrid
       self.height = agentGrid.height
       self.width = agentGrid.width
       self.obstacleGrid = self.obstacleMap()
    
    #--------------------------------------------------------------------------           

    #--------------------------------------------------------------------------    
    # Define the obstacle grid with forbidden locations
    #--------------------------------------------------------------------------    
    def obstacleMap(self):

        obstacleGrid = []

        for x in range(self.width):
            col = []
            for y in range(self.height):
                state = self.isObstacle(x,y)
                if(state):
                    col.append(1)
                else:
                    col.append(0)
            obstacleGrid.append(col)

 
        return(obstacleGrid)
    #--------------------------------------------------------------------------            
    
    #--------------------------------------------------------------------------    
    # Define the obstacle locations
    #--------------------------------------------------------------------------    
    def isObstacle(self,x,y):
        
        state = False
#        h = self.height
#        w = self.width
#        if x == w-1 and y == h-2:
#            state = True
        return(state)
    #--------------------------------------------------------------------------                    


    
#==============================================================================
# Class for creating a typical bridge world agent
#==============================================================================
class BeeAgent(Agent):
    

    #--------------------------------------------------------------------------
    # Initialize all agents start with 0 penalty
    #--------------------------------------------------------------------------        
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.penalty = 0
        self.reward = 0
        self.target = None
        # Target location
#        if(unique_id%2 == 1):
#            self.targetX = model.grid.width-1
#        else:
#            self.targetX = 0
        
        # Action space:
        self.action_space = {}
        self.action_space['Left']    = 0 # Left
        self.action_space['Right']   = 1 # Right
        self.action_space['Up']      = 2 # Up
        self.action_space['Down']    = 3 # Down
        #self.action_space['Stay']    = 4 # Stay 
        
        self.action_space_n = len(self.action_space)        

        # Default Action - no movement
        self.action = self.action_space['Left']
        
        
        # Comm. Action Space:
        self.comm_action_space = {}
        self.comm_action_space['Left']    = 0 # Left
        self.comm_action_space['Right']   = 1 # Right
        self.comm_action_space['Up']      = 2 # Up
        self.comm_action_space['Down']    = 3 # Down
        #self.comm_action_space['Stay']    = 4 # Stay 
        
        self.comm_action_space_n = len(self.comm_action_space)        

        # Default Action - no movement
        self.comm_action = self.comm_action_space['Left']        
        
        
        
        # Penatly types
        self.penalty_type = {}
        self.penalty_type['AA'] = -5 # Agent to Agent
        self.penalty_type['AO'] = -5#-0.5 # Agent to Obstacle
        self.penalty_type['AW'] = -5#-0.5 # Agent to Wall
        
        # Reward
        self.goalReward = 0#100
    #--------------------------------------------------------------------------    


    #--------------------------------------------------------------------------    
    # Function for deciding where to move, currently moves to a random unoccupied location
    # or stays put
    #--------------------------------------------------------------------------        
    def randMoveDecision(self):
        local_steps = self.model.grid.get_neighborhood(self.pos,moore=False,include_center=False)
        
        possible_steps = []
        for lcl_pos in local_steps:
            if(len(self.model.grid.get_cell_list_contents(lcl_pos)) == 0 ):
                possible_steps.append(lcl_pos)
        possible_steps.append(self.pos)
        
        self.new_position = random.choice(possible_steps)
        
        return()
    #--------------------------------------------------------------------------    

        
    #--------------------------------------------------------------------------            
    # Function for deciding on a move based on a given action, if new action leads to an 
    # unoccupied location
    #--------------------------------------------------------------------------    
    def directedMoveDecision(self, action):      
        
        if(action == self.action_space['Left']):
            self.new_position = (self.pos[0]-1, self.pos[1]+0)
        elif(action == self.action_space['Right']):
            self.new_position = (self.pos[0]+1, self.pos[1]+0)
        elif(action == self.action_space['Up']):
            self.new_position = (self.pos[0]+0, self.pos[1]+1)
        elif(action == self.action_space['Down']):
            self.new_position = (self.pos[0]+0, self.pos[1]-1)
        elif(action == self.action_space['Stay']):
            self.new_position = self.pos
        else:            
            print('Error- action not recongized')
            return(-1)            
        
        local_steps = self.model.grid.get_neighborhood(self.pos,moore=False,include_center=True)

        possible_steps = []
        for lcl_pos in local_steps:
            if(len(self.model.grid.get_cell_list_contents(lcl_pos)) == 0 ):
                possible_steps.append(lcl_pos)
        possible_steps.append(self.pos)
        

        if(not (self.new_position in local_steps)):
            # Penalty for hitting a wall at a known locations? If yes, add here
            self.new_position = self.pos
            self.updatePenalty(self.penalty_type['AW'])

        if(not (self.new_position in possible_steps)):
            # Penalty for hitting an agent at a known locations? If yes, add here
            self.new_position = self.pos
            
            
        #if(self.model.obstacleMap[self.new_position]==1):
        if self.checkObstacle(self.new_position):
            # Penalty for hitting an obstacle at a known locations? If yes, add here
            self.new_position = self.pos
            self.updatePenalty(self.penalty_type['AO'])
            
            
        return()    
    #--------------------------------------------------------------------------                
    def checkObstacle(self,pos):
        if pos in self.model.obstacleList:
            condition = True
        else:
            condition = False
        return condition
            
        
    #--------------------------------------------------------------------------            
    # Function for executing a decided move
    #--------------------------------------------------------------------------    
    def executeMove(self):               
        cell_list = self.model.grid.get_neighborhood(self.new_position,moore=False,include_center=True)
        cell_list.remove(self.pos)
        
        # Remove candidates with obstacles
        for cell in cell_list:
            if(self.checkObstacle(cell)):
#            if(self.model.obstacleMap[cell]==1):
                cell_list.remove(cell)
        
        move_competitors = self.model.grid.get_cell_list_contents(cell_list)
        
        who_else = []
        for a in move_competitors:
            if(a.new_position == self.new_position):
                who_else.append(a)                             

        if(len(who_else)>0):
            self.model.grid.move_agent(self, self.pos)                
            self.updatePenalty(self.penalty_type['AA'])
        else:    
            if(not self.checkObstacle(self.new_position)):                
                self.model.grid.move_agent(self, self.new_position)  
            else:                                
                self.model.grid.move_agent(self, self.pos)
                
        
        # Reset action to Stay 
        self.action = self.action_space['Left']

        return()
    #--------------------------------------------------------------------------            
                
        
    #--------------------------------------------------------------------------    
    # Increase penalty by 1 if there is a collision
    #--------------------------------------------------------------------------    
    def updatePenalty(self, penaltyIncrement):
        self.penalty = penaltyIncrement #self.penalty + penaltyIncrement      
        return()
    #--------------------------------------------------------------------------    
 
    
    #--------------------------------------------------------------------------    
    # Update reward function
    #--------------------------------------------------------------------------    
    def getReward(self):
        if(self.getEuclidDist() == 0):
            self.reward = self.goalReward + self.penalty 
            self.goalReward = 0
        else:
            self.reward = -5+self.penalty #-5*self.getEuclidDist() + self.penalty      
        return(self.reward)
    #--------------------------------------------------------------------------  


    #--------------------------------------------------------------------------    
    # Obtain shortest Euclidean distance
    #--------------------------------------------------------------------------    
    def getEuclidDist(self):
        (tx,ty) = self.target
        (px,py) = self.pos
        dist = (tx - px) + (ty-py)
        return(dist)
    #--------------------------------------------------------------------------  

    
    #--------------------------------------------------------------------------    
    # Returns the current state of the agent (current state of the neighbourhood
    # cells within a radius r) + send the coordinates of the robot + target
    #--------------------------------------------------------------------------    
    def getState(self, radius=1):
        
        # Neighbourhood cells
        # radius+1 used as a tentative fix, mesa got the moore radius wrong
        cell_list = self.model.grid.get_neighborhood(self.pos,moore=True,include_center=True, radius = radius+1)
        
        
        # Agent locations
        agent_list= self.model.grid.get_cell_list_contents(cell_list)
        
        agent_locs = []
        for a in agent_list:
            agent_locs.append(a.pos)
            
            
        obs_list = []
        # Obstacle locations
        for cell in cell_list:
            if(self.checkObstacle(cell)):
                obs_list.append(cell)            
        
        m = 2*radius+1
        #state = np.matrix(np.zeros((m,m))) # Empty cells are represented by 0
        state = np.matrix(np.ones((m,m)))*10 # Empty cells are represented by 10
        
        x_offset = self.pos[0] - radius
        y_offset = self.pos[1] - radius
           
        for y in range(0,m):
            for x in range(0,m):
                xn = x + x_offset
                yn = y + y_offset
                
                if((xn<0 or xn >= self.model.grid.height) or (yn<0 or yn >= self.model.grid.width)):
                    state[x,m-y-1] = 2 # This is a wall        
                else:
                    if(not((xn,yn) in cell_list)): 
                        state[x,m-y-1] = 3 # This is unobserved
                    
                    else:                             
                        if((xn,yn) in obs_list):
                            state[x,m-y-1] = 2 # This is an obstacle
                                                
                        if((xn,yn) in agent_locs):
                            state[x,m-y-1] = 1 # This is an agent
            
                        if((xn,yn)==self.pos):
                            state[x,m-y-1] = -1 # Self position
                        
        state = state.T
                                          

        (x,y) = self.pos
        (targX,targY) =  self.target                                                       
        
        return(state,x,y,targX,targY)
    #--------------------------------------------------------------------------     
        
    
    #--------------------------------------------------------------------------    
    # Returns the current state comm. state (i.e. all the comm variables of the
    # appropriate neighbours)
    #--------------------------------------------------------------------------      
    def getCommState(self):
        
        radius=1
        comm_list = []
        # Comm Neighbourhood cells (agents that can potentially collide without communication)
        # radius+1 used as a tentative fix, mesa got the moore radius wrong
        cell_list = self.model.grid.get_neighborhood(self.pos,moore=True,include_center=False, radius = radius+1)        
        remove_list = self.model.grid.get_neighborhood(self.pos,moore=False,include_center=False, radius = radius)
        
        if(len(cell_list)>0 and len(remove_list)>0):
            comm_list = [x for x in cell_list if x not in remove_list]     
        
        comm_state = np.matrix(np.ones((1,8)))*(-1) #-1 indicates no communication


        # Communication Agent locations
        agent_list= self.model.grid.get_cell_list_contents(comm_list)

        
        # Collect communication variables of appropriate neighbours
        if(len(agent_list)>0):
            
            for a in agent_list:
                diffPos = np.array(self.pos) - np.array(a.pos)
                
                if(np.array_equal(diffPos, [0,-2]) ): # Agent to the north
                    comm_state[0,0] = a.comm_action
                elif(np.array_equal(diffPos, [-1,-1]) ): # Agent to the northeast
                    comm_state[0,1] = a.comm_action
                elif(np.array_equal(diffPos, [-2,0]) ): # Agent to the east
                    comm_state[0,2] = a.comm_action
                elif(np.array_equal(diffPos, [-1,1]) ): # Agent to the southeast
                    comm_state[0,3] = a.comm_action
                elif(np.array_equal(diffPos, [0,2]) ): # Agent to the south
                    comm_state[0,4] = a.comm_action
                elif(np.array_equal(diffPos, [1,1]) ): # Agent to the southwest
                    comm_state[0,5] = a.comm_action
                elif(np.array_equal(diffPos, [2,0]) ): # Agent to the west
                    comm_state[0,6] = a.comm_action
                elif(np.array_equal(diffPos, [1,-1]) ): # Agent to the northwest
                    comm_state[0,7] = a.comm_action                
                else:
                    print('Error - do not recognize the agent position')                                                       
        
        return(comm_state)
    #-------------------------------------------------------------------------- 
    

    #--------------------------------------------------------------------------    
    # Function for defining all robot activity at each simulation step
    #--------------------------------------------------------------------------    
    def step(self):
        #self.randMoveDecision()
        self.directedMoveDecision(self.action)
        return()
    #--------------------------------------------------------------------------            
    
    
    #--------------------------------------------------------------------------        
    # Function for action execution
    #--------------------------------------------------------------------------    
    def advance(self):
        self.executeMove()
        return()
    #--------------------------------------------------------------------------    

#==============================================================================


#==============================================================================
# Class for the world
#==============================================================================
class WorldModel(Model):
    #--------------------------------------------------------------------------        
    # Initialize world with a number of agents
    #--------------------------------------------------------------------------    
    def __init__(self, N, width, height):
        self.running = True
        self.num_agents = N
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, False) # Non Toroidal World
        self.mapGrid = GridMap(self.grid)
        self.obstacleList = []
        self.targetList = []
        #self.target = (self.width-1,self.height-1)
        #self.obstacle = (self.width-1,self.height-2)
        self.schedule = SimultaneousActivation(self)                
        self.yticks = np.arange(-0.5, height+0.5, 1) 
        self.xticks = np.arange(-0.5, width+0.5, 1)

        # Create agents
#        oddCnt = 0
#        evenCnt = 0
        for i in range(self.num_agents):
            a = BeeAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            p = self.grid.find_empty()
            self.grid.place_agent(a,p)
            
        for i in range(self.num_agents):
            t = self.grid.find_empty()
            self.targetList.append(t)

        for i in range(self.num_agents):
            o = self.grid.find_empty()
            self.obstacleList.append(o)
            
        for i in range(self.num_agents):
            self.schedule.agents[i].target=self.targetList[i]


        self.datacollector = DataCollector(
            #model_reporters={"Gini": compute_gini},
            agent_reporters={"Penalty": lambda a: a.penalty})
    #--------------------------------------------------------------------------    
    

    
    #--------------------------------------------------------------------------        
    # End of game check
    #--------------------------------------------------------------------------        
    def isGameDone(self):
        L = 0
        for a in self.schedule.agents:
            
            L = a.getEuclidDist() + L    
            
        if(L==0):
            gameDone = 1
        else:
            gameDone = 0
            
        return(gameDone)
    #--------------------------------------------------------------------------        
    
    #--------------------------------------------------------------------------        
    # Reset function
    #--------------------------------------------------------------------------        
    def reset(self):
        # Create agents
        oddCnt = 0
        evenCnt = 0
        for a in self.schedule.agents:
            i = a.unique_id                        
            if(i%2 == 1):
                x = 0
                y = oddCnt #i%(self.grid.height) 
                oddCnt = oddCnt + 1 
                self.grid.place_agent(a,(x,y))
            else:
                x = self.grid.width-1
                y = evenCnt #i%(self.grid.height) 
                evenCnt = evenCnt + 1
                self.grid.place_agent(a,(x,y))
        return()
    #-------------------------------------------------------------------------- 



    #--------------------------------------------------------------------------        
    # All activities to be done in the world at each step
    #-------------------------------------------------------------------------- 
    def render(self):
        plt.clf()
        agents = []
        
        for agent in self.schedule.agents:
            agents.append(agent.pos)
        agents = np.matrix(agents)
        targets = np.matrix(self.targetList)
        obstacles = np.matrix(self.obstacleList)        
        # Scatter plot of agents and obstacles
        area = np.pi*(7)**2  # 15 point radii
        plt.scatter(np.array(agents[:,0]), np.array(agents[:,1]),s=area, c='orange', alpha=1.0)
        plt.scatter(np.array(targets[:,0]), np.array(targets[:,1]),s=area, c='green',marker='P', alpha=0.75)
        plt.scatter(np.array(obstacles[:,0]), np.array(obstacles[:,1]),s=area, c='red',marker='X', alpha=0.75)
        #(x,y) = self.target
        #plt.scatter(x,y,s=area,c='green',marker='P',alpha=0.75)
        #(x,y) = self.obstacle
        #plt.scatter(x,y,s=area,c='red',marker='X',alpha=0.75)
        plt.axes().set_yticks(self.yticks, minor=True)
        plt.axes().set_xticks(self.xticks, minor=True)
        plt.grid(which='minor')
        plt.ylim(-0.5,self.height-0.5)
        plt.xlim(-0.5,self.width-0.5)  
        plt.imshow(np.zeros((self.width,self.height)),cmap='Blues')  
        plt.pause(0.001)
        return()
    #--------------------------------------------------------------------------        
    
    
    
    #--------------------------------------------------------------------------        
    # All activities to be done in the world at each step
    #--------------------------------------------------------------------------    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        return()
    #--------------------------------------------------------------------------    

        
#==============================================================================