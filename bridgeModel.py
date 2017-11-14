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

import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid, SingleGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

#==============================================================================

#==============================================================================
# Class for grid objects (obstacles, targets etc)
#==============================================================================
class GridMap():
    
    # Initilize with agentGrid and add obstacles, targets etc
    def __init__(self, agentGrid):

       self.agentGrid = agentGrid
       self.obstacleGrid = obstacleMap(agentGrid)
       self.targetGrid = targetMap(agentGrid)
        
       
    # Define the obstacle grid with forbidden locations
    def obstacleMap(self,agentGrid):
        
        return(obstacleGrid)
    
    
    # Define the target grid 
    def targetMap(agentGrid):
        
        return(targetGrid)
#==============================================================================

#==============================================================================
# Class for creating a typical bridge world agent
#==============================================================================
class BridgeAgent(Agent):
    
    # Initialize all agents start with 0 penalty
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.penalty = 0


    # Function for deciding where to move, currently moves to a random unoccupied location
    # or stays put
    def randMoveDecision(self):
        local_steps = self.model.grid.get_neighborhood(self.pos,moore=False,include_center=False)
        
        possible_steps = []
        for lcl_pos in local_steps:
            if(len(self.model.grid.get_cell_list_contents(lcl_pos)) == 0 ):
                possible_steps.append(lcl_pos)
        possible_steps.append(self.pos)
        
        self.new_position = random.choice(possible_steps)
        
        
    # Function for making one move in a random direction
    def randMove(self):               
        cell_list = self.model.grid.get_neighborhood(self.new_position,moore=False,include_center=True)
        cell_list.remove(self.pos)
        move_competitors = self.model.grid.get_cell_list_contents(cell_list)
        
        who_else = []
        for a in move_competitors:
            if(a.new_position == self.new_position):
                who_else.append(a)                             

        if(len(who_else)>0):
            self.model.grid.move_agent(self, self.pos)    
            self.updatePenalty()
        else:    
            self.model.grid.move_agent(self, self.new_position)  


    # Increase penalty by 1 if there is a collision
    def updatePenalty(self):
        self.penalty = self.penalty + 1      

    # Function for defining all robot activity at each simulation step
    def step(self):
        self.randMoveDecision()

        
    # Function for action execution
    def advance(self):
        self.randMove()


#==============================================================================


#==============================================================================
# Class for the world
#==============================================================================
class WorldModel(Model):

    # Initialize world with a number of agents
    def __init__(self, N, width, height):
        self.running = True
        self.num_agents = N
        self.grid = SingleGrid(width, height, False) # Non Toroidal World
        self.schedule = SimultaneousActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = BridgeAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            #x = random.randrange(self.grid.width)
            #y = random.randrange(self.grid.height)
            #self.grid.place_agent(a, (x, y))
            self.grid.position_agent(a, x="random", y="random")
            
        self.datacollector = DataCollector(
            #model_reporters={"Gini": compute_gini},
            agent_reporters={"Penalty": lambda a: a.penalty})

    # All activities to be done in the world at each step
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        
#==============================================================================


























