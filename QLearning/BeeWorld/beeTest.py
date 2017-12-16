#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:54:22 2017

@author: nabil
"""

from beeModel import WorldModel

height = 10
width = 10
model = WorldModel(2,width,height)

    
agent = model.schedule.agents[0]
model.render()
for i in range(0,5,1):
    agent = model.schedule.agents[0]
    agent.action = 3
    model.step()
    model.render()
    
    
print(agent.pos)