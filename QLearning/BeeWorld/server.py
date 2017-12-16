# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:41:14 2017

@author: rajag038
"""


from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from bridgeModel import WorldModel


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}
    return portrayal

grid = CanvasGrid(agent_portrayal, 20, 10, 1000, 500)


# N can be max of 2*height
server = ModularServer(WorldModel,
                       [grid],
                       "Bridge World",
                       {"N": 20, "width": 20, "height": 10})