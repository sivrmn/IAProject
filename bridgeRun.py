# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:24:40 2017

@author: rajag038
"""
'''
import numpy as np
import matplotlib.pyplot as plt

from bridgeModel import WorldModel
from mesa.batchrunner import BatchRunner
'''
from server import server

'''
model = WorldModel(5, 10, 10)
for i in range(200):
    model.step()


plt.figure()
agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content, x, y = cell
    if(cell_content == None):
        agent_count = 0
    else:
        agent_count = 1#len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()


print(model.schedule.agents[0].penalty)

'''
'''
fixed_params = {"width": 10,
                "height": 10,
                "N": 5}

variable_params = {"N": range(10, 500, 10)}

all_params = {"width": 10,
              "height": 10,
              "N": 5}

# =============================================================================
batch_run = BatchRunner(WorldModel,
                        parameter_values=all_params,
                        iterations=1,
                        max_steps=100,
                        model_reporters={})
# =============================================================================

#==============================================================================
# batch_run = BatchRunner(MoneyModel,
#                         variable_parameters=variable_params,
#                         fixed_parameters=fixed_params,
#                         iterations=5,
#                         max_steps=100,
#                         model_reporters={"Gini": compute_gini})
#==============================================================================

batch_run.run_all()
'''

server.port = 8521 # The default
server.launch()