# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:47:58 2017

@author: siva_
"""
#==============================================================================
# Imports
#==============================================================================
import random
import numpy as np
import matplotlib.pyplot as plt
import time as time

from  BridgeWorld.bridgeModel import WorldModel

import gym

from collections import deque
import json
#import h5py
from keras.models import Sequential
from keras.models import model_from_json
#from keras.models import h5py
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
#import tensorflow as tf


#==============================================================================

#==============================================================================
# Program Constants
#==============================================================================
OBSERVATION = 10000 # Timesteps to observe before training
GAMMA = 0.99 # Decay rate of past observations

#-- Exploration - Explotiation balance --#
EXPLORE = 500000 # Frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # Final value of epsilon
INITIAL_EPSILON = 1 # Starting value of epsilon

#-- Training parameters --#
TRAIN_INTERVAL = 50
REPLAY_MEMORY = 400000 # Number of previous transitions to remember
BATCH = 100 # Size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-3


#-- Reward selection --#
REWARD_LOSS = -10
REWARD_NOLOSS = 0
REWARD_TOO_SLOW = 0#-1
REWARD_WELL_DONE = 100
#==============================================================================


#==============================================================================
# Building Q-Function model structure
#==============================================================================
def build_model(comm):
    
    if(comm == 1):
        print("Building Model")
        model = Sequential()
        model.add(Dense(50, input_dim = 9+3+8, activation = 'relu', kernel_initializer='he_uniform'))
        model.add(Dense(30, activation = 'relu', kernel_initializer='he_uniform'))
        #model.add(Dense(10, activation = 'relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation = 'linear', kernel_initializer='he_uniform'))
       
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)    
        #model.compile(loss = 'mse', optimizer = 'sgd')
        print("Model Summary")
        model.summary()
    else:
        print("Building Model")
        model = Sequential()
        model.add(Dense(30, input_dim = 9+3, activation = 'relu', kernel_initializer='he_uniform'))
        model.add(Dense(30, activation = 'relu', kernel_initializer='he_uniform'))
        #model.add(Dense(10, activation = 'relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation = 'linear', kernel_initializer='he_uniform'))
       
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)    
        #model.compile(loss = 'mse', optimizer = 'sgd')
        print("Model Summary")
        model.summary()
    
    
    return model    
#==============================================================================



# =============================================================================
# Train agents sequentially by taking taking actions after communicating
# =============================================================================
def getSeqActions(agents,model,commModel,withComm = 0):
    
    agShuffle = list(range(0,len(agents)))
    random.shuffle(agShuffle)
    
    a_t= []
    ca_t = []
    s_t = []
    
    
    for i in agShuffle:
        ag = agents[i]
        state = processStates([ag],withComm)

        action = predictActions(model,state)
        takeActions([ag],action, comm = 0)
        
        if(withComm == 1):
            commAction = predictActions(commModel,state)
        else:
            commAction = takeRandomActions(agents, comm = 1)  
            
        takeActions([ag],commAction, comm = 1)
        
        s_t.append(state[0])
        a_t.append(action[0])
        ca_t.append(commAction[0])
    
    # Reset communication after all values have been read and decisions made
    for i in agShuffle:
        ag = agents[i]
        ag.comm_action = -1
    
    return([a_t, ca_t, s_t])
# =============================================================================

    
    
#==============================================================================
# Loading a trained model
#==============================================================================
def load_model(model, file_name):
    model.load_weights(file_name)
    #model.compile(loss = 'mse', optimizer = 'sgd')
    return model    
#==============================================================================


#==============================================================================
# Obtain states from all the agents
#==============================================================================
def processStates(agents, withComm = 0):    
    states = []
    for a in agents:
        (s_t,x,y,targX) = a.getState()                
        s_t = np.array(s_t.reshape(1, s_t.shape[0]*s_t.shape[1]))    
        s_t = np.append(s_t,[x,y,targX])
        if(withComm==1):
            commState = a.getCommState()            
            s_t = np.append(s_t,np.array(commState))         
        s_t = np.array(s_t.reshape(1,s_t.shape[0]))

       

        states.append(s_t)            
    return(states)
#==============================================================================


#==============================================================================
# Set actions to all the agents
#==============================================================================
def takeActions(agents,a_t, comm = 0):
    cntA = 0
    for a in agents:
        if(comm == 0):
            a.action = a_t[cntA]
        else:
            a.comm_action = a_t[cntA]
        cntA = cntA + 1

    return()
#==============================================================================


#==============================================================================
# Set random actions to all agents
#==============================================================================
def takeRandomActions(agents, comm = 0):
    a_t = []
    for a in agents:
        if(comm == 0):
            a_t.append(random.sample(range(agents[0].action_space_n),1)[0]) 
        else:
            a_t.append(random.sample(range(agents[0].comm_action_space_n),1)[0]) 
    #print(a_t)
    return(a_t)
#==============================================================================

#==============================================================================
# Get the rewards from all the agents
#==============================================================================
def getRewards(agents):
    rewards = []
    for a in agents:
        rewards.append(a.getReward())
    rewards = np.array(rewards)
    return(rewards)
#==============================================================================


#==============================================================================
# Get the collisions from all the agents
#==============================================================================
def getCollisions(agents):
    colAA = []
    colAO = []
    colAW = []    
    for a in agents:
        colAA.append(a.collisionAA)
        colAO.append(a.collisionAO)
        colAW.append(a.collisionAW)
        
    colAA = np.median(np.array(colAA))
    colAO = np.median(np.array(colAO))
    colAW = np.median(np.array(colAW))     
    return([colAA, colAO, colAW])
#==============================================================================
    
    

#==============================================================================
# Predict the actions given the model and states
#==============================================================================
def predictActions(model,s_t):    
    a_t = []
    
    for s in s_t:
        q = model.predict(s)
        act = np.argmax(q)
        a_t.append(act)
        
    return(a_t)
#==============================================================================


#==============================================================================
# Updating replay memory
#==============================================================================
def updateReplayMemory(D,s_t1,a_t,ca_t,r_t,s_t2,done):
    cntA = 0    
    for s in s_t1:
        D.append([s_t1[cntA], a_t[cntA], ca_t[cntA], r_t[cntA], s_t2[cntA], done])
        cntA = cntA + 1
        
    return(D)
#==============================================================================


#==============================================================================
# Reset by recreating environment
#==============================================================================
def resetGame(height = 11, width = 11, noAgents = 22):
    
    #height = 11
    #width = 11
    #noAgents = 22
    env = WorldModel(noAgents, width, height) 
    #stateRadius = 2
    agents = env.schedule.agents       

    return(env, agents)
#==============================================================================


#==============================================================================
# Select between model training and evaluation
#==============================================================================
modelName = 'cModel1'#'twoAgentModel4'
height = 11
width = 11
noAgents = 1
totAgents = 2*height

FORWARD_LOOKING = 1
COMM_ON = False
    
REND = 0    
WATCHDOG = 100
TRIALS = 5 



cntAA = np.zeros(TRIALS)
cntAO = np.zeros(TRIALS)
cntAW = np.zeros(TRIALS)

collAA = {}
collAA['Model'] = []
collAA['Comm'] = []
collAA['Rand'] = []

collAO = {}
collAO['Model'] = []
collAO['Comm'] = []
collAO['Rand'] = []

collAW = {}
collAW['Model'] = []
collAW['Comm'] = []
collAW['Rand'] = []

scores = {}
scores['Model'] = []
scores['Comm'] = []
scores['Rand'] = []



for agNo in range(0,totAgents):
            
    noAgents = agNo + 1

    for commCnt in range(0,2):
            
        COMM_ON = commCnt

        if(COMM_ON == 1):
            modelName = 'cModel1'
        else:#'twoAgentModel4'        
            modelName = 'twoAgentModel4'
        
        model = build_model(COMM_ON)
        commModel = build_model(COMM_ON)
        
        [env, agents] = resetGame(height, width, noAgents)
        
        file_name = modelName+".h5"
        load_model(model, file_name)
        
        if(COMM_ON == 1):
            comm_file_name = "comm_"+modelName+".h5"
            load_model(commModel, comm_file_name)
        
    
        
        #-- Evaluation --#
        if(FORWARD_LOOKING == 1):
            avgT = np.zeros(TRIALS)
            avgTR = np.zeros(TRIALS)
            for i_episode in range(TRIALS):
                [env, agents] = resetGame(height, width, noAgents)
                s_t = processStates(agents, withComm = COMM_ON)
                t = 0
                totR = 0
                done = 0
                while(done == 0):
                    if(REND == 1):
                        env.render()
                        time.sleep(0.5)
                    ## play the game with model    
                    a_t = predictActions(model,s_t) 
                    takeActions(agents,a_t,comm=0)                   
                    #print('Action = ', a_t)
                    
                    if(COMM_ON == 1):
                        ca_t = predictActions(commModel,s_t)
                        takeActions(agents,ca_t,comm=1)
                        #print('Comm Action = ', ca_t)
                        
    
                        
                    env.step()
                    s_t = processStates(agents, withComm = COMM_ON)
                    r_t = getRewards(agents)
                    done = env.isGameDone()  
                    t = t + 1
                    totR = totR + np.mean(r_t)
                    if(done or (t > WATCHDOG)):
                        if(REND == 1):
                            env.render()
                            time.sleep(0.5)      
                            if(t==0):
                                time.sleep(3)
                        print("Cumulative Reward =  {}".format(totR))
                        [cntAA[i_episode], cntAO[i_episode], cntAW[i_episode]] = getCollisions(agents)
                        avgT[i_episode] = totR                         
                        [env, agents] = resetGame(height, width, noAgents)
                        break        
        else:
            avgT = np.zeros(TRIALS)
            avgTR = np.zeros(TRIALS)
            for i_episode in range(TRIALS):
                [env, agents] = resetGame(height, width, noAgents)
                s_t = processStates(agents, withComm = COMM_ON)
                t = 0
                totR = 0
                done = 0
                while(done == 0):
                    if(REND == 1):
                        env.render()
                        time.sleep(0.5)
                    ## play the game with model 
                    [a_t, ca_t, s_t] = getSeqActions(agents,model,commModel,withComm = COMM_ON)
                    #a_t = predictActions(model,s_t) 
                    #takeActions(agents,a_t,comm=0)                   
                    #print('Action = ', a_t)
                    
                    #if(COMM_ON == 1):
                        #ca_t = predictActions(commModel,s_t)
                        #takeActions(agents,ca_t,comm=1)
                        #print('Comm Action = ', ca_t)
                        
            
                        
                    env.step()
                    #s_t = processStates(agents, withComm = COMM_ON)
                    r_t = getRewards(agents)
                    done = env.isGameDone()  
                    t = t + 1
                    totR = totR + np.mean(r_t)
                    if(done or (t > WATCHDOG)):
                        if(REND == 1):
                            env.render()
                            time.sleep(0.5)      
                            if(t==0):
                                time.sleep(3)
                        print("Cumulative Reward =  {}".format(totR))
                        [cntAA[i_episode], cntAO[i_episode], cntAW[i_episode]] = getCollisions(agents)
                        avgT[i_episode] = totR
                        [env, agents] = resetGame(height, width, noAgents)
                        break
                
        if(commCnt == 0):
            scores['Model'].append(np.mean(avgT))
            collAA['Model'].append(np.mean(cntAA))
            collAO['Model'].append(np.mean(cntAO))
            collAW['Model'].append(np.mean(cntAW))
            #scoresModel[agNo] = np.mean(avgT)
        else:
            scores['Comm'].append(np.mean(avgT))
            collAA['Comm'].append(np.mean(cntAA))
            collAO['Comm'].append(np.mean(cntAO))
            collAW['Comm'].append(np.mean(cntAW))            
            #scoresComm[agNo] = np.mean(avgT)


    for i_episode in range(TRIALS):
        t = 0
        totR = 0
        done = 0        
        while(done == 0):
            if(REND == 1):
                env.render()
    
            ## play the game randomly
            a_t = takeRandomActions(agents)  
            takeActions(agents,a_t,comm = 0)
            #print('Action = ', a_t)
            if(COMM_ON == 1):
                ca_t = takeRandomActions(agents,comm=1)
                takeActions(agents,ca_t, comm = 1)
                #print('comm Action = ', ca_t)
                
            env.step()
            r_t = getRewards(agents)
            done = env.isGameDone()                  
            t = t + 1
            totR = totR + np.mean(r_t)
            if(done or (t > WATCHDOG)):
                print("Cumulative Reward =  {}".format(totR))
                [cntAA[i_episode], cntAO[i_episode], cntAW[i_episode]] = getCollisions(agents)
                avgTR[i_episode] = totR
                [env, agents] = resetGame(height, width, noAgents)
                break
        
        
    print("\n")
    print("Average Peformances")
    print("Average RL reward = {}".format(np.mean(avgT)))   
    print("Average Random reward = {}".format(np.mean(avgTR)))
    percentImprove = ((np.mean(avgT) - np.mean(avgTR))/abs((np.mean(avgTR)) ))*100
    print("Percentage improvement = {}".format(percentImprove) )

    
    scores['Rand'].append(np.mean(avgTR))
    collAA['Rand'].append(np.mean(cntAA))
    collAO['Rand'].append(np.mean(cntAO))
    collAW['Rand'].append(np.mean(cntAW))      
    #scoresRand[agNo]= np.mean(avgTR)   
#==============================================================================
    
#%%
#==============================================================================
# Plotting and saving all scores
#==============================================================================
plt.figure()
agents = list(range(1,totAgents+1))
line1, = plt.plot(agents,scores['Rand']-np.mean(scores['Rand']), label = 'Random')
line2, = plt.plot(agents,scores['Model']-np.mean(scores['Rand']), label = 'No Comm.')
line3, = plt.plot(agents,scores['Comm']-np.mean(scores['Rand']), label = 'With Comm.')
plt.legend(handles = [line1,line2,line3])
plt.grid(True)
plt.title('Average Rewards')
plt.xlabel('Number of Agents')
plt.ylabel('Score')
plt.savefig('BridgeWorldRewardComparisons.eps', bbox_inches='tight')
#==============================================================================

#==============================================================================
# Plotting and saving all AA collisions
#==============================================================================
plt.figure()
agents = list(range(1,totAgents+1))
line1, = plt.plot(agents,collAA['Rand'], label = 'Random')
line2, = plt.plot(agents,collAA['Model'], label = 'No Comm.')
line3, = plt.plot(agents,collAA['Comm'], label = 'With Comm.')
plt.legend(handles = [line1,line2,line3])
plt.grid(True)
plt.title('Average Agent-Agent Collisions')
plt.xlabel('Number of Agents')
plt.ylabel('Number of Collisions')
plt.savefig('BridgeWorldCollisionsAA.eps', bbox_inches='tight')
#==============================================================================

#==============================================================================
# Plotting and saving all AO collisions
#==============================================================================
plt.figure()
agents = list(range(1,totAgents+1))
line1, = plt.plot(agents,collAO['Rand'], label = 'Random')
line2, = plt.plot(agents,collAO['Model'], label = 'No Comm.')
line3, = plt.plot(agents,collAO['Comm'], label = 'With Comm.')
plt.legend(handles = [line1,line2,line3])
plt.grid(True)
plt.title('Average Agent-Obstacle Collisions')
plt.xlabel('Number of Agents')
plt.ylabel('Number of Collisions')
plt.savefig('BridgeWorldCollisionsAO.eps', bbox_inches='tight')
#==============================================================================

#==============================================================================
# Plotting and saving all AW collisions
#==============================================================================
plt.figure()
agents = list(range(1,totAgents+1))
line1, = plt.plot(agents,collAW['Rand'], label = 'Random')
line2, = plt.plot(agents,collAW['Model'], label = 'No Comm.')
line3, = plt.plot(agents,collAW['Comm'], label = 'With Comm.')
plt.legend(handles = [line1,line2,line3])
plt.grid(True)
plt.title('Average Agent-Wall Collisions')
plt.xlabel('Number of Agents')
plt.ylabel('Number of Collisions')
plt.savefig('BridgeWorldCollisionsAW.eps', bbox_inches='tight')
#==============================================================================

