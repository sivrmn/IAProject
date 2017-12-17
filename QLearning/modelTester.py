# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:54:55 2017

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
INITIAL_EPSILON = 0.5 # Starting value of epsilon

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
def build_model():
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
def processStates(agents):    
    states = []
    for a in agents:
        (s_t,x,y,targX) = a.getState()                
        s_t = np.array(s_t.reshape(1, s_t.shape[0]*s_t.shape[1]))    
        s_t = np.append(s_t,[x,y,targX])
        s_t = np.array(s_t.reshape(1,s_t.shape[0]))
        states.append(s_t)
        
    return(states)
#==============================================================================


#==============================================================================
# Set actions to all the agents
#==============================================================================
def takeActions(agents,a_t):
    cntA = 0
    for a in agents:
        a.action = a_t[cntA]
        cntA = cntA + 1

    return()
#==============================================================================


#==============================================================================
# Set random actions to all agents
#==============================================================================
def takeRandomActions(agents):
    a_t = []
    for a in agents:
        a_t.append(random.sample(range(agents[0].action_space_n),1)[0]) 
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
def updateReplayMemory(D,s_t1,a_t,r_t,s_t2,done):
    cntA = 0    
    for s in s_t1:
        D.append([s_t1[cntA], a_t[cntA], r_t[cntA], s_t2[cntA], done])
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
# Model tester
#==============================================================================
modelName = 'twoAgentModel4'
height = 11
width = 11
noAgents = 1
totAgents = 4#2*height

REND = 0
WATCHDOG = 100
TRIALS = 1   

randScores = np.zeros(totAgents)
modelScores = np.zeros(totAgents)


for agNo in range(0,totAgents):
    noAgents = agNo + 1
    model = build_model()
    [env, agents] = resetGame(height, width, noAgents)
    
    file_name = modelName+".h5"
    load_model(model, file_name)
    
    
    #-- Evaluation --#
    avgT = np.zeros(TRIALS)
    avgTR = np.zeros(TRIALS)
    for i_episode in range(TRIALS):
        [env, agents] = resetGame(height, width, noAgents)
        s_t = processStates(agents)
        t = 0
        totR = 0
        done = 0
        while(done == 0):
            if(REND == 1):
                env.render()
                time.sleep(0.5)
            ## play the game with model    
            a_t = predictActions(model,s_t) 
            takeActions(agents,a_t)
            #print(a_t)
            env.step()
            s_t = processStates(agents)
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
                avgT[i_episode] = totR
                [env, agents] = resetGame(height, width, noAgents)
                break
    
    
    for i_episode in range(TRIALS):
        t = 0
        totR = 0
        done = 0        
        while(done == 0):
            if(REND == 1):
                env.render()
    
            ## play the game randomly
            a_t = takeRandomActions(agents)   
            #print(a_t)
            takeActions(agents,a_t)
            env.step()
            r_t = getRewards(agents)
            done = env.isGameDone()                  
            t = t + 1
            totR = totR + np.mean(r_t)
            if(done or (t > WATCHDOG)):
                print("Cumulative Reward =  {}".format(totR))
                avgTR[i_episode] = totR
                [env, agents] = resetGame(height, width, noAgents)
                break
        
        
    print("\n")
    print("Average Peformances")
    print("Average RL reward = {}".format(np.mean(avgT)))   
    print("Average Random reward = {}".format(np.mean(avgTR)))
    percentImprove = ((np.mean(avgT) - np.mean(avgTR))/abs((np.mean(avgTR)) ))*100
    print("Percentage improvement = {}".format(percentImprove) )


    modelScores[agNo] = np.mean(avgT)
    randScores[agNo]= np.mean(avgTR)
#==============================================================================



