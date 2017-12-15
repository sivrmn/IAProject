# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 20:00:07 2017

@author: rajag038
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
    model.add(Dense(30, input_dim = 9+3+8, activation = 'relu', kernel_initializer='he_uniform'))
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
# Training the network
#==============================================================================
def train_network(model, commModel, COMM_ON, env, agents, modelName):

    #-- Program Constants --#
    ACTIONS = env.schedule.agents[0].action_space_n # Number of valid actions 
    COMM_ACTIONS = env.schedule.agents[0].comm_action_space_n # Number of valid comm actions
    REND = 0
    RECORD_DIV = 1000
    #-----------------------------------------------------#

    #-- Variable initializations --#
    done = False
    t = 0 
    lclT = 0
    #r_t = []
    a_t = []
    ca_t = []

    loss = 0
    comm_loss = 0
    Q_sa = 0     
    comm_Q_sa = 0
    rcdCnt = 0 
    Q_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
    Loss_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
    
    Comm_Q_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
    Comm_Loss_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
        
    s_t1 = processStates(agents, withComm = COMM_ON)

    
    #-- Storage for replay memory --#
    D = deque()

    #-- Exploration of the game state space begins --#
    epsilon = INITIAL_EPSILON
    
    
    
    start_time = time.time()
    
    while t < EXPLORE:
                       
        if done:        
            [env,agents] = resetGame()
            s_t1 = processStates(agents, withComm = COMM_ON)       
            
        #-- Choosing an epsilong greedy action --#
        if np.random.random() <= epsilon:
            a_t = takeRandomActions(agents, comm = 0)               
            ca_t = takeRandomActions(agents, comm = 1)
        else:
            a_t = predictActions(model,s_t1)
            if(COMM_ON == 1):
                ca_t = predictActions(commModel,s_t1)
            else:
                ca_t = takeRandomActions(agents, comm = 1)

        
        #-- Exploration annealing --#
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            

        #observation, reward, done, info = env.step(action)
        takeActions(agents,a_t, comm = 0)
        if(COMM_ON == 1):
            takeActions(agents,ca_t, comm = 1)
            
        env.step()
        s_t2 = processStates(agents, withComm = COMM_ON)
        r_t = getRewards(agents)
        done = env.isGameDone()
      
        if(done == 1):
            if(lclT >= 100): #499 before
                r_t = r_t +  REWARD_TOO_SLOW
                lclT = 0
                [env, agents] = resetGame()
            else:
                r_t = r_t + REWARD_WELL_DONE
                lclT = 0

        else:
            if(lclT >= 100):
                r_t = r_t + REWARD_WELL_DONE
                [env, agents] = resetGame()
                lclT = 0
            else:
                lclT = lclT + 1
   
     
        
        D = updateReplayMemory(D,s_t1,a_t,ca_t,r_t,s_t2,done)                    


            
        #-- Update graphics based on action taken --#
        if(REND == 1):
            if(t>OBSERVATION):
                if(t% RECORD_DIV > 950):
                    env.render()
                

        
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        
        #-- Training after the initial observation is complete --#
        if((t > OBSERVATION)  & (t % TRAIN_INTERVAL == 0)):


            minibatch = random.sample(D, BATCH) 
            
            inputs = np.zeros((BATCH, s_t1[0].shape[1]))
            targets = np.zeros((inputs.shape[0], ACTIONS))
            if(COMM_ON == 1):
                comm_targets = np.zeros((inputs.shape[0], ACTIONS))

            for batchCnt in range(0, len(minibatch)):
                state_t = minibatch[batchCnt][0]
                action_t = minibatch[batchCnt][1]
                comm_action_t = minibatch[batchCnt][2]
                reward_t = minibatch[batchCnt][3]
                state_t1 = minibatch[batchCnt][4]
                terminal = minibatch[batchCnt][5]
    
                inputs[batchCnt:batchCnt+1] = state_t
                targets[batchCnt] = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                
                if(COMM_ON == 1):
                    comm_targets[batchCnt] = commModel.predict(state_t)
                    comm_Q_sa = commModel.predict(state_t1)
    
                #-- Bellman-Deep Q update equations --#
                if terminal:
                    targets[batchCnt, action_t] = reward_t
                    if(COMM_ON == 1):
                        comm_targets[batchCnt, comm_action_t] = reward_t
                else:
                    targets[batchCnt, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    if(COMM_ON == 1):
                        comm_targets[batchCnt, comm_action_t] = reward_t + GAMMA * np.max(comm_Q_sa)
                        
                        
            loss = model.train_on_batch(inputs, targets)
            if(COMM_ON == 1):
                comm_loss = commModel.train_on_batch(inputs,comm_targets)
            

        s_t1 = s_t2
        t = t + 1

        #-- Saving progress every 1000 iterations --#
        if((t % RECORD_DIV == 0)):# or (np.max(Q_sa)>=50000)):
            print('Saving Model')
            model.save_weights(modelName+".h5", overwrite = True)                
            with open(modelName+".json", "w") as outfile:
                json.dump(model.to_json(), outfile)

            if(COMM_ON == 1):
                commModel.save_weights("comm_"+modelName+".h5", overwrite = True)
                with open("comm_"+modelName+".json", "w") as outfile:
                    json.dump(commModel.to_json(), outfile)
                    
            # Local heuristic to stop if sufficiently high 'Q' is reached
            #if(np.max(Q_sa)>=50000):
            #    t = EXPLORE
   
            
        #-- Print updates of progress every 1000 iterations --#
        if t % RECORD_DIV == 0:
            Q_Arr[rcdCnt] = np.max(Q_sa)
            Loss_Arr[rcdCnt] = loss
            if(COMM_ON == 1):
                Comm_Q_Arr[rcdCnt] = np.max(comm_Q_sa)
                Comm_Loss_Arr[rcdCnt] = comm_loss
                
            rcdCnt = rcdCnt + 1
            
            print("TIMESTEP", t, "/ EPSILON", np.round(epsilon,3), "/ ACTION", a_t, "/ REWARD", r_t,  "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
            if(COMM_ON == 1):
                print("TIMESTEP", t, "/ EPSILON", np.round(epsilon,3), "/ commACTION", ca_t, "/ REWARD", r_t,  "/ cQ_MAX " , np.max(Q_sa), "/ comm_Loss ", loss)
                
    end_time = time.time()
    print('Execution time')
    print(end_time - start_time)
    print('Time per iteration')
    print((end_time - start_time)/EXPLORE)                
                    
                
    return(Q_Arr, Loss_Arr, Comm_Q_Arr, Comm_Loss_Arr)
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
def resetGame():
    
    height = 11
    width = 11
    noAgents = 4
    env = WorldModel(noAgents, width, height) 
    #stateRadius = 2
    agents = env.schedule.agents       

    return(env, agents)
#==============================================================================


#==============================================================================
# Select between model training and evaluation
#==============================================================================
def deepQ(select, modelName):
    COMM_ON = True
        
    Q_Arr = 0
    Loss_Arr = 0
    if(select == 'Train'):
        
        model = build_model()                 
        commModel = build_model()
        [env, agents] = resetGame()
        
        [Q_Arr, Loss_Arr, Comm_Q_Arr, Comm_Loss_Arr] = train_network(model, commModel, COMM_ON, env, agents, modelName)
        
        plt.plot(Q_Arr)
        plt.figure()
        plt.plot(Loss_Arr)
        
        plt.figure()
        plt.plot(Comm_Q_Arr)
        plt.figure()
        plt.plot(Comm_Loss_Arr)        
        
    elif(select == 'Test'):
                
        model = build_model()
        commModel = build_model()
        
        [env, agents] = resetGame()
        
        file_name = modelName+".h5"
        load_model(model, file_name)
        
        if(COMM_ON == 1):
            comm_file_name = "comm_"+modelName+".h5"
            load_model(commModel, comm_file_name)
        
        REND = 1    
        WATCHDOG = 5
        TRIALS = 1   
        
        #-- Evaluation --#
        avgT = np.zeros(TRIALS)
        avgTR = np.zeros(TRIALS)
        for i_episode in range(TRIALS):
            [env, agents] = resetGame()
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
                print('Action = ', a_t)
                
                if(COMM_ON == 1):
                    ca_t = predictActions(commModel,s_t)
                    takeActions(agents,ca_t,comm=1)
                    print('Comm Action = ', ca_t)
                    

                    
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
                    avgT[i_episode] = totR
                    [env, agents] = resetGame()
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
                takeActions(agents,a_t,comm = 0)
                print('Action = ', a_t)
                if(COMM_ON == 1):
                    ca_t = takeRandomActions(agents,comm=1)
                    takeActions(agents,ca_t, comm = 1)
                    print('comm Action = ', ca_t)
                    
                env.step()
                r_t = getRewards(agents)
                done = env.isGameDone()                  
                t = t + 1
                totR = totR + np.mean(r_t)
                if(done or (t > WATCHDOG)):
                    print("Cumulative Reward =  {}".format(totR))
                    avgTR[i_episode] = totR
                    [env, agents] = resetGame()
                    break
            
            
        print("\n")
        print("Average Peformances")
        print("Average RL reward = {}".format(np.mean(avgT)))   
        print("Average Random reward = {}".format(np.mean(avgTR)))
        percentImprove = ((np.mean(avgT) - np.mean(avgTR))/abs((np.mean(avgTR)) ))*100
        print("Percentage improvement = {}".format(percentImprove) )
    return(Q_Arr, Loss_Arr)
#==============================================================================


#==============================================================================
# Main function area
#==============================================================================
[Q_Arr, Loss_Arr] = deepQ('Test', 'cModel1')
#==============================================================================
