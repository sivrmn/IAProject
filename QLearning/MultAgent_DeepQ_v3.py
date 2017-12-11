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
INITIAL_EPSILON = 1 # Starting value of epsilon

#-- Training parameters --#
TRAIN_INTERVAL = 50
REPLAY_MEMORY = 200000 # Number of previous transitions to remember
BATCH = 100 # Size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-3


#-- Reward selection --#
REWARD_LOSS = -10
REWARD_NOLOSS = 0
REWARD_TOO_SLOW = 0#-1
REWARD_WELL_DONE = 0#100
#==============================================================================


#==============================================================================
# Building Q-Function model structure
#==============================================================================
def build_model():
    print("Now we build the model")
    model = Sequential()
    model.add(Dense(30, input_dim = 9+3, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(Dense(30, activation = 'relu', kernel_initializer='he_uniform'))
    #model.add(Dense(10, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(Dense(4, activation = 'linear', kernel_initializer='he_uniform'))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)    
    #model.compile(loss = 'mse', optimizer = 'sgd')
    print("We finish building the model")
    return model    
#==============================================================================



#==============================================================================
# Training the network
#==============================================================================
def train_network(model, env, agents, modelName):

    #-- Program Constants --#
    ACTIONS = env.schedule.agents[0].action_space_n # Number of valid actions 
    REND = 1
    RECORD_DIV = 1000
    #-----------------------------------------------------#

    #-- Variable initializations --#
    done = False
    t = 0 
    lclT = 0
    #r_t = []
    a_t = []

    loss = 0
    Q_sa = 0     
    rcdCnt = 0 
    Q_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
    Loss_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
    
    s_t1 = processStates(agents)
    #[s_t1A1, s_t1A2] = processStates(agents)  
#    (s_t,x,y,targX) = agent.getState()
#    s_t = np.array(s_t.reshape(1, s_t.shape[0]*s_t.shape[1]))    
#    s_t = np.append(s_t,[x,y,targX])
#    s_t = np.array(s_t.reshape(1,s_t.shape[0]))
    
    #-- Storage for replay memory --#
    D = deque()
    

    #-- Exploration of the game state space begins --#
    epsilon = INITIAL_EPSILON
    
    
    
    start_time = time.time()
    
    while t < EXPLORE:
                       
        if done:        
            [env,agents] = resetGame()
            s_t1 = processStates(agents)
            #[s_t1A1, s_t1A2] = processStates(agents)  
            #(s_t,x,y,targX) = agent.getState()
            #s_t = agent.getState()
            #s_t = np.array(s_t.reshape(1, s_t.shape[0]*s_t.shape[1]))
            #s_t = np.append(s_t,[x,y,targX])
            #s_t = np.array(s_t.reshape(1,s_t.shape[0]))            
            
        #-- Choosing an epsilong greedy action --#
        if np.random.random() <= epsilon:
            a_t = []
            a_t.append(random.sample(range(agents[0].action_space_n),1)[0])
            a_t.append(random.sample(range(agents[1].action_space_n),1)[0])
        else:
            a_t = predictActions(model,s_t1)

            #q = model.predict(s_t1A1)
            #a_tA1 = np.argmax(q)
            #q = model.predict(s_t1A2)
            #a_tA2 = np.argmax(q)
            #a_t = [a_tA1, a_tA2]
        
        #-- Exploration annealing --#
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            

        #observation, reward, done, info = env.step(action)
        takeActions(agents,a_t)
        #agent.action = a_t
        env.step()
        s_t2 = processStates(agents)
        #[s_t2A1, s_t2A2] = processStates(agents) 
        #(s_t1,x,y,targX) = agent.getState()
        #s_t1 = np.array(s_t1.reshape(1, s_t1.shape[0]*s_t1.shape[1]))
        #s_t1 = np.append(s_t1,[x,y,targX])
        #s_t1 = np.array(s_t1.reshape(1,s_t1.shape[0])) 
        
        rewards = getRewards(agents)
        #r_t = agent.getReward()
        #[s_t1, r_t] = [agent.getState(), agent.getReward()]
        done = env.isGameDone()
      
        if(done == 1):
            if(lclT >= 100): #499 before
                rewards = rewards +  REWARD_TOO_SLOW
                #r_t = r_t + REWARD_TOO_SLOW#REWARD_NOLOSS
                lclT = 0
                [env, agents] = resetGame()
            else:
                rewards = rewards + REWARD_WELL_DONE
                #r_t = r_t + REWARD_WELL_DONE
                lclT = 0

        else:
            if(lclT >= 100):
                rewards = rewards + REWARD_WELL_DONE
                #r_t = r_t + REWARD_WELL_DONE
                [env, agents] = resetGame()
                lclT = 0
            else:
                #r_t = REWARD_NOLOSS
                lclT = lclT + 1
   
     
        
        D.append([s_t1[0], a_t[0], rewards[0], s_t2[0], done])
        D.append([s_t1[1], a_t[1], rewards[1], s_t2[1], done])
        
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

            for batchCnt in range(0, len(minibatch)):
                state_t = minibatch[batchCnt][0]
                action_t = minibatch[batchCnt][1]
                reward_t = minibatch[batchCnt][2]
                state_t1 = minibatch[batchCnt][3]
                terminal = minibatch[batchCnt][4]
    
                inputs[batchCnt:batchCnt+1] = state_t
                targets[batchCnt] = model.predict(state_t)
                Q_sa = model.predict(state_t1)
    
                #-- Bellman-Deep Q update equations --#
                if terminal:
                    targets[batchCnt, action_t] = reward_t
                else:
                    targets[batchCnt, action_t] = reward_t + GAMMA * np.max(Q_sa)
    
                    
            loss = model.train_on_batch(inputs, targets)



        s_t1 = s_t2
        t = t + 1

        #-- Saving progress every 1000 iterations --#
        if((t % RECORD_DIV == 0)):# or (np.max(Q_sa)>=50000)):
            print('Saving Model')
            model.save_weights(modelName+".h5", overwrite = True)
            with open(modelName+".json", "w") as outfile:
                json.dump(model.to_json(), outfile)
                
            # Local heuristic to stop if sufficiently high 'Q' is reached
            #if(np.max(Q_sa)>=50000):
            #    t = EXPLORE
   
            
        #-- Print updates of progress every 1000 iterations --#
        if t % RECORD_DIV == 0:
            Q_Arr[rcdCnt] = np.max(Q_sa)
            Loss_Arr[rcdCnt] = loss
            rcdCnt = rcdCnt + 1
            
            print("TIMESTEP", t, "/ EPSILON", np.round(epsilon,3), "/ ACTION", a_t, "/ REWARD", rewards,  "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
                    
    end_time = time.time()
    print('Execution time')
    print(end_time - start_time)
    print('Time per iteration')
    print((end_time - start_time)/EXPLORE)                
                    
                
    return(Q_Arr, Loss_Arr)
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
# Reset by recreating environment
#==============================================================================
def resetGame():
    
    height = 11
    width = 11
    noAgents = 2
    env = WorldModel(noAgents, width, height) 
    #stateRadius = 2
    agents = env.schedule.agents       

    return(env, agents)
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
    print(a_t)
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
# Select between model training and evaluation
#==============================================================================
def deepQ(select, modelName):
    Q_Arr = 0
    Loss_Arr = 0
    if(select == 'Train'):
        
        model = build_model()
                 
        [env, agents] = resetGame()

        #init_s = agent.getState() # Agent 0 selected
        
        [Q_Arr, Loss_Arr] = train_network(model, env, agents, modelName)
        
        plt.plot(Q_Arr)
        plt.figure()
        plt.plot(Loss_Arr)
        
    elif(select == 'Test'):
                
        model = build_model()
        [env, agents] = resetGame()
        
        file_name = modelName+".h5"
        load_model(model, file_name)
        
        REND = 1    
        WATCHDOG = 5   
        TRIALS = 3   
        
        #-- Evaluation --#
        avgT = np.zeros(TRIALS)
        avgTR = np.zeros(TRIALS)
        for i_episode in range(TRIALS):
            [env, agents] = resetGame()
            
            s_t = processStates(agents)
            
            #(s_t,x,y,targX) = agent.getState()
            #s_t = agent.getState()

            
            #observation = env.reset()
            t = 0
            totR = 0
            done = 0
            while(done == 0):
                if(REND == 1):
                    env.render()
                    time.sleep(0.3)
                ## play the game with model    
                
                #s_t = np.array(s_t.reshape(1, s_t.shape[0]*s_t.shape[1]))
                #s_t = np.append(s_t,[x,y,targX])
                #s_t = np.array(s_t.reshape(1,s_t.shape[0]))              
                
                
                #s_t = observation.reshape(1, observation.shape[0])
                #q = model.predict(s_t)
                #action = np.argmax(q)
                a_t = predictActions(model,s_t)           
                #observation, reward, done, info = env.step(action)
                
                #agent.action = action
                takeActions(agents,a_t)
                print(a_t)
                env.step()
                s_t = processStates(agents)
                #(s_t,x,y,targX) = agent.getState()
                #r_t = agent.getReward()
                r_t = getRewards(agents)
                #[s_t, r_t] = [agent.getState(), agent.getReward()]
                done = env.isGameDone()  
                t = t + 1
                totR = totR + np.mean(r_t)
                if(done or (t > WATCHDOG)):
                    print("Cumulative Reward =  {}".format(totR))
                    avgT[i_episode] = totR
                    [env, agent] = resetGame()
                    break


        for i_episode in range(TRIALS):
            #observation = agent.getState()
            t = 0
            totR = 0
            done = 0
            while(done == 0):
                if(REND == 1):
                    env.render()
        
                ## play the game randomly
                #action = env.action_space.sample()
                #action = random.sample(range(agent.action_space_n),1)[0]
                a_t = takeRandomActions(agents)
                takeActions(agents,a_t)
                #observation, reward, done, info = env.step(action)
                #agent.action = action
                env.step()
                s_t = processStates(agents)
                r_t = getRewards(agents)
                #[s_t, r_t] = [agent.getState(), agent.getReward()]
                done = env.isGameDone()                  
                t = t + 1
                totR = totR + np.mean(r_t)
                if(done or (t > WATCHDOG)):
                    print("Cumulative Reward =  {}".format(totR))
                    avgTR[i_episode] = totR
                    [env, agent] = resetGame()
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
[Q_Arr, Loss_Arr] = deepQ('Train', 'twoAgentModel1')
#==============================================================================
