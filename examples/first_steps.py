import math
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import glob
import os
import sys
import cv2
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout , Conv2D, MaxPooling2D ,Activation,Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import gym
import numpy as np
from enum import Enum
import random
import ray
import tensorflow as tf
from typing import List
from gym import spaces
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import ModelConfigDict, TensorType, AlgorithmConfigDict, EnvCreator
#from environments.knapsack import KnapsackEnv
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.models.tf.tf_action_dist import get_categorical_class_with_temperature
from ray.rllib.utils.tf_utils import reduce_mean_ignore_inf

from tqdm import tqdm
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as backend
from threading import Thread

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('C:/Users/parli/Desktop/CARLA_0.9.8/WindowsNoEditor/PythonAPI/Carla_rl/carla')

    
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
from agents.navigation.local_planner import LocalPlanner

from agents.env_config import NPC, Sync_Actor

Replay_MEMORY_SIZE =  5000
MIN_REPLAY_MEMORY_SIZE=1000
MINI_BATCH_SIZE=16
PREDICTION_BATCH_SIZE =1
TRAINING_BATCH_SIZE = MINI_BATCH_SIZE//4
UPDATE_TARGET_STEPS = 5
MEMORY_FRACTION=0.2
MIN_REWARD =-200
DISCOUNT=0.99
EPISODES =100
e=1
SECONDS_PER_EPISODE=10.0
IM_WIDTH=160
IM_HEIGHT=120
e_decay=0.95

MODEL_NAME="dqn"
AGGREGATE_STATS_EVERY = 10


       

class ActionType(Enum):
    ULTRA_SLOW_ACCELARATION = 0
    SLOW_ACCELARATION = 1
    MEDIUM_ACCELARATION = 2
    HIGH_ACCELARATION = 3
    SLOW_BREAK = 4
    MEDIUM_BREAK = 5
    HIGH_BREAK = 6
    FULL_BREAK = 7
    LEAVE_PEDALS = 8
    LOW_DECCELARATION = 9
    MEDIUM_DECCELARATION = 10
    HIGH_DECCELARATION =11
    SLOW_RELEASE_BREAK =12
    MEDIUM_RELEASE_BREAK=13
    HIGH_RELEASE_BREAK =14

    


class CarEnv(gym.Env):
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(80.0)  # duration for errors
        self.accelaration= 0
        self.brake= 0
        self.previous_accelaration=0
        self.previous_brake=0
        self.set_up_car= Sync_Actor()
        
        


    def reset(self):
        self.set_up_car.spawn_car()   
        self.episode_start =time.time()
        
       

    def step(self, function):
        if(function==ActionType.ULTRA_SLOW_ACCELARATION.value):                                                                                           # ultra slow accelaration (for example stop sign going in a blind turn without priority) #possibly not needed
            self.accelaration+= 0.1
            self.brake=0
        elif (function == ActionType.SLOW_ACCELARATION.value):                                                                                      #slightly_accelarate 
            self.accelaration+= 0.2
            self.brake=0
        elif (function == ActionType.MEDIUM_ACCELARATION.value):                                                                                      # accelarate medium
            self.accelaration+= 0.4
            self.brake=0
        elif (function ==  ActionType.HIGH_ACCELARATION.value):                                                                                      # accelarate heavily
            self.accelaration+= 0.6
            self.brake=0
        elif (function ==  ActionType.SLOW_BREAK.value):                                                                                       #slighty increase break
            self.accelaration+= 0.0
            self.brake+=0.1
        elif (function == ActionType.MEDIUM_BREAK.value):                                                                                       #  increase break medium
            self.accelaration=0.0
            self.brake+= 0.3
        elif (function == ActionType.HIGH_BREAK.value):                                                                                       # heavily increase break
            self.accelaration= 0.0
            self.brake+=0.5
        elif (function == ActionType.FULL_BREAK.value):                                                                                       #  full break 
            self.accelaration= 0.0
            self.brake=1.0
        elif (function == ActionType.LEAVE_PEDALS.value):                                                                                       # dont press any pedals
            self.accelaration= 0.0
            self.brake=0.0
        elif (function == ActionType.LOW_DECCELARATION.value):                                                                                      #slightly deccelarate 
            self.accelaration-= 0.2
            self.brake=0
        elif (function == ActionType.MEDIUM_DECCELARATION.value):                                                                                      # deccelarate medium
            self.accelaration-= 0.4
            self.brake=0
        elif (function == ActionType.HIGH_DECCELARATION.value):                                                                                      # deccelarate heavily
            self.accelaration-= 0.6
            self.brake=0
        elif (function == ActionType.SLOW_RELEASE_BREAK.value):                                                                                       #slighty decrease break
            self.accelaration= 0.0
            self.brake-=0.1
        elif (function == ActionType.MEDIUM_RELEASE_BREAK.value):                                                                                       #  increase break medium
            self.accelaration=0.0
            self.brake-= 0.3
        elif (function == ActionType.HIGH_RELEASE_BREAK.value):                                                                                       # heavily increase break
            self.accelaration= 0.0
            self.brake-=0.5
        
        self.brake= max(0.0,self.brake)
        self.brake= min(1.0,self.brake)
        self.accelaration=max(0.0,self.accelaration)
        self.accelaration=min(1.0,self.accelaration)
        self.set_up_car.move(self.accelaration,self.brake)
        velocity= self.set_up_car.vehicle.get_velocity()
        done= False
        reward=1
        kmh= int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        if (len(self.set_up_car.collision_hist) != 0):
            done =True
            reward = -1
        elif kmh <10:  
            done =False
            reward += -0.01
        if self.previous_brake==0 and self.brake==1.0:
            reward += -0.3
        if (self.brake - self.previous_brake)>=0.5:
            reward= -10
        if(self.accelaration-self.previous_accelaration)>=0.5:
            reward= -10
        # detect obstacle upfront to check if safety distance exists penalize if not |||| this will be added
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            print("time is up")
            done=True
        return reward, done
    


        


if __name__ == '__main__':
    env = CarEnv()
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"Episode: {episode + 1}")  # Debug: Print the current episode
        env.reset()
        done = False
        step_count = 0
        while not done:
            action = 1
            reward, done = env.step(action)
            
            step_count += 1
            #print(f"Step: {step_count}, Reward: {reward}, Done: {done}")
        env.set_up_car.destroy_all_actors()
        
        
        

