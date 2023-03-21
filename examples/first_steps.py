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
SECONDS_PER_EPISODE=4.0
Replay_MEMORY_SIZE =  5000
MIN_REPLAY_MEMORY_SIZE=1000
MINI_BATCH_SIZE=16
PREDICTION_BATCH_SIZE =1
TRAINING_BATCH_SIZE = MINI_BATCH_SIZE//4
UPDATE_TARGET_STEPS = 5
actor_list= []
MEMORY_FRACTION=0.2
MIN_REWARD =-200
DISCOUNT=0.99
EPISODES =100
e=1
IM_WIDTH=160
IM_HEIGHT=120
e_decay=0.95
SHOW_METRICS=True
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
    show_metrics= SHOW_METRICS
    front_camera = None
    radar =None
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(80.0)  # duration for errors
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = self.blueprint_library.filter("model3")[0]
        self.accelaration= 0
        self.brake= 0
        self.previous_accelaration=0
        self.previous_brake=0
        self.ai_walker_list=[]
        self.front_depth_camera= None
        self.radar= None
        self.front_segmentation_camera= None 

        


    def reset(self):

        self.destroy_all_actors()

        self.collision_hist=[]
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        actor_list.append(self.vehicle)

        #self.spawn_npc(10,5,self.spawn_point.location)

        self.cam_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.cam_bp.set_attribute("fov", "60")


        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))  # question specific spawn spots to place the camera except coordinates
        self.camera_sensor = self.world.spawn_actor(self.cam_bp, spawn_point, attach_to=self.vehicle)
        actor_list.append(self.camera_sensor)



        self.depth_cam_bp = self.blueprint_library.find("sensor.camera.depth")
        self.depth_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.depth_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.depth_cam_bp.set_attribute("fov", "60")

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7)) 
        self.depth_camera_sensor = self.world.spawn_actor(self.depth_cam_bp, spawn_point, attach_to=self.vehicle)
        actor_list.append(self.depth_camera_sensor)


        # listen to the camera
        self.camera_sensor.listen(lambda data: self.process_semantic_image(data))
        self.depth_camera_sensor.listen(lambda depth_data: self.depth_process_image(depth_data))
        # Get the blueprint for the radar sensor
        self.radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        # Set the attributes for the radar sensor
        self.radar_bp.set_attribute('horizontal_fov', '30')
        self.radar_bp.set_attribute('vertical_fov', '30')
        self.radar_bp.set_attribute('points_per_second', str(1000))
        self.radar_bp.set_attribute('range', '50')
        spawn_point = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.8))
        self.radar_sensor = self.world.spawn_actor(self.radar_bp, spawn_point, attach_to=self.vehicle)
        
        actor_list.append(self.radar_sensor)


        # Convert the horizontal and vertical FOV to radians
        horizontal_fov_rad = np.radians(float(self.radar_bp.get_attribute('horizontal_fov')))
        vertical_fov_rad = np.radians(float(self.radar_bp.get_attribute('vertical_fov')))
        min_azimuth_angle_rad = -horizontal_fov_rad / 2
        max_azimuth_angle_rad = horizontal_fov_rad / 2
        min_altitude_angle_rad = -vertical_fov_rad / 2
        max_altitude_angle_rad = vertical_fov_rad / 2
        print("yo")
        print(min_altitude_angle_rad ,"-",max_altitude_angle_rad)
        print(min_altitude_angle_rad ,"-",max_altitude_angle_rad)

        # listen to the radar
        self.radar_sensor.listen(lambda data: self.process_radar(data))
        colsensor =self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor,spawn_point,attach_to=self.vehicle)
        actor_list.append(self.colsensor)
        #check for imapcts
        self.colsensor.listen (lambda impact: self.process_collision(impact))
        while self.front_camera is None:
            time.sleep(0.01)
        self.episode_start =time.time()
        return self.front_camera, self.radar

    def process_collision(self, impact):
        self.collision_hist.append(impact)


    def process_image(self,image):
        i = np.array(image.raw_data)
        #print(i.shape)# check image input
        i2=i.reshape((IM_HEIGHT,IM_WIDTH,4))
        i3=i2[:,:,:3] #if we want to exclude alpha
        if self.show_metrics:
            cv2.imshow("segmantation",i3)
            cv2.waitKey(1)
        self.front_camera= i3/255 # normalize so the values are from 0 to 1 for faster proccessing

    def process_semantic_image(self,image):
        # Convert the semantic segmentation image to a NumPy array
        img_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img_data = img_data.reshape((image.height, image.width, 4))

        # Extract the segmentation labels
        segmentation_labels = img_data[:, :, 2]

        # Define the color mapping for pedestrians, cars, and background
        color_map = {
            0: [0, 0, 0],      # Background
            4: [255, 0, 0],    # Pedestrians
            10: [0, 255, 0],   # Cars
        }

        # Apply the color mapping
        result = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        for label, color in color_map.items():
            result[segmentation_labels == label] = color
        #if (self.show_metrics):
            # Display the segmented image
            #cv2.imshow('Segmented Image', result)
            #cv2.waitKey(1)
        self.front_segmentation_camera =result

    def depth_process_image(self, image):
        i = np.array(image.raw_data)
        # Reshape the input image
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        normalized =np.zeros((IM_HEIGHT, IM_WIDTH, 1))
        # Extract the depth information from the image
        normalized = (i2[:, :, 0] + i2[:, :, 1] * 256 + i2[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1)
        depth = 1000* normalized
        # Convert the depth values to logarithmic scale
        log_depth = np.log(depth + 1)
        log_depth_norm = (log_depth - np.min(log_depth)) / (np.max(log_depth) - np.min(log_depth))
        # Display the processed image if show_metrics is True
        #if self.show_metrics:
            #cv2.imshow("depth", log_depth_norm)
            #cv2.waitKey(1)
        # Normalize the image so the values are from 0 to 1 for faster processing
        self.front_depth_camera = log_depth_norm

    def process_radar(self,data):
        # Get the radar data as a numpy array
        radar_data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        radar_data=radar_data.reshape((radar_data.shape[0]//4,4))
        if self.show_metrics:
           print(radar_data.shape)
        self.radar=radar_data
        radar_image = np.ones((60,60,2), dtype=float) * 100
        horizontal_radians = np.radians(float(self.radar_bp.get_attribute('horizontal_fov')))
        vertical_radians = np.radians(float(self.radar_bp.get_attribute('vertical_fov')))


        for i in range (len(radar_data)):
            radar_data[i][0]
            normalized_vertical_angle = int(((radar_data[i][0] + vertical_radians/2 ) /  vertical_radians)*60)
            normalized_horizontal_angle =int(((radar_data[i][1] + horizontal_radians/2 ) / horizontal_radians)*60)
            radar_image[normalized_vertical_angle][normalized_horizontal_angle][0]= radar_data[i][2]
            radar_image[normalized_vertical_angle][normalized_horizontal_angle][1]= radar_data[i][3]
        debug_image=np.zeros((60,60), dtype=float)
        for i in range (60):
            for j in range (60):
                debug_image[i][j]=radar_image[i][j][0]
                

        
        
        cv2.namedWindow('Radar', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Radar', 600, 600)
        cv2.imshow('Radar', debug_image)
        cv2.waitKey(1)




    def move_walker(self, walker):
        max_distance = 100
        target_location = walker.get_location() + carla.Location(x=random.uniform(-max_distance, max_distance),
                                                                y=random.uniform(-max_distance, max_distance), z=0.0)
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        walker_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
        self.ai_walker_list.append(walker_controller)
        walker_controller.start()
        walker_controller.go_to_location(target_location)


    def generate_spawn_point_near_location(self, location, radius):
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_points_near_location = [sp for sp in spawn_points if sp.location.distance(location) <= radius]
        if not spawn_points_near_location:
            return random.choice(spawn_points)
        return random.choice(spawn_points_near_location)

    def spawn_npc(self, number_of_vehicles, number_of_walkers,main_actor_location):
        # Get the blueprint for all vehicles and all walkers
        self.walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')
        self.vehicle_blueprints =self.blueprint_library.filter('vehicle.*.*')
        
        # Spawn the vehicles
        for i in range(number_of_vehicles):
            # Choose a random blueprint for the vehicle
            # Spawn the vehicle at a random location in the world
            npc = None
            self.bp = random.choice(self.vehicle_blueprints)
            while npc is None:
                spawn_point = self.generate_spawn_point_near_location(main_actor_location, radius=100)
                npc = self.world.try_spawn_actor(self.bp, spawn_point)
            npc.set_autopilot(True)
            print('created %s' % npc.type_id)
            actor_list.append(npc)
        print('cars done')
        # Spawn the pedestrians
        self.walker_controller_list = []
        max_attempts = 10  # Maximum number of attempts for each pedestrian
        for i in range(number_of_walkers):
            npc2 = None
            bp2 = random.choice(self.walker_blueprints)
            for _ in range(max_attempts):
                spawn_point = self.generate_spawn_point_near_location(main_actor_location, radius=100)
                npc2 = self.world.try_spawn_actor(bp2, spawn_point)
                if npc2 is not None:
                    break
            if npc2 is None:
                print(f"Failed to spawn walker {i + 1} after {max_attempts} attempts")
                continue
            self.move_walker(npc2)
            print(f"Created walker {i + 1}")
            actor_list.append(npc2)
        

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
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.accelaration , brake=self.brake))

        velocity= self.vehicle.get_velocity()
        done= False
        reward=1
        kmh= int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        if (len(self.collision_hist) != 0):
            done =True
            reward = -200
        elif kmh <10:  
            done =False
            reward += -1
        if self.previous_brake==0 and self.brake==1.0:
            reward += -20
        if (self.brake - self.previous_brake)>=0.5:
            reward= -10
        if(self.accelaration-self.previous_accelaration)>=0.5:
            reward= -10
        # detect obstacle upfront to check if safety distance exists penalize if not |||| this will be added
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            print("time is up")
            done=True
        return self.front_segmentation_camera, self.front_depth_camera, self.radar ,reward, done


    def destroy_all_actors(self):
        global actor_list
        print("initialize destroy")
        for controller in (self.ai_walker_list):
          if controller is not None:
            try:
                controller.stop()
                print(f"Destroying walker controller: {controller.id}")
                controller.destroy()
                print(f"Destroyed walker controller: {controller.id}")
            except Exception as e:
                print(f"Error while destroying walker controller: {controller.id}, {e}")
      
        for actor in reversed(actor_list):
            if actor is not None:
                try:
                    print(f"Destroying actor: {actor.type_id}")
                    if 'vehicle' in actor.type_id:
                        actor.set_autopilot(enabled=False)
                        print(f"Destroying actor: {actor.id}")
                    actor.destroy()
                    print(f"Destroying actor: {actor.id}")
                except Exception as e:
                    print(f"Error while destroying actor: {actor.id}, {e}")
        print("destroyed")
        actor_list = [] 
        self.ai_walker_list=[]
        


if __name__ == '__main__':
    env = CarEnv()
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"Episode: {episode + 1}")  # Debug: Print the current episode
        env.reset()
        done = False
        step_count = 0
        while not done:
            action = 4
            _,_,_, reward, done = env.step(action)
            step_count += 1
            #print(f"Step: {step_count}, Reward: {reward}, Done: {done}")
        env.destroy_all_actors()

