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
SECONDS_PER_EPISODE=15.0
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


class CarEnv:
    Show_metrics= SHOW_METRICS
    front_camera = None
    radar =None
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(80.0)  # duration for errors
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = self.blueprint_library.filter("model3")[0]
        


    def reset(self):
        self.collision_hist=[]

      
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        actor_list.append(self.vehicle)

        self.spawn_npc(0,0,self.spawn_point.location)

        self.cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.cam_bp.set_attribute("fov", "110")

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))  # question specific spawn spots to place the camera except coordinates
        self.camera_sensor = self.world.spawn_actor(self.cam_bp, spawn_point, attach_to=self.vehicle)
        actor_list.append(self.camera_sensor)
        # listen to the camera
        self.camera_sensor.listen(lambda data: self.process_img(data))
        # Get the blueprint for the radar sensor
        self.radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        # Set the attributes for the radar sensor
        self.radar_bp.set_attribute('horizontal_fov', '40')
        self.radar_bp.set_attribute('vertical_fov', '30')
        self.radar_bp.set_attribute('points_per_second', str(2000))
        self.radar_bp.set_attribute('range', '50')
        spawn_point = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.8))
        self.radar_sensor = self.world.spawn_actor(self.radar_bp, spawn_point, attach_to=self.vehicle)
        actor_list.append(self.radar_sensor)
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


    def process_img(self,image):
        i = np.array(image.raw_data)
        #print(i.shape)# check image input
        i2=i.reshape((IM_HEIGHT,IM_WIDTH,4))
        i3=i2[:,:,:3] #if we want to exclude alpha
        if self.Show_metrics:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera= i3/255 # normalize so the values are from 0 to 1 for faster proccessing

    def process_radar(self,data):
        # Get the radar data as a numpy array
        radar_data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        radar_data=radar_data.reshape((radar_data.shape[0]//4,4))
       # if self.Show_metrics:
           # print(radar_data.shape)
        self.radar=radar_data

    def move_walker(self, walker):
        max_distance = 100
        target_location = walker.get_location() + carla.Location(x=random.uniform(-max_distance, max_distance),
                                                                y=random.uniform(-max_distance, max_distance), z=0.0)
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        walker_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
        actor_list.append(walker_controller)
        self.walker_controller_list.append(walker_controller)
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
        if(function==0):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1 , brake=0.0))
        elif (function == 1):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0))
        elif (function == 2):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, brake=0.0))
        elif (function == 2):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, brake=0.0))
        elif (function == 2):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, brake=0.0))
        elif (function == 2):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0))
        elif (function == 3):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        elif (function == 4):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        velocity= self.vehicle.get_velocity()
        done= False
        kmh= int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        if (len(self.collision_hist) != 0):
            done =True
            reward = -200
        elif kmh <30:  # will change into above speed limit and
            done =False
            reward= -1
        else:
            done = False
            reward=1
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            print("time is up")
            done=True
        return self.front_camera ,reward, done

def destroy_all_actors():
    global actor_list
    print(" initialize destroy")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    print("destroyed")
    actor_list = []

if __name__ == '__main__':
    env = CarEnv()
    num_episodes = 10
    for episode in range(num_episodes):
        print(f"Episode: {episode + 1}")  # Debug: Print the current episode
        env.reset()
        done = False
        step_count = 0
        while not done:
            action = 2
            _, reward, done = env.step(action)
            step_count += 1
            print(f"Step: {step_count}, Reward: {reward}, Done: {done}")
            time.sleep(0.1)
        destroy_all_actors()

