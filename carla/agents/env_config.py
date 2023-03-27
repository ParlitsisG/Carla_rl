import random
import glob
import os
import sys
import math
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from queue import Queue


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH=160
IM_HEIGHT=120
SAFETY_DIST= 2
SHOW_METRICS=True

class NPC():
    def __init__(self):
        self.ai_walker_list=[]
        self.actor_list= []
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(80.0)  # duration for errors
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.world = self.client.get_world()


    def generate_spawn_point_near_location(self, location, radius):
        print("enter generate_spawn_point_near_location")
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_points_near_location = [sp for sp in spawn_points if sp.location.distance(location) <= radius]
        if not spawn_points_near_location:
            #print("exit generate_spawn_point_near_location")
            return random.choice(spawn_points)
        #print("exit generate_spawn_point_near_location")
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
            
            counter=0
            while npc is None:
                spawn_point = self.generate_spawn_point_near_location(main_actor_location, radius=100)
                npc = self.world.try_spawn_actor(self.bp, spawn_point)
                counter +=1
                if counter==10:
                    print("failed to spawn car")
                    break
                    

            npc.set_autopilot(True)
            print('created %s' % npc.type_id)
            if npc is not None:
                self.actor_list.append(npc)
        print('cars done')



        SpawnActor = carla.command.SpawnActor
        # Spawn the pedestrians
        self.walker_controller_list = []
        batch = []
        walker_speed = []
        walkers_list = []
        spawn_points = []
        max_attempts = 10  # Maximum number of attempts for each pedestrian
        for i in range(number_of_walkers):
            spawn_point = self.generate_spawn_point_near_location(main_actor_location, radius=100)
            print(f"found spot {i + 1}")
            spawn_points.append(spawn_point)

       
        #how many run how many walk
        pedestrian_split=0.5
        for spawn_point in spawn_points:
            walker_bp = random.choice(self.walker_blueprints)
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > pedestrian_split):
                # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            batch.append(SpawnActor(walker_bp, spawn_point))
        results =  self.client.apply_batch_sync(batch, True)


        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results =  self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            self.walker_controller_list.append(walkers_list[i]["con"])
            self.walker_controller_list.append(walkers_list[i]["id"])
        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        #self.world.tick()
        
    def destroy_all_actors(self):
        print("initialize_npc_destroy")
        for controller in (self.ai_walker_list):
            print("initialize_npc_destroy")
            for controller in (self.ai_walker_list):
                if controller is not None:
                    try:
                        controller.stop()
                        print(f"Destroying walker controller: {controller.id}")
                        controller.destroy()
                        print(f"Destroyed walker controller: {controller.id}")
                    except Exception as e:
                        print(f"Error while destroying walker controller: {controller.id}, {e}")
        
        for actor in reversed(self.actor_list):
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
            self.actor_list = [] 
            self.ai_walker_list=[]

class Sync_Actor():
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(30.0)  # duration for errors
        self.synchronous_mode = True
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = self.blueprint_library.filter("model3")[0]
        self.npc_manager= NPC()
        self.show_metrics=SHOW_METRICS
        self.actor_list=[]
        self.vehicle = None
        self.depth_camera_data= None
        self.segmentation_camera_data= None
        self.radar_data= None

    def spawn_car(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = self.synchronous_mode
        settings.fixed_delta_seconds = None# You can set a fixed time step for each tick here
        self.world.apply_settings(settings)


        self.collision_hist=[]
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        self.actor_list.append(self.vehicle)

        self.npc_manager.spawn_npc(10,10,self.spawn_point.location)
        
        queue = Queue()
        

        ###
        ### SEMANTIC CAMERA
        ##

        self.cam_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.cam_bp.set_attribute("fov", "60")
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))  # question specific spawn spots to place the camera except coordinates
        self.camera_sensor = self.world.spawn_actor(self.cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera_sensor)
        # listen to the camera
        self.camera_sensor.listen(lambda data: self.process_semantic_image(queue,data))

        ###
        ### DEPTH CAMERA
        ##
        self.depth_cam_bp = self.blueprint_library.find("sensor.camera.depth")
        self.depth_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.depth_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.depth_cam_bp.set_attribute("fov", "60")

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7)) 
        self.depth_camera_sensor = self.world.spawn_actor(self.depth_cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.depth_camera_sensor)
        # listen to the camera
        self.depth_camera_sensor.listen(lambda depth_data: self.depth_process_image(queue,depth_data))

        ###
        ### RADAR
        ##
        # Get the blueprint for the radar sensor
        self.radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        # Set the attributes for the radar sensor
        self.radar_bp.set_attribute('horizontal_fov', '30')
        self.radar_bp.set_attribute('vertical_fov', '30')
        self.radar_bp.set_attribute('points_per_second', str(1000))
        self.radar_bp.set_attribute('range', '50')
        spawn_point = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.8))
        self.radar_sensor = self.world.spawn_actor(self.radar_bp, spawn_point, attach_to=self.vehicle)
        
        self.actor_list.append(self.radar_sensor)

        # listen to the radar
        self.radar_sensor.listen(lambda data: self.process_radar(queue,data))

        ###
        ### COLLISION
        ##          

        colsensor =self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor,spawn_point,attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        #check for imapcts
        self.colsensor.listen (lambda impact: self.process_collision(impact))

        print("outside")
        self.world.tick()
        
        print("inside")
        sensor_data ={'segmentation': None ,'depth': None , 'radar':None}
        for i in range(3):
            name, data = queue.get(True, 1.0)
            sensor_data[name] = data
            self.depth_camera_data= sensor_data['depth']
            self.segmentation_camera_data =  sensor_data['segmentation']
            self.radar_data = sensor_data['radar']
        
        
        


        
    def process_collision(self, impact):
        self.collision_hist.append(impact)

    def process_semantic_image(self,queue,image):
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
        if (self.show_metrics):
            # Display the segmented image
            cv2.imshow('Segmented Image', result)
            cv2.waitKey(1)
        queue.put( ('segmentation', result), True, 1.0 )

    def depth_process_image(self,queue,image):
        i = np.array(image.raw_data)
        # Reshape the input image
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        normalized =np.zeros((IM_HEIGHT, IM_WIDTH, 1))
        # Extract the depth information from the image
        normalized = (i2[:, :, 0] + i2[:, :, 1] * 256 + i2[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1)
        depth = 1000* normalized
        queue.put( ('depth', depth), True, 1.0 )
        # Convert the depth values to logarithmic scale
        log_depth = np.log(depth + 1)
        log_depth_norm = (log_depth - np.min(log_depth)) / (np.max(log_depth) - np.min(log_depth))
        # Display the processed image if show_metrics is True
        if self.show_metrics:
            cv2.imshow("depth", log_depth_norm)
            cv2.waitKey(1)
        # Normalize the image so the values are from 0 to 1 for faster processing
        

    def process_radar(self,queue,data):
        # Get the radar data as a numpy array
        radar_measurment = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        radar_measurment=radar_measurment.reshape((radar_measurment.shape[0]//4,4))
       
        
        radar_image = np.ones((60,60,2), dtype=float) * 100
        horizontal_radians = np.radians(15.0)
        vertical_radians = np.radians(15.0)

        iterations=min(len(radar_measurment),60*60)
        for i in range (iterations):
            vertical=min(radar_measurment[i][0],vertical_radians)
            vertical=max( vertical,-vertical_radians)
            horizontal=min(radar_measurment[i][1],horizontal_radians)
            horizontal=max(horizontal,-horizontal_radians)

            normalized_vertical_angle = (vertical +vertical_radians)/  (vertical_radians*2)
            normalized_horizontal_angle =(horizontal+horizontal_radians) / (horizontal_radians*2)
        
            normalized_horizontal_angle=int(normalized_horizontal_angle*59.0)
            normalized_vertical_angle= int(normalized_vertical_angle*59.0) 
            radar_image[normalized_vertical_angle][normalized_horizontal_angle][0]= radar_measurment[i][2]
            radar_image[normalized_vertical_angle][normalized_horizontal_angle][1]= radar_measurment[i][3]
        debug_image=np.zeros((60,60), dtype=float)
        for i in range (60):
            for j in range (60):
                debug_image[i][j]=radar_image[i][j][0]
        
        queue.put( ('radar', radar_image), True, 1.0 )
        if self.show_metrics:
            cv2.namedWindow('Radar', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Radar', 600, 600)
            cv2.imshow('Radar', debug_image)
            cv2.waitKey(1)


    def move(self,accelarate , press_brake):
        self.vehicle.apply_control(carla.VehicleControl(throttle=accelarate, brake=press_brake))


    def destroy_all_actors(self):
        self.npc_manager.destroy_all_actors()
        print("initialize_CAR_destroy")
        for actor in reversed(self.actor_list):
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
            self.actor_list = [] 
    
    def detect_obstacle(self):
        safe=True
        self.show_debug()
        print("breaching")
        distance = 50
        safety_distance_breached=0
        for i in range (IM_HEIGHT):
            for j in range(IM_WIDTH):
                # if green or red
                if self.segmentation_camera_data[i][j][0]!= 0 or self.segmentation_camera_data[i][j][1] !=0:
                    obs_exists=True
                    if self.depth_camera_data[i][j] < 2.0:
                        safe=False
                        if(distance>self.depth_camera_data[i][j]):
                            distance= self.depth_camera_data[i][j]
        velocity= self.vehicle.get_velocity()
        kmh= int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        safe_distance =(kmh* (1000/3600) / (2 * 7.5))  #7.5 decelaration for standard conditions
        if distance<50 and safe_distance-distance<0:
            safety_distance_breached=safe_distance-distance
        return distance , safe ,safety_distance_breached

    def show_debug(self):
            cv2.imshow("depth", self.depth_camera_data)
            cv2.imshow("segmentation", self.segmentation_camera_data)
            cv2.waitKey(1)
                    


    


    class env_dict():
        pass