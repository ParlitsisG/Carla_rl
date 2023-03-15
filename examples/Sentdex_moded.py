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
    sys.path.append('C:/Users/parli/Desktop/CARLA_0.9.8/WindowsNoEditor/PythonAPI/carla')
    from agents.navigation.local_planner import LocalPlanner

except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

SECONDS_PER_EPISODE=10.0
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
IM_WIDTH=640
IM_HEIGHT=480
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
        self.client.set_timeout(30.0)  # duration for errors
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = self.blueprint_library.filter("model3")[0]
        self.spawn_npc(20,20)


    def reset(self):
        self.collision_hist=[]


        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        self.actor_list.append(self.vehicle)

        self.cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        self.cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        self.cam_bp.set_attribute("fov", "110")

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))  # question specific spawn spots to place the camera except coordinates
        self.camera_sensor = self.world.spawn_actor(self.cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera_sensor)
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
        self.actor_list.append(self.radar_sensor)
        # listen to the radar
        self.radar_sensor.listen(lambda data: self.process_radar(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0 , brake=0.0))
        time.sleep(4)
        colsensor =self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor,spawn_point,attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        #check for imapcts
        self.colsensor.listen (lambda impact: self.process_collision(impact))
        while self.front_camera is None:
            time.sleep(0.01)
        self.episode_start =time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0 , brake=0.0))
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
        if self.Show_metrics:
            print(radar_data.shape)
        self.radar=radar_data

    def move_walker(self, walker):
        max = 100
        target_location = walker.get_location() + carla.Location(x=random.uniform(-max, max),
                                                                 y=random.uniform(-max, max), z=0.0)
        walker.set_target_location(target_location)

    def spawn_npc(self, number_of_vehicles, number_of_walkers):
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
                spawn_point = random.choice(self.world.get_map().get_spawn_points())
                npc = self.world.try_spawn_actor(self.bp, spawn_point)
            npc.set_autopilot(True)
            print('created %s' % npc.type_id)
            self.actor_list.append(npc)
        print('cars done')
        # Spawn the pedestrians
        for i in range(number_of_walkers):
            npc2 = None
            bp2 = random.choice(self.walker_blueprints)
            while npc is None:
                spawn_point = random.choice(self.world.get_map().get_spawn_points())
                npc2 = self.world.try_spawn_actor(bp2, spawn_point)
                # move_walker(npc)
            print('created walker')
            self.actor_list.append(npc2)

    def step(self, function):
        if(function==0):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1 , brake=0.0))
        elif (function == 1):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0))
        elif (function == 2):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, brake=0.0))
        elif (function == 3):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3))
        elif (function == 4):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        velocity= self.vehicle.get_velocity
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
            done=True
        return self.front_camera ,reward, done

class DQNAgent:
    def __init__(self):
        self.learning_rate= 0.001
        self.model =self.create_model()
        self.target_model= self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=Replay_MEMORY_SIZE)
        self.tensorboard =  self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter =0
        self.graph = tf.compat.v1.get_default_graph()


        #flags
        self.terminate= False
        self.last_logged_episode =0
        self.training_initialized =False

    def create_model(self):
        model  = Sequential()
        model.add(Conv2D(256,(3,3), input_shape=(IM_HEIGHT,IM_WIDTH,3 )))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(2, Activation("linear")))
        model.compile(loss="mse",
                      optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]


    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch=random.sample(self.replay_memory, MINI_BATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        with self.graph.as_default():
            future_qs_list = self.model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

            log_this_step = False
            if self.tensorboard.step > self.last_logged_episode:
                log_this_step = True
                self.last_log_episode = self.tensorboard.step

            with self.graph.as_default():
                self.model.fit(np.array(X) / 255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                               callbacks=[self.tensorboard] if log_this_step else None)
            if log_this_step:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_STEPS:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 2)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)



if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    FPS =20
    ep_rewards =[-200]
    np.random.seed(1)
    tf.random.set_seed(1)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    if not os.path.isdir("models"):
        os.makedirs("models")

    agent =DQNAgent()
    print("check dqn agent init")
    env= CarEnv()
    print("check CarEnv init")
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    print("gonna initialize train")
    while not agent.training_initialized:
        time.sleep(0.01)
        print("init training")
    print("init trained")

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((IM_HEIGHT,IM_WIDTH, 3)))
    print("gonna  train")
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # try:

        env.collision_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > e:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, 3)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1 / FPS)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=e)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if e > MIN_EPSILON:
            e *= EPSILON_DECAY
            e= max(MIN_EPSILON, e)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(
        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
