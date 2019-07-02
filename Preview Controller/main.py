"""
This will the main script, this is what you have to run.
This script communicates with the Carla Server to send control
commands and recieve the states of the actors. 
This script also calls the controller.py script to send the vehicle
state and receive the next control commands.
[To control a sicngle vehicle using the controller designed in controller.py]

Author: Ashish Roongta
Copyright @ SafeAI lab-Carnegie Mellon University
"""

from __future__ import print_function

import numpy as np
import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random


# -------importing the Controller that you will design-------------
from controller import *

#  -------impoting camera_manager
from camera_manager import *

# ------importing plotter-----------------
from plot_states import *

class controller1():
    def __init__(self):
        # Defining member variables------------
        self.no_rendering=True           # pygame window toggle boolean
        self.world=None                  # pointer to the Carla world object
        self.map=None                    # pointer to the Carla Map
        self.ref_traj=None               # to sotre the reference trajectory
        self.contoller=None              # pointer to the object for thr controller class you will design
        self.spawn_point=None            # vehicle spawn point [initial location and orientation]
        self.player=None                 # pinter to the actor-vehicle
        self._recording_enabled=True     # toggle for the camera record functionality

    def _render(self,world):
        '''
        Function to disable rendering
        '''
        settings=world.get_settings()
        settings.no_rendering_mode=True
        world.apply_settings(settings)

    def actor_spawn(self):
        '''
        Function to spawn the actor
        '''
        spawn_point1=random.choice(self.map.get_spawn_points())  # Randomly choosing a spawn point by querying from the map
        
        # Computing the initial (x,y) location and yar orientation of the vehicle
        # from the reference trajectory.
        spawn_point1.location.x=self.ref_traj[0,0]
        spawn_point1.location.y=self.ref_traj[0,1]
        spawn_point1.location.z=2
        spawn_point1.rotation.pitch=0.00193294
        spawn_point1.rotation.yaw=self.ref_traj[0,2]
        spawn_point1.rotation.roll=-0.00494385

        self.spawn_point=spawn_point1

        # Specifying the vehicle type and make to be spawned- [choosing a BMW Grandtourer here]
        blueprint_library=self.world.get_blueprint_library()
        bp=random.choice(blueprint_library.filter('vehicle.bmw.grandtourer'))
        color=random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color',color)
        
        self.player=self.world.spawn_actor(bp,self.spawn_point)


        
    def game_loop(self,args):
        '''
        Function to execute the game loop.
        Instantiates the carla world, connects to the client
        '''
        try:
            client=carla.Client(args.host,args.port)
            client.set_timeout(2.0)

            self.world=client.get_world()  # recieveing the world info from the simulator
            self.map=self.world.get_map()
            if self.no_rendering:  # Disabling the rendering 
                self._render(self.world)
            
            # self.ref_traj=np.load('Trajectory/Buggy_Track03.npy')
            # self.ref_traj=np.load('Trajectory/Uber_Route.npy')
            self.ref_traj=np.load('Trajectory/Buggy_Track02.npy')
            self.actor_spawn()  #spawning the actor

            # -----Instantiating the Controller class that you designed in controller.py
            self.contoller=Controller2D(self.player,self.ref_traj,carla)
            """
            The constructor for the controller class that you design should take in the 
            following arguments:
            *pointer to the vehicle-actor
            *the reference trajectory
            *pointer to the carla library imported
            """
            # spawining and attaching a camera sensor to the vehicle actor
            self.camera_manager=CameraManager(self.player)
            cam_index = self.camera_manager._index if self.camera_manager is not None else 0
            cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0    
            self.camera_manager._transform_index=cam_pos_index
            cam_index=0
            self.camera_manager.set_sensor(cam_index,notify=False)
            if self._recording_enabled:
                CameraManager._recording=True
                client.start_recorder("manual_recording.rec")

            while True:
            """
            fucntions from the controller class you designed will be called here
            to communicate vehicle states & control commands.
            """
                self.contoller.update_values()
                if self.contoller.update_controls():
                    print('Completed........!!')
                    self.player.destroy()
                    break
        finally:
            if self.player is not None:
                self.player.destroy()   # destroying the vehicle player
            
            if self._recording_enabled:
                client.stop_recorder()   # stopped recording
            
            if self.camera_manager is not None:
                self.camera_manager.sensor.destroy()   # destroying the camera sensor
            
            

def main():
    argparser=argparse.ArgumentParser(description='CARLA Control Client')
    argparser.add_argument('-v', '--verbose',action='store_true',dest='debug',help='print debug information')
    argparser.add_argument('--host',metavar='H',default='127.0.0.1',help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port',metavar='P',default=2000,type=int,help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot',action='store_true',help='enable autopilot')
    argparser.add_argument('--res',metavar='WIDTHxHEIGHT',default='1280x720',help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter',metavar='PATTERN',default='vehicle.*',help='actor filter (default: "vehicle.*")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    # log_level = logging.DEBUG if args.debug else logging.INFO
    # logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # logging.info('listening to server %s:%s', args.host, args.port)

    # print(__doc__)

    try:
        cnlr=controller1()   # instantiating the controller class defined above
        cnlr.game_loop(args)  # initiating the game loop

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__=='__main__':
    main()