"""
You will design your lateral and longitudinal controllers in this
script and execute the main.py script to check.

This script interfaces with the Carla server, receives the states
 of the actor vehicles and sends the control commands.
[Script to Control a Single Vehicle]

Author: Ashish Roongta
Copyright @ SafeAI lab-Carnegie Mellon University
"""

import numpy as np
import matplotlib.pyplot as plt
import control
import os
import scipy
from scipy.ndimage import gaussian_filter1d
import scipy.signal
import time

#  Define the Controller2D class and all functions below:

class Controller2D(object):
    def __init__(self, vehicle,waypoints,carla):
        """
        Constructor
        [Input]
        *self: pointer to the self-class
        *vehicle: pointer to the Carla vehicle-actor
        *waypoints: the reference trajectory [x,y,yaw]
        *carla: pointer to the carla library imported
        
        [The variable names are indicative of what they represent. 
        Additional comments have been added to clarify wherever needed]

        Remember: the control commands for vehicle in Carla includes the following-
        *throttle [0,1.0]
        *brake [0,1.0]
        *steer [-1.0,1.0]
        
        *** Carla works on a Left Hand Coordinate system(LHS), so the angles are
        positive in clockwise direction. To convert the states into Right Hand 
        coordinate system(RHS), all the angles and values along Y-axis are needed to 
        multiplied with - sign; i.e. y(RHS)=-y(LHS) and angle(RHS)=-angle(LHS).
        All computations and calculations are to be done in the RHS.
        """

        self._vehicle=vehicle
        self._controller=carla.VehicleControl()  # pointer to the carla controller interface
        loc=vehicle.get_transform()  # vehicle location object(transform type): [x,y,z]
        self._current_x          = loc.location.x
        self._current_y          = loc.location.y
        self._current_yaw        = loc.rotation.yaw
        self._current_vX      = 0  # Vx: longitudinal velocity of the vehicle
        self._desired_vY      = 0  # Vy: lateral velocity of the vehicle
        self._start_control_loop = True  # boolean to initate the control loop
        self._set_throttle       = 0  # the throttle command value [0,1]
        self._set_brake          = 0  # the brake command value [0,1]
        self._set_steer          = 0  # the steering command value [-1,1]
        self._waypoints          = waypoints
        self._waypoints[:,1]=-self._waypoints[:,1]   # converting LHS to RHS

        self._conv_rad_to_steer  = 0.28   # constant multiplication factor
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self._lr=1.5   #length from rear tire to center of mass
        self._lf=1.0    # length from front tire to the center of mass
        self._Ca=13000  # cornering stiffness of each tire
        self._Iz=3500   # Yaw inertia
        self._f=0.01   # friction coefficient
        phy=vehicle.get_physics_control()
        phy.mass=3500   # setting the mass of vehicle to 200 kg
        vehicle.apply_physics_control(phy)
        self._m=3500   # mass of the vehicle
        self._g=10  # acceleration to the gravity (m/s^2)
        self._last_x=0.0  #to store the last x position of the vehicle
        self._last_y=0.0  # to store the previous y position of the vehicle
        self._t=time.time()  # initiating the time count
        self._last_timestamp=0.0
        self._dt=1/30   # dt for fixed time step, at 30 fps
        self._last_yaw=0.0
        self._look_ahead=5
        self._curv_ld=5
        self._frame=0.0
        self._last_vx_error=0.0
        self.v_desired=8*np.ones(self._waypoints.shape[0])
        #  Member variables for the plots
        self._posx=[]
        self._posy=[]
        self._velx=[]
        self._vely=[]
        self._oryaw=[]
        self._error=[]
        self._throttle=[]
        self._brake=[]
        self._steer=[]
        


    def update_values(self):
        """
        Function to update the state values.
        """
        loc=self._vehicle.get_transform()
        self._current_x         = loc.location.x
        self._current_y         = loc.location.y
        self._current_yaw       = self._vehicle.get_transform().rotation.yaw
        self._current_vX     = self._vehicle.get_velocity().x
        self._current_vY=self._vehicle.get_velocity().y
        self._frame+=1
        # Appending parameters to the plot variables
        # Appending parameters to the plot variables
        self._posx.append(self._current_x)
        self._posy.append(-self._current_y)
        self._velx.append(self._vlong)
        self._vely.append(self._vlat)
        self._oryaw.append(self._fix_yaw)
        self._error.append(self._current_error)
        self._throttle.append(self._set_throttle)
        self._steer.append(self._set_steer)
        self._brake.append(self._set_brake)

    def plot_graphs(self):
        '''
		Function to plot graphs for the states of the vehicle
		'''
        fig,axes=plt.subplots(4,2,figsize=(32,20))
		#  plotting trajectory of the vehicle
        axes[0,0].plot(self._posx,self._posy,color='Green')
        axes[0,0].plot(self._waypoints[:,0],self._waypoints[:,1],color='Red')
        axes[0,0].set_title('Trajectory of vehcile (m)')
        axes[0,0].set_ylabel('Y')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_aspect(aspect=1.0)
        axes[0,0].legend()

        # plotting vx
        axes[1,0].plot(self._velx)
        axes[1,0].set_title('Vehicle Longitudanal Velocity (m/s)')
        axes[1,0].set_ylabel('Vx')
        axes[1,0].set_xlabel('Time')

        # plotting vy
        axes[2,0].plot(self._vely)
        axes[2,0].set_title('Vehicle Lateral Velocity (m/s)')
        axes[2,0].set_ylabel('Vy')
        axes[2,0].set_xlabel('Time')

        #  plotting the vehicle error
        axes[0,1].plot(self._error)
        axes[0,1].set_title('Error (m)')
        axes[0,1].set_ylabel('Error')
        axes[0,1].set_xlabel('Time')

        # plotting throttle
        axes[1,1].plot(self._throttle)
        axes[1,1].set_title('Throttle Command [0,1]')
        axes[1,1].set_ylabel('Throttle')
        axes[1,1].set_xlabel('Time')

        #  PLotting Steering Command
        axes[2,1].plot(self._steer)
        axes[2,1].set_title('Steering Commmand [-1,1] (freq_0.5hzs')
        axes[2,1].set_ylabel('Steer')
        axes[2,1].set_xlabel('Time')

        #  Plotting the vehcile yaw
        axes[3,1].plot(self._oryaw)
        axes[3,1].set_title('Vehicle Yaw (Radians)')
        axes[3,1].set_ylabel('Vehicle Yaw')
        axes[3,1].set_xlabel('Time')

        # plt.show()
        plt.savefig('Response_result.png')

    def dlqr(self,A,B,Q,R):
        '''
        Function to solve the Ricardi equation
        ref http://www.kostasalexis.com/lqr-control.html
        '''
        X=np.matrix(scipy.linalg.solve_discrete_are(A,B,Q,R))
        #  Computing the LQR gain
        K=np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
        return K
    
    def PID_longitudanal(self,dt,vx_error):
        """
        Function to compute the throttle and the brake output using a PID controller.
        """
        kp=15
        kd=0.1
        ki=-150
        integral_error=vx_error+self._last_vx_error
        derivative_error=vx_error-self._last_vx_error
        delta=kp*vx_error+ki*integral_error*dt+kd*derivative_error/dt
        if delta>0:
            throttle_output=delta
            brake_output=0.0
        else:
            throttle_output=0.0
            brake_output=0.5

        self._last_vx_error=vx_error
        return throttle_output,brake_output


    def find_nearest_points(self,X, Y, traj):
        dist_sq = np.zeros(traj.shape[0])
        for j in range(traj.shape[0]):
            dist_sq[j] = (traj[j,0] - X)**2 + (traj[j,1] - Y)**2
        minDistSqure, minIdx = min((dist_sq[i], i) for i in range(len(dist_sq)))
        return np.sqrt(minDistSqure), minIdx
    
    def curvature(self,waypoints):
        '''
        Function to compute the curvature of the reference trajectory.
        Returns an array containing the curvature at each waypoint
        '''
        # waypoints=np.asarray(waypoints)
        x=waypoints[:,0]
        y=waypoints[:,1]
        sig=10
        x1=gaussian_filter1d(x,sigma=sig,order=1,mode="wrap")
        x2=gaussian_filter1d(x1,sigma=sig,order=1,mode="wrap")
        y1=gaussian_filter1d(y,sigma=sig,order=1,mode="wrap")
        y2=gaussian_filter1d(y1,sigma=sig,order=1,mode="wrap")
        curv=np.divide(np.abs(x1*y2-y1*x2),np.power(x1**2+y1**2,3./2))
        return curv
    
    def wrap2pi(self,a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def update_controls(self):
        """
        Function to compute the new control commands and send to the Carla Sevrer.
        The Brake, Throttle and Steering command values need to be computed.
        """
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = (np.pi/180)*self._current_yaw   # converting yaw into radians
        waypoints       = self._waypoints
        last_x=self._last_x
        last_y=self._last_y
        last_yaw=self._last_yaw
        vehicle=self._vehicle
        
        # --Changing LHS to RHS
        yaw=-yaw
        y=-y
        # -------------------
        vX              = self._current_vX
        vY=-self._current_vY
        
        dt=self._dt   #fixed time step dt for fps 3-
        # dt=time.time()-self._t  #  Variable time step dt

        d_yaw=(yaw-self._last_yaw)/dt   # computing yaw rate
        Ca=self._Ca
        Iz=self._Iz
        lr=self._lr
        lf=self._lf
        m=self._m
        # vx,vy=self.d_velocities(dt,x,y,last_x,last_y,yaw)   #callin function to compute the x and y velocites
        # ################## Compute local velocities vx and vy here--------------------------
        vy=vY*np.cos(yaw)-vX*np.sin(yaw)
        vx=vY*np.sin(yaw)+vX*np.cos(yaw) 

        vx=max(vx,0.1)
        # print('vehicle speed={}, vx={},vy={}, yaw={}'.format(v,vx,vy,yaw*180/np.pi))
        curv=self.curvature(waypoints) #computing the curvatue of the reference trajectory at each index
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0
        min_idx=0
        self.v_desired=8   # setting desired speed to a constant 8 m/s

        # Computing instantaneous lateral error for the response plot
        _,min_idx=self.find_nearest_points(x,y,waypoints)
        self._current_error=np.linalg.norm(np.array([waypoints[min_idx,1]-y,waypoints[min_idx,0]-x]))

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            
   
            if self._frame>1:

                ###############################################################
                ###############################################################
                #########< Implement your Lateral Controller Here >############
                ###############################################################

                K=1
                # looping over waypoints to compute the one closest to lookhead distance
                for i in range(len(self._waypoints)):
                    dist = np.linalg.norm(np.array([self._waypoints[i][0] - self._current_x,self._waypoints[i][1] - self._current_y]))
                    if dist>=10.0:   # taking a look ahead distance of 10m
                        ind=i
                        break
                alpha_hat=np.arctan2((self._waypoints[ind][1]-self._current_y),(self._waypoints[ind][0]-self._current_x))
                alpha=alpha_hat-self._current_yaw
                steer_output=np.arctan2(2*2*np.sin(alpha),(K*self._current_speed))

                ################################################################
                ################################################################
                ########< Implement you Longitudinal Controller Here >##########
                # ---------------PID Longitudinal Conntroller--------------------
                kp=15
                kd=-150
                ki=.1
                error=self.v_desired-vx
                integral_error=(error+self._last_vx_error)
                derivative_error=(error-self._last_vx_error)
                delta=kp*error+kd*integral_error*dt+ki*derivative_error/dt
                if delta>0:
                    throttle_output=delta
                    brake_output=0
                else:
                    throttle_output=0
                    brake_output=-delta
                # ------Longitudanal PID control-----------
                # throttle_output,brake_output=self.PID_longitudanal(dt,self.v_desired[min_idx]-vx)
                
                # -----Bang Bang Longitudanal Control------------
                """
                if np.linalg.norm(np.array([vx,vy]))<V_n:
                    throttle_output=1.0
                    brake_output=0.0
                else:
                    throttle_output=0.0
                    brake_output=0.0
                """

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            # self.set_throttle(throttle_output)  # in percent (0 to 1)
            # self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            # self.set_brake(brake_output)        # in percent (0 to 1)
            ############################################################
            self._controller.throttle=throttle_output
            self._controller.steer=max(-1.0,(min(1.0,steer_output)))
            self._controller.brake=brake_output
            vehicle.apply_control(self._controller)
            if min_idx==(len(waypoints)-1):  # destination reached
                self.plot_graphs()  # plotting the response plots
                return True
            # print(throttle_output,max(-1.0,min(1.0,steer_output)),brake_output)
        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        # self._last_timestamp=t
        self._last_x=x
        self._last_y=y
        self._last_yaw=yaw
        self._t=time.time()
        self._last_vx_error=error
        # Storing parameter values for the plots
        self._vlong=vx
        self._vlat=vy
        self._fix_yaw=yaw
        self._set_throttle=throttle_output
        self._set_brake=brake_output
        self._set_steer=steer_output

        return False