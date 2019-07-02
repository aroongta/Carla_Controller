"""
This is an implementation of the Preview Control from the paper:
'Design, Analysis and Experiments of Preview Path Tracking 
Control for Autonomous Vehicles; Shaobin XU and Huei Peng'.

Author: Ashish Roongta
SafeAI lab
Carnegie Mellon University

Copyright @ SafeAI lab-Carnegie Mellon University
"""

from BuggySimulator import *
import numpy as np
import scipy
import control
from scipy.ndimage import gaussian_filter1d
from util import *
import scipy.signal

def curvature(traj):
    # calculate curvature
    x = traj[:, 0]
    y = traj[:, 1]
    sig = 20
    x1 = gaussian_filter1d(x, sigma=sig, order=1, mode='wrap')
    x2 = gaussian_filter1d(x1, sigma=sig, order=1, mode='wrap')
    y1 = gaussian_filter1d(y, sigma=sig, order=1, mode='wrap')
    y2 = gaussian_filter1d(y1, sigma=sig, order=1, mode='wrap')
    return np.abs(x1*y2 - y1*x2) / np.power(x1**2 + y1**2, 3./2)

def dlqr(A,B,Q,R):
        #ref http://www.kostasalexis.com/lqr-control.html
        #Solve ricatti equation
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        #Compute LQR gain
        # K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
        return X

def compute_psi(A,B,R,Ro):
    """
    Function to compute Psi
    """
    psi=np.identity(4)+np.matmul(Ro,np.matmul(B,B.T))/R
    # print(psi.shape)
    return np.matmul(A.T,np.linalg.inv(psi))

def controller(traj,vehicle,curv):

    lr = vehicle.lr
    lf = vehicle.lf
    Ca = vehicle.Ca
    Iz = vehicle.Iz
    f = vehicle.f
    m = vehicle.m
    g = vehicle.g

    delT = 0.05

    #reading current vehicle states
    X = vehicle.state.X
    Y = vehicle.state.Y
    xdot = vehicle.state.xd
    ydot = vehicle.state.yd
    phi = vehicle.state.phi
    phidot = vehicle.state.phid
    delta = vehicle.state.delta

    mindist, index = closest_node(X, Y, traj)
    #print(index)
        
    lkd=100    # number of steps to look ahead

    #Vx = 20-60*np.exp(-1/np.amax(np.absolute(curv[index:index+idxvel_fwd])))
    
    # curv_fwd = np.amax(np.absolute(curv[index:index+idxvel_fwd]))
    # if curv_fwd>0.4:
    #     Vx = 6
    # elif curv_fwd>0.3:
    #     Vx = 20
    # elif curv_fwd>0.2:
    #     Vx = 30
    # elif curv_fwd>0.1:
    #     Vx = 30
    # else:
    #     Vx = 30
    
    # Vx = max(Vx,6)
    Vx = 6

    # #Ref Eq (2.45) Vehicle Dynamics and Control by Rajesh Rajamani
    # A = [[0,1,0,0],[0,-4*Ca/(m*Vx),4*Ca/m,2*Ca*(lr-lf)/(m*Vx)],[0,0,0,1],[0,2*Ca*(lr-lf)/(Iz*Vx),2*Ca*(lf-lr)/Iz,-2*Ca*(lr*lr+lf*lf)/(Iz*Vx)]]
    # B = [[0],[2*Ca/m],[0],[2*Ca*lf/Iz]]
    # #B = [[0,0],[2*Ca/m,-Vx+2*Ca*(lr-lf)/(m*Vx)],[0,0],[2*Ca*lf/Iz,-2*Ca*(lr*lr+lf*lf)/(Iz*Vx)]]
    
    # C = np.identity(4)
    
    # D = [[0],[0],[0],[0]]
    # #D = [[0,0],[0,0],[0,0],[0,0]]
    
    #  Lumped Coefficients
    s1=2*(Ca+Ca)   # assuming same Ca for the front and rear wheels
    s2=-2*(lf*Ca-lr*Ca)
    s3=-2*(lf*lf*Ca+lr*lr*Ca) 
    
    # C = np.identity(4)
    #D = [[0,0],[0,0],[0,0],[0,0]]

    A=np.array([[0,1,0,0],[0,-s1/(m*xdot),s1/m,s2/(m*xdot)],[0,0,0,1],[0,s2/(Iz*xdot),-s2/Iz,s3/(Iz*xdot)]])
    B=np.array([[0],[2*Ca/m],[0],[2*lf*Ca/Iz]])
    C=np.identity(4)
    D=np.array([0,(s2/m)-xdot*xdot,0,s3/Iz]).reshape(-1,1)
    
    Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,100,0],[0,0,0,100]])
    
    R = 100
 
    #State space system (continuous)
    syscont = scipy.signal.StateSpace(A,B,C,D)
    
   #Discretizing state space system
    sysdisc = syscont.to_discrete(delT)
    Ad = sysdisc.A
    Bd = sysdisc.B
    Dd=sysdisc.D
    
    # Formulating the Augmented State Space System Matrices:
    D_=np.hstack((Dd,np.zeros((4,lkd))))
    lam=np.hstack((np.zeros((lkd,1)),np.identity(lkd)))
    lam=np.vstack((lam,np.zeros((1,lkd+1))))

    #Calculating LQR gain
    Ro = dlqr(Ad,Bd,Q,R)
    
    #  Computing Psi
    Psi=compute_psi(Ad,Bd,R,Ro)
    # print('Psi shape:{}'.format(Psi.shape))
    # print('Ro shape: ',Ro.shape)
    # Computing Ro_c
    Ro_c=np.matmul(Psi,np.matmul(Ro,Dd))
    
    for h in range(2,lkd+2):
        Ro_c=np.hstack((Ro_c,np.matmul(np.linalg.matrix_power(Psi,h),np.matmul(Ro,Dd))))

    # Computing the disturbance matrix (road curvature)
    if (index+lkd+1)<len(traj):
        Cr=curv[index:index+lkd+1]
    else:
        Cr=curv[index:-1]
        Cr=np.append(Cr,np.zeros(lkd+1-len(Cr)))

    # Computing Kb and Kf
    pre=np.linalg.inv(R+np.matmul(Bd.T,np.matmul(Ro,Bd)))
    Kb=np.matmul(pre,np.matmul(Bd.T,np.matmul(Ro,Ad)))
    Kf=np.matmul(pre,np.matmul(Bd.T,np.matmul(Ro,D_)+np.matmul(Ro_c,lam)))
    
    #  The error state vector:
    phides = np.arctan2((traj[index][1]-Y),(traj[index][0]-X))
    phidesdot = xdot*curv[index]

    e = np.zeros(4)
    
    # #Ref p34 Vehicle Dynamics and Control by Rajesh Rajamani
    e[0] = (Y - traj[index][1])*np.cos(phides) - (X - traj[index][0])*np.sin(phides)
    # e[2] = wrap2pi(phi - phides)
    # e[1] = ydot + xdot*e[2]
    # e[3] = phidot - phidesdot
    # e[0]=(Y-traj[index][1])*np.cos(phides)
    e[2]=wrap2pi(phi-phides)
    e[1]=ydot+xdot*e[2]
    e[3]=phidot-phidesdot
    error = np.matrix(e)

    # deltades = float(-K*np.transpose(error))
    #deltades = float(-K[0,:]*np.transpose(error))
    deltades=float(-np.matmul(Kb,error.T)-np.matmul(Kf,Cr))
    print(-np.matmul(Kb,error.T),-np.matmul(Kf,Cr),deltades)
    deltad = (deltades - delta)/0.05
    
    #Bang-bang control for F
    if xdot > Vx:
        F= -8000
    else:
        F= 2500

    controlinp = vehicle.command(F,deltad)

    return controlinp



