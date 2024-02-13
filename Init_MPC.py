import numpy as np
import pdb
from scipy import linalg
from PredictiveControllers import MPC, MPCParams
from MPC_branch import BranchMPC, BranchMPCParams

def initMPCParams(nx, d, N, M, m, ydes,vdes,am,rm,N_lane,W):
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[0., 1., 0., 0.],
                   [0., -1., 0., 0.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., -1.]])
    Fx = np.hstack((Fx,np.zeros([Fx.shape[0],m*M])))

    bx = np.array([[N_lane*3.6-W/2],   # max y
                   [-W/2],             # min y
                   [0.25],             # max psi
                   [0.25]]),           # min psi

    Fu = np.kron(np.eye(2), np.array([1, -1])).T
    bu = np.array([[am],   # -Min Acceleration
                   [0.5*am],   # Max Acceleration
                   [rm],  # -Min Steering
                   [rm]]) # Max Steering

    # Tuning Parameters
    Qx = np.diag([0., 0.5, 0.2, 5.]) # vx, vy, wz, epsi, s, ey
    Q = linalg.block_diag(Qx,np.zeros([M*m,M*m]))
    R = np.diag([30, 100.0])                  # delta, a
    xRef   = np.append(np.array([0,ydes,vdes,0]),np.zeros(M*m))
    Qslack = 1 * np.array([0, 1000])

    mpcParameters    = MPCParams(n=nx+M*m, d=d, N=N, Q=Q, R=R, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True, Qslack=Qslack, timeVarying = True)
    return mpcParameters
def initBranchMPC(n,d,N,NB,xRef,am,rm,N_lane,W):
    Fx = np.array([[0., 1., 0., 0.],
                   [0., -1., 0., 0.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., -1.]])

    bx = np.array([[N_lane*3.6-W/2],   # max y
                   [-W/2],             # min y
                   [0.25],             # max psi
                   [0.25]]),           # min psi

    Fu = np.kron(np.eye(2), np.array([1, -1])).T
    bu = np.array([[am],   # -Min Acceleration
                   [am],   # Max Acceleration
                   [rm],  # -Min Steering
                   [rm]]) # Max Steering

    # Tuning Parameters
    Q = np.diag([0., 3, 3, 10.]) # vx, vy, wz, epsi, s, ey
    R = np.diag([1, 100.0])                  # delta, a

    Qslack = 1 * np.array([0, 300])

    mpcParameters    = BranchMPCParams(n=n, d=d, N=N, NB = NB, Q=Q, R=R, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True, Qslack=Qslack, timeVarying = True)
    return mpcParameters
def initquadBranchMPC(n,d,N,NB,xRef,vxm,vym,rm):
    Fx = np.empty([0,n])

    bx = np.empty([0,1]),           # min psi

    Fu = np.kron(np.eye(3), np.array([1, -1])).T
    bu = np.array([[vxm],   # -Min Acceleration
                   [0],   # Max Acceleration
                   [vym],  # -Min Steering
                   [vym],
                   [rm],
                   [rm]]) # Max Steering

    # Tuning Parameters
    Q = np.diag([1., 1., 1]) # vx, vy, wz, epsi, s, ey
    R = np.diag([1., 100., 1.])                  # delta, a
    dR = np.array([0.9,5,1])
    Qslack = 1 * np.array([0, 300])

    mpcParameters    = BranchMPCParams(n=n, d=d, N=N, NB = NB, Q=Q, R=R,dR = dR, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True, Qslack=Qslack, timeVarying = True)
    return mpcParameters
