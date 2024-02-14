import numpy as np
import pdb
from scipy import linalg
from PredictiveControllers import MPC, MPCParams
from MPC_branch import BranchMPC, BranchMPCParams


def initBranchMPC(n, d, N, NB, xRef, am, rm, N_lane, W):
    Fx = np.array([[0., 1., 0., 0.],
                   [0., -1., 0., 0.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., -1.]])

    bx = np.array([[N_lane * 3.6 - W / 2],  # max y
                   [-W / 2],  # min y
                   [0.25],  # max psi
                   [0.25]]),  # min psi

    Fu = np.kron(np.eye(2), np.array([1, -1])).T
    bu = np.array([[am],  # -Min Acceleration
                   [am],  # Max Acceleration
                   [rm],  # -Min Steering
                   [rm]])  # Max Steering

    # Tuning Parameters
    Q = np.diag([0., 3, 3, 10.])  # vx, vy, wz, epsi, s, ey
    R = np.diag([1, 100.0])  # delta, a

    Qslack = 1 * np.array([0, 300])

    mpcParameters = BranchMPCParams(n=n, d=d, N=N, NB=NB, Q=Q, R=R, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True,
                                    Qslack=Qslack, timeVarying=True)
    return mpcParameters
