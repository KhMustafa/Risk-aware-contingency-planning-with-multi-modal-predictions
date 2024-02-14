import pdb
import osqp
import argparse
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import numpy as np
from scipy.io import loadmat
from scipy import interpolate, sparse
import random
import math
from numpy import linalg as LA
from numpy.linalg import norm
from highway_branch_dyn import *
import pickle

v0 = 15
f0 = np.array([v0, 0, 0, 0])
lane_width = 3.6
lm = np.arange(0, 7) * lane_width


def with_probability(P=1):
    return np.random.uniform() <= P


class vehicle():
    def __init__(self, state=[0, 0, v0, 0], v_length=4, v_width=2.4, dt=0.05, backupidx=0, laneidx=0):
        self.state = np.array(state)
        self.dt = dt
        self.v_length = v_length
        self.v_width = v_width
        self.x_pred = []
        self.y_pred = []
        self.xbackup = None
        self.backupidx = backupidx
        self.laneidx = laneidx

    def step(self, u):  # controlled vehicle
        dxdt = np.array([self.state[2] * np.cos(self.state[3]), self.state[2] * np.sin(self.state[3]), u[0], u[1]])
        self.state = self.state + dxdt * self.dt


class Highway_env():
    def __init__(self, NV, mpc, N_lane=6, timestep=0, ego_state=[], obst_new_state=[]):
        '''
        Input: NV: number of vehicles
               mpc: mpc controller for the controlled vehicle
               N_lane: number of lanes
        '''
        self.dt = mpc.predictiveModel.dt
        self.veh_set = []
        self.NV = NV
        self.N_lane = N_lane
        self.desired_x = [None] * NV
        self.mpc = mpc
        self.predictiveModel = mpc.predictiveModel
        self.backupcons = mpc.predictiveModel.backupcons

        self.m = len(self.backupcons)
        self.cons = mpc.predictiveModel.cons
        self.LB = [self.cons.W / 2, N_lane * 3.6 - self.cons.W / 2]

        # x0 = np.array([[0,1.8,v0,0],[5,5.4,v0,0]])
        if timestep == 0:
            x0 = np.array([[-8, 1.8, v0, 0], [-0, 5.4, v0, 0]])
        else:
            x0 = np.array([[ego_state.position[0], ego_state.position[1], ego_state.velocity,
                            ego_state.orientation], obst_new_state])
        for i in range(0, self.NV):
            self.veh_set.append(vehicle(x0[i], dt=self.dt, backupidx=0))
            self.desired_x[i] = np.array([0, x0[i, 1], v0, 0])

    def step(self, t_):
        # initialize the trajectories to be propagated forward under the backup policy
        u_set = [None] * self.NV
        xx_set = [None] * self.NV
        u0_set = [None] * self.NV
        x_set = [None] * self.NV

        umax = np.array([self.cons.am, self.cons.rm])
        # generate backup trajectories
        self.xbackup = np.empty([0, (self.mpc.N + 1) * 4])
        for i in range(0, self.NV):
            z = self.veh_set[i].state
            xx_set[i] = self.predictiveModel.zpred_eval(z)
            newlaneidx = round((z[1] - 1.8) / 3.6)

            if t_ == 0 or (newlaneidx != self.veh_set[i].laneidx and abs(z[1] + 1.8 + 3.6 * newlaneidx) < 1.4):
                # update the desired lane
                self.veh_set[i].laneidx = newlaneidx
                self.desired_x[i][1] = 1.8 + newlaneidx * 3.6
                if i == 1:
                    if self.veh_set[0].laneidx < self.veh_set[1].laneidx:
                        xRef = np.array([0, 1.8 + 3.6 * (self.veh_set[1].laneidx - 1), v0, 0])
                    elif self.veh_set[0].laneidx > self.veh_set[1].laneidx:
                        xRef = np.array([0, 1.8 + 3.6 * (self.veh_set[1].laneidx + 1), v0, 0])
                    else:
                        if self.veh_set[1].laneidx > 0:
                            xRef = np.array([0, 1.8 + 3.6 * (self.veh_set[1].laneidx - 1), v0, 0])
                        else:
                            xRef = np.array([0, 1.8 + 3.6 * (self.veh_set[1].laneidx + 1), v0, 0])
                    backupcons = [lambda x: backup_maintain(x, self.cons), lambda x: backup_brake(x, self.cons),
                                  lambda x: backup_lc(x, xRef)]
                    self.predictiveModel.update_backup(backupcons)

        idx0 = self.veh_set[0].backupidx
        n = self.predictiveModel.n
        x1 = xx_set[0][:, idx0 * n:(idx0 + 1) * n]
        for i in range(0, self.NV):
            if i != 0:
                hi = np.zeros(self.m)
                for j in range(0, self.m):
                    hi[j] = min(
                        np.append(veh_col(x1, xx_set[i][:, j * n:(j + 1) * n], [self.cons.L + 1, self.cons.W + 0.2]),
                                  lane_bdry_h(x1, self.LB[0], self.LB[1])))
                self.veh_set[i].backupidx = np.argmax(hi)

            u0_set[i] = self.backupcons[self.veh_set[i].backupidx](self.veh_set[i].state)

        # set x_ref for the overtaking maneuver and call the MPC

        if self.veh_set[0].state[0] < self.veh_set[1].state[0]:
            Ydes = 1.8 + self.veh_set[0].laneidx * 3.6
        else:
            Ydes = self.veh_set[1].state[1]
        if abs(self.veh_set[0].state[1] - Ydes) < 1 and self.veh_set[0].state[0] > self.veh_set[1].state[0] + 3:
            vdes = v0
        else:
            vdes = self.veh_set[1].state[2] + 1 * (self.veh_set[1].state[0] + 1.5 - self.veh_set[0].state[0])

        xRef = np.array([0, Ydes, vdes, 0])
        self.mpc.solve(self.veh_set[0].state, self.veh_set[1].state, xRef)

        xPred, zPred, utraj, branch_w = self.mpc.BT2array()

        for i in range(1, self.NV):
            u_set[i] = u0_set[i]
            self.veh_set[i].step(u_set[i])
            x_set[i] = self.veh_set[i].state

        print('obstacle vehicle velocity: ', x_set[1][2])
        return u_set, x_set, xx_set, xPred, zPred, branch_w


def Highway_sim(env, T):
    # simulate the scenario
    collision = False
    dt = env.dt
    t = 0
    Ts_update = 4
    N = int(round(T / dt))
    state_rec = np.zeros([env.NV, N, 4])
    backup_rec = [None] * env.NV
    backup_choice_rec = [None] * env.NV

    for i in range(0, env.NV):
        backup_rec[i] = [None] * N
        backup_choice_rec[i] = [None] * N

    for i in range(0, len(env.veh_set)):
        state_rec[i][t] = env.veh_set[i].state

    dis = 100

    if not collision:
        for i in range(0, env.NV):
            for j in range(0, env.NV):
                if i != j:
                    dis = max(abs(env.veh_set[i].state[0] - env.veh_set[j].state[0]) - 0.5 * (
                            env.veh_set[i].v_length + env.veh_set[j].v_length),
                              abs(env.veh_set[i].state[1] - env.veh_set[j].state[1]) - 0.5 * (
                                      env.veh_set[i].v_width + env.veh_set[j].v_width))
            if dis < 0:
                collision = True

    print("t=", t * env.dt)

    u_set, x_set, xx_set, xPred, zPred, branch_w = env.step(t)

    # x_set[1] includes the updated state of the obstacle vehicle
    return xx_set, zPred, x_set[1], state_rec, branch_w


def sim_overtake(mpc, N_lane, timestep, ego_state, obst_new_state, trajectory):
    env = Highway_env(NV=2, mpc=mpc, N_lane=N_lane, timestep=timestep, ego_state=ego_state,
                      obst_new_state=obst_new_state)

    backup, zPred, obst_new_state, state_rec, branch_w = Highway_sim(env, 0.1)

    return backup, zPred, obst_new_state, branch_w, state_rec
