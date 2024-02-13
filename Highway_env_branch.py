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

        # Ydes = 1.8+self.veh_set[0].laneidx*3.6
        # vdes = self.veh_set[1].state[2]+5
        xRef = np.array([0, Ydes, vdes, 0])
        self.mpc.solve(self.veh_set[0].state, self.veh_set[1].state, xRef)

        xPred, zPred, utraj, branch_w = self.mpc.BT2array()
        # ToDo: should we update the state of the ego_vehicle here?
        '''
        self.veh_set[0].step(u_set[0])
        x_set[0] = self.veh_set[0].state
        '''

        for i in range(1, self.NV):
            u_set[i] = u0_set[i]
            self.veh_set[i].step(u_set[i])
            x_set[i] = self.veh_set[i].state

        print('obstacle vehicle velocity: ', x_set[1][2])
        return u_set, x_set, xx_set, xPred, zPred, branch_w

    def replace_veh(self, idx, dir=2):
        if idx == 0:
            return
        if dir == 0:
            UB = self.veh_set[0].state[0] + 13
            LB = self.veh_set[0].state[0] + 8
        elif dir == 1:
            UB = self.veh_set[0].state[0] - 5
            LB = self.veh_set[0].state[0] - 13
        else:
            UB = self.veh_set[0].state[0] + 15
            LB = self.veh_set[0].state[0] - 15

        if self.veh_set[0].laneidx == 0:
            laneidx = 1
        elif self.veh_set[0].laneidx == self.N_lane - 1:
            laneidx = self.N_lane - 2
        else:
            if with_probability(0.5):
                laneidx = self.veh_set[0].laneidx - 1
            else:
                laneidx = self.veh_set[0].laneidx + 1
        success = False
        count = 0
        while not success:
            count += 1
            Y = (laneidx + 0.5) * lane_width + np.random.normal(0, 0.1)
            X = random.random() * (UB - LB) + LB
            collision = False
            for i in range(0, self.NV):
                if i != idx:
                    if abs(Y - self.veh_set[i].state[1]) <= 2.2 and abs(X - self.veh_set[i].state[0]) <= 5:
                        collision = True
                        break
            if not collision:
                success = True
            if count > 20:
                return False
        self.veh_set[idx] = vehicle([X, Y, self.veh_set[0].state[2], 0], dt=self.dt, backupidx=0, laneidx=laneidx)
        return True


def merge_geometry(N_lane, merge_lane, merge_s, merge_R, merge_side=0):
    '''
    generate the merging geometry
    input: N_lane: number of lanes on the main highway
           merge_lane: number of lanes on the ramp
           merge_s: X coordinate of the merging position
           merge_R: radius of the arc for the ramp
           merge_side: left or right
    '''
    merge_theta = np.arccos(1 - lane_width * merge_lane / merge_R)

    merge_end = merge_s + merge_R * np.sin(merge_theta)
    if merge_side == 0:
        merge_arc_center = np.array(
            [merge_s + merge_R * np.sin(merge_theta), (N_lane - merge_lane) * lane_width + merge_R])
        merge_lane_start = np.array(
            [merge_s - merge_s * np.cos(merge_theta), N_lane * lane_width + np.sin(merge_theta) * merge_s])
    else:
        merge_arc_center = np.array([merge_s + merge_R * np.sin(merge_theta), merge_lane * lane_width - merge_R])
        merge_lane_start = np.array(
            [merge_s - merge_s * np.cos(merge_theta), -np.sin(merge_theta) * merge_s - lane_width * merge_lane])

    merge_lane_ref_s1 = np.linspace(0, merge_s, num=int(merge_s / 0.5), endpoint=False)  # straight portion
    merge_lane_ref_s2 = merge_s + np.linspace(0, merge_R * merge_theta,
                                              num=int(merge_R * merge_theta / 0.5))  # arc portion
    if merge_side == 0:
        merge_lane_ref_X1 = merge_lane_start[0] + merge_lane_ref_s1 * np.cos(merge_theta)
        merge_lane_ref_Y1 = merge_lane_start[1] - merge_lane_ref_s1 * np.sin(merge_theta)
        merge_lane_ref_psi1 = -np.ones(merge_lane_ref_s1.shape) * merge_theta
        merge_lane_ref_psi2 = (merge_lane_ref_s2 - merge_lane_ref_s2[-1]) / merge_R
        merge_lane_ref_X2 = merge_arc_center[0] + np.sin(merge_lane_ref_psi2) * merge_R
        merge_lane_ref_Y2 = merge_arc_center[1] - np.cos(merge_lane_ref_psi2) * merge_R
    else:
        merge_lane_ref_X1 = merge_lane_start[0] + merge_lane_ref_s1 * np.cos(merge_theta)
        merge_lane_ref_Y1 = merge_lane_start[1] + merge_lane_ref_s1 * np.sin(merge_theta)
        merge_lane_ref_psi1 = np.ones(merge_lane_ref_s1.shape) * merge_theta
        merge_lane_ref_psi2 = (merge_lane_ref_s2[-1] - merge_lane_ref_s2) / merge_R
        merge_lane_ref_X2 = merge_arc_center[0] - np.sin(merge_lane_ref_psi2) * merge_R
        merge_lane_ref_Y2 = merge_arc_center[1] + np.cos(merge_lane_ref_psi2) * merge_R - merge_lane * lane_width

    return merge_lane_ref_X1, merge_lane_ref_X2, merge_lane_ref_Y1, merge_lane_ref_Y2, merge_lane_ref_psi1, merge_lane_ref_psi2


class Highway_env_merge():
    '''
    Similar object, for merging simulation
    '''

    def __init__(self, NV, N_lane, mpc, pred_model, merge_lane=2, merge_s=50, merge_R=300, merge_side=0, dt=0.05):
        self.dt = dt
        self.veh_set = []
        self.NV = NV
        self.laneID = [1] + [0] * (NV - 1)
        self.N_lane = N_lane
        self.merge_lane = merge_lane
        self.desired_x = [None] * NV
        self.merge_s = merge_s
        self.merge_R = merge_R
        self.merge_side = merge_side
        self.pred_model = pred_model
        self.mpc = mpc
        self.m = [None] * len(pred_model)
        self.backupcons = [None] * len(pred_model)
        for i in range(0, len(pred_model)):
            self.backupcons[i] = self.pred_model[i].backupcons
            self.m[i] = len(self.backupcons[i])
        self.cons = mpc.predictiveModel.cons
        self.LB = [self.cons.W / 2, N_lane * 3.6 - self.cons.W / 2]

        merge_lane_ref_X1, merge_lane_ref_X2, merge_lane_ref_Y1, merge_lane_ref_Y2, merge_lane_ref_psi1, merge_lane_ref_psi2 = merge_geometry(
            N_lane, merge_lane, merge_s, merge_R, merge_side)
        self.merge_theta = np.arccos(1 - lane_width * merge_lane / merge_R)
        self.merge_end = merge_s + merge_R * np.sin(self.merge_theta)
        self.merge_lane_ref_Y = np.append(merge_lane_ref_Y1, merge_lane_ref_Y2)
        self.merge_lane_ref_X = np.append(merge_lane_ref_X1, merge_lane_ref_X2)
        self.merge_lane_ref_psi = np.append(merge_lane_ref_psi1, merge_lane_ref_psi2)
        self.merge_lane_ref_Y1 = merge_lane_ref_Y1
        self.merge_lane_ref_Y2 = merge_lane_ref_Y2
        self.merge_lane_ref_X1 = merge_lane_ref_X1
        self.merge_lane_ref_X2 = merge_lane_ref_X2
        self.merge_lane_ref_psi1 = merge_lane_ref_psi1
        self.merge_lane_ref_psi2 = merge_lane_ref_psi2
        self.refY = interpolant('refY', 'linear', [self.merge_lane_ref_X], self.merge_lane_ref_Y)
        self.refpsi = interpolant('refY', 'linear', [self.merge_lane_ref_X], self.merge_lane_ref_psi)
        UB = 30
        LB = 0
        x0 = np.array([[24, 13, v0, -0.2], [15, 5.4, v0, 0]])
        for i in range(0, self.NV):
            self.veh_set.append(vehicle(x0[i], dt=self.dt, backupidx=0))
            self.desired_x[i] = np.array([0, x0[i, 1], v0, 0])

    def step(self, t_):
        u_set = [None] * self.NV
        xx_set = [None] * self.NV
        u0_set = [None] * self.NV
        x_set = [None] * self.NV

        umax = np.array([self.cons.am, self.cons.rm])
        # generate backup trajectories
        self.xbackup = np.empty([0, (self.mpc.N + 1) * 4])
        for i in range(0, self.NV):
            z = self.veh_set[i].state
            if self.veh_set[i].state[0] > self.merge_s + 8:
                self.laneID[i] = 0
            xx_set[i] = self.pred_model[self.laneID[i]].zpred_eval(z)

        idx0 = self.veh_set[0].backupidx
        n = self.pred_model[self.laneID[0]].n
        x1 = xx_set[0][:, idx0 * n:(idx0 + 1) * n]
        for i in range(0, self.NV):
            if i != 0:
                hi = np.zeros(self.m[self.laneID[i]])
                if self.laneID[i] == 0:
                    for j in range(0, self.m[0]):
                        hi[j] = min(np.append(
                            veh_col(x1, xx_set[i][:, j * n:(j + 1) * n], [self.cons.L + 1, self.cons.W + 0.2]),
                            lane_bdry_h(xx_set[i][:, j * n:(j + 1) * n], self.LB[0], self.LB[1])))
                elif self.laneID[i] == 1:
                    for j in range(0, self.m[1]):
                        hi[j] = veh_col(x1, xx_set[i][:, j * n:(j + 1) * n], [self.cons.L + 1, self.cons.W + 0.2])
                self.veh_set[i].backupidx = np.argmax(hi)
            self.veh_set[i].backupidx = 0
            u0_set[i] = self.backupcons[self.laneID[i]][self.veh_set[i].backupidx](self.veh_set[i].state)

        x = self.veh_set[0].state
        if self.laneID[0] == 0:
            S = np.eye(4)
            xRef = np.array([0, (self.N_lane - 0.5) * 3.6, v0, 0])
            bx = self.mpc.param.bx

        else:
            y0 = float(self.refY(x[0]))
            psi0 = float(self.refpsi(x[0]))
            S = np.array([[1., 0, 0, 0], [-np.tan(psi0), 1., 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            xRef = np.array([0, -np.tan(psi0) * x[0] + y0 + 1.8, v0, psi0])
            bx = np.array([-np.tan(psi0) * x[0] + y0 + 3.6 * self.merge_lane - self.cons.W / 2,
                           np.tan(psi0) * x[0] - y0 - self.cons.W / 2, psi0 + self.mpc.psimax, -psi0 + self.mpc.psimax])

        self.mpc.solve(self.veh_set[0].state, self.veh_set[1].state, xRef, S, Fx=None, bx=bx)

        u_set[0] = self.mpc.uPred[0]
        xPred, zPred, uPred, branch_w = self.mpc.BT2array()
        self.veh_set[0].step(u_set[0])
        x_set[0] = self.veh_set[0].state

        for i in range(1, self.NV):
            u_set[i] = u0_set[i]
            self.veh_set[i].step(u_set[i])
            x_set[i] = self.veh_set[i].state

        return u_set, x_set, xx_set, xPred, zPred, branch_w


def Highway_sim(env, T):
    # simulate the scenario
    collision = False
    dt = env.dt
    t = 0
    Ts_update = 4
    N_update = int(round(Ts_update / dt))
    N = int(round(T / dt))
    state_rec = np.zeros([env.NV, N, 4])
    b_rec = [None] * N
    backup_rec = [None] * env.NV
    backup_choice_rec = [None] * env.NV
    xPred_rec = [None] * N
    zPred_rec = [None] * N
    branch_w_rec = [None] * N
    for i in range(0, env.NV):
        backup_rec[i] = [None] * N
        backup_choice_rec[i] = [None] * N
    input_rec = np.zeros([env.NV, N, 2])
    f0 = np.array([v0, 0, 0, 0])
    for i in range(0, len(env.veh_set)):
        state_rec[i][t] = env.veh_set[i].state

    xx_set = []
    dis = 100
    # while t<N:
    if not collision:
        for i in range(0, env.NV):
            for j in range(0, env.NV):
                if i != j:
                    dis = max(abs(env.veh_set[i].state[0] - env.veh_set[j].state[0]) - 0.5 * (
                            env.veh_set[i].v_length + env.veh_set[j].v_length), \
                              abs(env.veh_set[i].state[1] - env.veh_set[j].state[1]) - 0.5 * (
                                      env.veh_set[i].v_width + env.veh_set[j].v_width))
            if dis < 0:
                collision = True

    print("t=", t * env.dt)

    u_set, x_set, xx_set, xPred, zPred, branch_w = env.step(t)
    '''
    xPred_rec[t] = xPred
    zPred_rec[t] = zPred
    branch_w_rec[t] = branch_w
    for i in range(0, env.NV):
        input_rec[i][t] = u_set[i]
        state_rec[i][t] = x_set[i]
        backup_rec[i][t] = xx_set[i]
        backup_choice_rec[i][t] = env.veh_set[i].backupidx

    return state_rec, input_rec, backup_rec, backup_choice_rec, xPred_rec, zPred_rec, branch_w_rec, collision
    '''
    # x_set[1] includes the updated state of the obstacle vehicle
    return xx_set, zPred, x_set[1], state_rec, branch_w


def plot_snapshot(x, z, env, idx=None, varycolor=True, zpatch=True, arrow=True, legend=True):
    '''
    Ploting a snapshot of the simulation, for debugging
    '''
    plot_merge = isinstance(env, Highway_env_merge)
    if plot_merge:
        fig = plt.figure(figsize=(15, 6))
    else:
        fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ego_idx = 0
    ego_veh = env.veh_set[ego_idx]
    veh_patch = [None] * env.NV
    for i in range(0, env.NV):
        if i == ego_idx:
            veh_patch[i] = plt.Rectangle(
                (ego_veh.state[0] - ego_veh.v_length / 2, ego_veh.state[1] - ego_veh.v_width / 2), ego_veh.v_length,
                ego_veh.v_width, fc='r', zorder=0)
        else:
            veh_patch[i] = plt.Rectangle(
                (ego_veh.state[0] - ego_veh.v_length / 2, ego_veh.state[1] - ego_veh.v_width / 2), ego_veh.v_length,
                ego_veh.v_width, fc='b', zorder=0)

    ego_y = ego_veh.state[1]
    ego_x = ego_veh.state[0]
    if plot_merge:
        xmin = ego_x - 10
        xmax = ego_x + 40
        ymin = -5
        ymax = 15
    else:
        xmin = ego_x - 10
        xmax = ego_x + 40
        ymin = ego_y - 5
        ymax = ego_y + 10
    try:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-ymax, -ymin)
    except:
        pdb.set_trace()
    ts = ax.transData
    for i in range(0, env.NV):
        coords = ts.transform([env.veh_set[i].state[0], -env.veh_set[i].state[1]])
        tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -env.veh_set[i].state[3])
        te = ts + tr
        veh_patch[i].set_xy([env.veh_set[i].state[0] - env.veh_set[i].v_length / 2,
                             -env.veh_set[i].state[1] - env.veh_set[i].v_width / 2])
        veh_patch[i].set_transform(te)
        ax.add_patch(veh_patch[i])

    colorset = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                'tab:olive', 'tab:cyan', 'y', 'm', 'c', 'g']
    xPred, zPred, uPred, branch_w = env.mpc.BT2array()
    if idx is None:
        idx = range(0, len(zPred))

    for j in idx:
        for k in range(0, xPred[j].shape[0]):
            if k % 2 == 1:
                if varycolor:
                    x_patch = plt.Rectangle((xPred[j][k, 0] - env.veh_set[ego_idx].v_length / 2,
                                             -xPred[j][k, 1] - env.veh_set[ego_idx].v_width / 2),
                                            env.veh_set[ego_idx].v_length, env.veh_set[ego_idx].v_width, ec=colorset[j],
                                            fc=colorset[j], alpha=0.2, zorder=0)
                else:
                    x_patch = plt.Rectangle((xPred[j][k, 0] - env.veh_set[ego_idx].v_length / 2,
                                             -xPred[j][k, 1] - env.veh_set[ego_idx].v_width / 2),
                                            env.veh_set[ego_idx].v_length, env.veh_set[ego_idx].v_width, ec='y', fc='y',
                                            alpha=0.2, zorder=0)
                coords = ts.transform([xPred[j][k, 0], -xPred[j][k, 1]])
                if arrow:
                    arr = ax.arrow(xPred[j][k, 0], -xPred[j][k, 1], uPred[j][k, 0] * np.cos(xPred[j][k, 3]),
                                   -uPred[j][k, 0] * np.sin(xPred[j][k, 3]), head_width=0.5, length_includes_head=True)
                else:
                    arr = None
                tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred[j][k, 3])
                x_patch.set_transform(ts + tr)
                ax.add_patch(x_patch)

    for j in idx:
        z_patch = plt.plot(zPred[j][:, 0], -zPred[j][:, 1], 'm--', linewidth=2)[0]
        if zpatch:
            for k in range(0, zPred[j].shape[0]):
                if k % 2 == 1:
                    if varycolor:
                        z_patch = plt.Rectangle((zPred[j][k, 0] - env.veh_set[1].v_length / 2,
                                                 -zPred[j][k, 1] - env.veh_set[1].v_width / 2), env.veh_set[1].v_length,
                                                env.veh_set[1].v_width, ec=colorset[-1 - j], fc=colorset[-1 - j],
                                                alpha=0.2, zorder=0)
                    else:
                        z_patch = plt.Rectangle((zPred[j][k, 0] - env.veh_set[1].v_length / 2,
                                                 -zPred[j][k, 1] - env.veh_set[1].v_width / 2), env.veh_set[1].v_length,
                                                env.veh_set[1].v_width, ec='c', fc='c', alpha=0.2, zorder=0)
                    coords = ts.transform([zPred[j][k, 0], -zPred[j][k, 1]])
                    tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -zPred[j][k, 3])
                    z_patch.set_transform(ts + tr)
                    ax.add_patch(z_patch)
    if legend:
        ax.legend([x_patch, z_patch, arr],
                  ['Planned trajectory for ego vehicle', 'Predicted trajectory for the uncontrolled vehicle',
                   'Ego vehicle acceleration'], fontsize=15)
    if plot_merge:
        if env.merge_side == 0:
            plt.plot([-10, 1000], [-lm[0], -lm[0]], 'g', linewidth=2)
            for j in range(1, env.N_lane):
                plt.plot([-10, 1000], [-lm[j], -lm[j]], 'g--', linewidth=1)

            plt.plot([-10, env.merge_s], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)
            plt.plot([env.merge_end, 1000], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)

            plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1, 'g', linewidth=2)
            plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2, 'g--', linewidth=1)
            for j in range(1, env.merge_lane):
                plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1 - j * lane_width, 'g--', linewidth=1)
                plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2 - j * lane_width, 'g--', linewidth=1)
            plt.plot(env.merge_lane_ref_X, -env.merge_lane_ref_Y - env.merge_lane * lane_width, 'g', linewidth=2)
        else:
            plt.plot([-10, 1000], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)
            for j in range(1, env.N_lane):
                plt.plot([-10, 1000], [-lm[j], -lm[j]], 'g--', linewidth=1)

            plt.plot([-10, env.merge_s], [-lm[0], -lm[0]], 'g', linewidth=2)
            plt.plot([env.merge_end, 1000], [-lm[0], -lm[0]], 'g', linewidth=2)

            plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1, 'g', linewidth=2)
            plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2, 'g', linewidth=2)
            for j in range(1, env.merge_lane):
                plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1 - j * lane_width, 'g--', linewidth=1)
                plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2 - j * lane_width, 'g--', linewidth=1)
            plt.plot(env.merge_lane_ref_X, -env.merge_lane_ref_Y - env.merge_lane * lane_width, 'g', linewidth=2)
    else:
        plt.plot([xmin - 50, xmax + 50], [-lm[0], -lm[0]], 'g', linewidth=2)
        for j in range(1, env.N_lane):
            plt.plot([xmin - 50, xmax + 50], [-lm[j], -lm[j]], 'g--', linewidth=1)
        plt.plot([xmin - 50, xmax + 50], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)
    plt.show()


def plot_scenario(mpc, N_lane, timestep, ego_state, obst_new_state, trajectory, state_rec, zPred):
    env = Highway_env(NV=2, mpc=mpc, N_lane=N_lane, timestep=timestep, ego_state=ego_state,
                      obst_new_state=obst_new_state)
    animate_scenario(env, state_rec, trajectory, zPred, 'animation.mp4')


def animate_scenario(env, state_rec, xPred_rec, zPred_rec, output=None):
    '''
    Animate the simulation
    '''
    if output:
        matplotlib.use("Agg")
    ego_idx = 0
    plot_merge = isinstance(env, Highway_env_merge)
    if plot_merge:
        fig = plt.figure(figsize=(10, 8))
    else:
        if not plt.fignum_exists(1):
            fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(111)


    # plt.clf()

    # ax = plt.subplot()

    plt.grid()

    nframe = len(state_rec)
    ego_veh = env.veh_set[ego_idx]
    veh_patch = [None] * env.NV
    for i in range(0, env.NV):
        if i == ego_idx:
            veh_patch[i] = plt.Rectangle(
                (ego_veh.state[0] - ego_veh.v_length / 2, ego_veh.state[1] - ego_veh.v_width / 2), ego_veh.v_length,
                ego_veh.v_width, fc='r', zorder=0)

        else:
            veh_patch[i] = plt.Rectangle(
                (ego_veh.state[0] - ego_veh.v_length / 2, ego_veh.state[1] - ego_veh.v_width / 2), ego_veh.v_length,
                ego_veh.v_width, fc='b', zorder=0)

    for patch in veh_patch:
        ax.add_patch(patch)

    def animate(t, veh_patch, state_rec, xPred_rec, zPred_rec, env, ego_idx=0):
        plot_merge = isinstance(env, Highway_env_merge)
        N_veh = len(state_rec)
        ego_y = state_rec[ego_idx][t][1]
        ego_x = state_rec[ego_idx][t][0]
        ax.clear()
        if plot_merge:
            xmin = ego_x - 5
            xmax = ego_x + 45
            ymin = -5
            ymax = 35
        else:
            xmin = ego_x - 10
            xmax = ego_x + 40
            ymin = ego_y - 10
            ymax = ego_y + 10
        try:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-ymax, -ymin)
        except:
            pdb.set_trace()
        ts = ax.transData
        for i in range(0, N_veh):
            coords = ts.transform([state_rec[i][t][0], -state_rec[i][t][1]])
            tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -state_rec[i][t][3])
            te = ts + tr
            veh_patch[i].set_xy(
                [state_rec[i][t][0] - env.veh_set[i].v_length / 2, -state_rec[i][t][1] - env.veh_set[i].v_width / 2])
            veh_patch[i].set_transform(te)
            ax.add_patch(veh_patch[i])

        colorset = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                    'tab:olive', 'tab:cyan', 'y', 'm', 'c', 'g']
        # for j in range(0, len(xPred_rec[t])):
        plt.plot(xPred_rec['shared_plan'].x, -xPred_rec['shared_plan'].y, 'b--', linewidth=2)
        for k in range(0, xPred_rec['shared_plan'].x.shape[0]):
            if k % 2 == 1:
                newpatch = plt.Rectangle((xPred_rec['shared_plan'].x[k] - env.veh_set[ego_idx].v_length / 2,
                                          -xPred_rec['shared_plan'].y[k] - env.veh_set[ego_idx].v_width / 2),
                                         env.veh_set[ego_idx].v_length, env.veh_set[ego_idx].v_width,
                                         fc=colorset[0], alpha=0.3, zorder=0)
                coords = ts.transform([xPred_rec['shared_plan'].x[k], -xPred_rec['shared_plan'].y[k]])
                tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1],
                                                                    -xPred_rec['shared_plan'].yaw[k])
                newpatch.set_transform(ts + tr)
                ax.add_patch(newpatch)

        keys_list = list(xPred_rec.keys())

        for key in keys_list[1:-1]:
            plt.plot(xPred_rec[key].x, -xPred_rec[key].y, 'b--', linewidth=1)
            for k in range(0, xPred_rec[key].x.shape[0]):
                if k % 2 == 1:
                    newpatch = plt.Rectangle((xPred_rec[key].x[k] - env.veh_set[ego_idx].v_length / 2,
                                              -xPred_rec[key].y[k] - env.veh_set[ego_idx].v_width / 2),
                                             env.veh_set[ego_idx].v_length, env.veh_set[ego_idx].v_width,
                                             fc=colorset[1], alpha=0.3, zorder=0)
                    coords = ts.transform([xPred_rec[key].x[k], -xPred_rec[key].y[k]])
                    tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred_rec[key].yaw[k])
                    newpatch.set_transform(ts + tr)
                    ax.add_patch(newpatch)

        '''
        plt.plot(xPred_rec[0].x, -xPred_rec[0].y, 'g--', linewidth=1)
        for k in range(0, xPred_rec[0].x.shape[0]):
            if k % 2 == 1:
                newpatch = plt.Rectangle((xPred_rec[0].x[k] - env.veh_set[ego_idx].v_length / 2,
                                            -xPred_rec[0].y[k] - env.veh_set[ego_idx].v_width / 2),
                                         env.veh_set[ego_idx].v_length, env.veh_set[ego_idx].v_width,
                                         fc=colorset[1], alpha=0.3, zorder=0)
                coords = ts.transform([xPred_rec[0].x[k], -xPred_rec[0].y[k]])
                tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred_rec[0].yaw[k])
                newpatch.set_transform(ts + tr)
                ax.add_patch(newpatch)

        plt.plot(xPred_rec[1].x, -xPred_rec[1].y, 'c--', linewidth=1)
        for k in range(0, xPred_rec[1].x.shape[0]):
            if k % 2 == 1:
                newpatch = plt.Rectangle((xPred_rec[1].x[k] - env.veh_set[ego_idx].v_length / 2,
                                            -xPred_rec[1].y[k] - env.veh_set[ego_idx].v_width / 2),
                                         env.veh_set[ego_idx].v_length, env.veh_set[ego_idx].v_width,
                                         fc=colorset[2], alpha=0.3, zorder=0)
                coords = ts.transform([xPred_rec[1].x[k], -xPred_rec[1].y[k]])
                tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred_rec[1].yaw[k])
                newpatch.set_transform(ts + tr)
                ax.add_patch(newpatch)

        plt.plot(xPred_rec[2].x, -xPred_rec[2].y, 'k--', linewidth=1)
        for k in range(0, xPred_rec[1].x.shape[0]):
            if k % 2 == 1:
                newpatch = plt.Rectangle((xPred_rec[2].x[k] - env.veh_set[ego_idx].v_length / 2,
                                            -xPred_rec[2].y[k] - env.veh_set[ego_idx].v_width / 2),
                                         env.veh_set[ego_idx].v_length, env.veh_set[ego_idx].v_width,
                                         fc=colorset[3], alpha=0.3, zorder=0)
                coords = ts.transform([xPred_rec[2].x[k], -xPred_rec[2].y[k]])
                tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred_rec[2].yaw[k])
                newpatch.set_transform(ts + tr)
                ax.add_patch(newpatch)
        '''
        for j in range(0, len(zPred_rec)):
            plt.plot(zPred_rec[j][:, 0], -zPred_rec[j][:, 1], 'r--', linewidth=2)

        if plot_merge:
            if env.merge_side == 0:
                plt.plot([-10, 1000], [-lm[0], -lm[0]], 'g', linewidth=2)
                for j in range(1, env.N_lane):
                    plt.plot([-10, 1000], [-lm[j], -lm[j]], 'g--', linewidth=1)

                plt.plot([-10, env.merge_s], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)
                plt.plot([env.merge_end, 1000], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)

                plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1, 'g', linewidth=2)
                plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2, 'g--', linewidth=1)
                for j in range(1, env.merge_lane):
                    plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1 - j * lane_width, 'g--', linewidth=1)
                    plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2 - j * lane_width, 'g--', linewidth=1)
                plt.plot(env.merge_lane_ref_X, -env.merge_lane_ref_Y - env.merge_lane * lane_width, 'g', linewidth=2)
            else:
                plt.plot([-10, 1000], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)
                for j in range(1, env.N_lane):
                    plt.plot([-10, 1000], [-lm[j], -lm[j]], 'g--', linewidth=1)

                plt.plot([-10, env.merge_s], [-lm[0], -lm[0]], 'g', linewidth=2)
                plt.plot([env.merge_end, 1000], [-lm[0], -lm[0]], 'g', linewidth=2)

                plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1, 'g', linewidth=2)
                plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2, 'g', linewidth=2)
                for j in range(1, env.merge_lane):
                    plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1 - j * lane_width, 'g--', linewidth=1)
                    plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2 - j * lane_width, 'g--', linewidth=1)
                plt.plot(env.merge_lane_ref_X, -env.merge_lane_ref_Y - env.merge_lane * lane_width, 'g', linewidth=2)
        else:
            plt.plot([xmin - 50, xmax + 50], [-lm[0], -lm[0]], 'g', linewidth=2)
            for j in range(1, env.N_lane):
                plt.plot([xmin - 50, xmax + 50], [-lm[j], -lm[j]], 'g--', linewidth=1)
            plt.plot([xmin - 50, xmax + 50], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)

        return veh_patch

    def animate_all(t, veh_patch, state_rec, xPred_rec, zPred_rec, env, ego_idx=0):
        plot_merge = isinstance(env, Highway_env_merge)
        N_veh = 2
        ego_y = state_rec[t][ego_idx][0][1]
        ego_x = state_rec[t][ego_idx][0][0]
        ax.clear()
        if plot_merge:
            xmin = ego_x - 5
            xmax = ego_x + 45
            ymin = -5
            ymax = 35
        else:
            xmin = ego_x - 10
            xmax = ego_x + 40
            ymin = ego_y - 10
            ymax = ego_y + 10
        try:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-ymax, -ymin)
        except:
            pdb.set_trace()
        ts = ax.transData
        for i in range(0, N_veh):
            coords = ts.transform([state_rec[t][i][0][0], -state_rec[t][i][0][1]])
            tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -state_rec[t][i][0][3])
            te = ts + tr
            veh_patch[i].set_xy(
                [state_rec[t][i][0][0] - env.veh_set[i].v_length / 2, -state_rec[t][i][0][1] - env.veh_set[i].v_width / 2])
            veh_patch[i].set_transform(te)
            ax.add_patch(veh_patch[i])

        colorset = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                    'tab:olive', 'tab:cyan', 'y', 'm', 'c', 'g']

        keys_list = list(xPred_rec[t].keys())
        j = 0
        for key in keys_list[:-1]:
            plt.plot(xPred_rec[t][key].x, -xPred_rec[t][key].y, 'b--', linewidth=2)
            for k in range(0, xPred_rec[t][key].x.shape[0]):
                if k % 2 == 1:
                    newpatch = plt.Rectangle((xPred_rec[t][key].x[k] - env.veh_set[ego_idx].v_length / 2,
                                                -xPred_rec[t][key].y[k] - env.veh_set[ego_idx].v_width / 2),
                                             env.veh_set[ego_idx].v_length, env.veh_set[ego_idx].v_width,
                                             fc=colorset[j], alpha=0.3, zorder=0)
                    coords = ts.transform([xPred_rec[t][key].x[k], -xPred_rec[t][key].y[k]])
                    tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred_rec[t][key].yaw[k])
                    newpatch.set_transform(ts + tr)
                    ax.add_patch(newpatch)
            j = j + 1

        for j in range(0, len(zPred_rec[t])):
            plt.plot(zPred_rec[t][j][:, 0], -zPred_rec[t][j][:, 1], 'r--', linewidth=2)

        plt.plot([xmin - 50, xmax + 50], [-lm[0], -lm[0]], 'g', linewidth=2)
        for j in range(1, env.N_lane):
            plt.plot([xmin - 50, xmax + 50], [-lm[j], -lm[j]], 'g--', linewidth=1)
        plt.plot([xmin - 50, xmax + 50], [-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)

        return veh_patch

    anim = animation.FuncAnimation(fig, animate_all, fargs=(
        veh_patch, state_rec, xPred_rec, zPred_rec, env, ego_idx,),
                                   frames=nframe,
                                   interval=50,
                                   blit=False, repeat=False)
    '''
    filepath = 'state_rec.pkl'
    with open(filepath, 'wb') as file:
        pickle.dump(state_rec, file)

    filepath = 'xPred_rec.pkl'
    with open(filepath, 'wb') as file:
        pickle.dump(xPred_rec, file)
    '''
    if output:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1 / env.dt), metadata=dict(artist='Me'), bitrate=1800)
        anim_name = output
        anim.save(anim_name, writer=writer)
    else:
        plt.show()


    # animate_all(0, veh_patch, state_rec, xPred_rec, zPred_rec, env, ego_idx)

    # animate(0, veh_patch, state_rec, xPred_rec, zPred_rec, env, ego_idx)

    '''
    if output:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1 / env.dt), metadata=dict(artist='Me'), bitrate=1800)
        anim_name = output
        anim.save(anim_name, writer=writer)
    else:
    '''
    # plt.ion()
    # plt.pause(0.000001)


def sim_overtake(mpc, N_lane, timestep, ego_state, obst_new_state, trajectory):
    env = Highway_env(NV=2, mpc=mpc, N_lane=N_lane, timestep=timestep, ego_state=ego_state,
                      obst_new_state=obst_new_state)

    backup, zPred, obst_new_state, state_rec, branch_w = Highway_sim(env, 0.1)
    '''
    state_rec, input_rec, backup_rec, backup_choice_rec, xPred_rec, zPred_rec, branch_w_rec, collision = Highway_sim(
        env, 10)


    animate_scenario(env, state_rec, backup_rec, backup_choice_rec, xPred_rec, zPred_rec, lm)
    br = np.reshape(np.array(branch_w_rec), [-1, 12])
    

    if timestep > 0:
        animate_scenario(env, state_rec, trajectory, zPred)
    '''
    return backup, zPred, obst_new_state, branch_w, state_rec


def sim_merge(mpc, pred_model, N_lane, merge_lane, merge_s, merge_R, merge_side):
    NV = 2
    env = Highway_env_merge(NV, N_lane, mpc, pred_model, merge_lane, merge_s, merge_R, merge_side, pred_model[0].dt)
    state_rec, input_rec, backup_rec, backup_choice_rec, xPred_rec, zPred_rec, branch_w_rec, collision = Highway_sim(
        env, 6)
    animate_scenario(env, state_rec, backup_rec, backup_choice_rec, xPred_rec, zPred_rec, lm)
