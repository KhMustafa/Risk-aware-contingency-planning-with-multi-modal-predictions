from casadi import *
import pdb
import itertools
import numpy as np
from scipy import sparse
from itertools import product

'''
Use CasADi to calculate the dynamics, propagate trajectories under the backup policies, and calculate the branching probabilities
'''


def dubin(x, u):
    if isinstance(x, numpy.ndarray):
        xdot = np.array([x[2] * cos(x[3]), x[2] * sin(x[3]), u[0], u[1]])
    elif isinstance(x, casadi.SX):
        xdot = SX(4, 1)
        xdot[0] = x[2] * cos(x[3])
        xdot[1] = x[2] * sin(x[3])
        xdot[2] = u[0]
        xdot[3] = u[1]
    elif isinstance(x, casadi.MX):
        xdot = MX(4, 1)
        xdot[0] = x[2] * cos(x[3])
        xdot[1] = x[2] * sin(x[3])
        xdot[2] = u[0]
        xdot[3] = u[1]
    return xdot


def softsat(x, s):
    return (np.exp(s * x) - 1) / (np.exp(s * x) + 1) * 0.5 + 0.5


def backup_maintain(x, cons, psiref=None):
    if psiref is None:
        if isinstance(x, casadi.MX):
            u = MX(2, 1)
            u[0] = 0
            u[1] = -cons.Kpsi * x[3]
            return u
        elif isinstance(x, casadi.SX):
            u = SX(2, 1)
            u[0] = 0
            u[1] = -cons.Kpsi * x[3]
            return u
        else:
            return np.array([0., -cons.Kpsi * x[3]])
    else:
        if isinstance(x, casadi.MX):
            u = MX(2, 1)
            u[1] = psiref(x[0]) - cons.Kpsi * x[3]
            return u
        elif isinstance(x, casadi.SX):
            u = SX(2, 1)
            u[1] = psiref(x[0]) - cons.Kpsi * x[3]
            return u
        else:
            return np.array([0., psiref(x[0]) - cons.Kpsi * x[3]])


def backup_brake(x, cons, psiref=None):
    if psiref is None:
        if isinstance(x, casadi.MX):
            u = MX(2, 1)
            u[0] = softmax(vertcat(-7, -x[2]), 5)
            u[1] = -cons.Kpsi * x[3]
            return u
        elif isinstance(x, casadi.SX):
            u = SX(2, 1)
            u[0] = softmax(vertcat(-7, -x[2]), 5)
            u[1] = -cons.Kpsi * x[3]
            return u
        else:
            return np.array([softmax(vertcat(-5, -x[2]), 3), -cons.Kpsi * x[3]])
    else:
        if isinstance(x, casadi.MX) or isinstance(x, casadi.SX):
            u = 0. * x[0:2]
            u[0] = softmax(vertcat(-5, -x[2]), 3)
            u[1] = psiref(x[0]) - cons.Kpsi * x[3]
            return u

        else:
            return np.array([softmax(vertcat(-5, -x[2]), 3), psiref(x[0]) - cons.Kpsi * x[3]])


def backup_lc(x, x0):
    if isinstance(x, casadi.MX):
        u = MX(2, 1)
        u[0] = -0.8558 * (x[2] - x0[2])
        u[1] = -0.3162 * (x[1] - x0[1]) - 3.9889 * (x[3] - x0[3])
        return u
    elif isinstance(x, casadi.SX):
        u = SX(2, 1)
        u[0] = -0.8558 * (x[2] - x0[2])
        u[1] = -0.3162 * (x[1] - x0[1]) - 3.9889 * (x[3] - x0[3])
        return u
    else:
        return np.array([-0.8558 * (x[2] - x0[2]), -0.3162 * (x[1] - x0[1]) - 3.9889 * (x[3] - x0[3])])


def softmin(x, gamma=1):
    if isinstance(x, casadi.SX) or isinstance(x, casadi.MX):
        return sum1(exp(-gamma * x) * x) / sum1(exp(-gamma * x))
    else:
        return np.sum(np.exp(-gamma * x) * x) / np.sum(np.exp(-gamma * x))


def softmax(x, gamma=1):
    if isinstance(x, casadi.SX) or isinstance(x, casadi.MX):
        return sum1(exp(gamma * x) * x) / sum1(exp(gamma * x))
    else:
        return np.sum(np.exp(gamma * x) * x) / np.sum(np.exp(gamma * x))


def propagate_backup(x, dyn, N, ts):
    '''
    Euler forward integration of the dynamics under the policy
    '''
    if isinstance(x, numpy.ndarray):
        xs = np.empty([N, x.shape[0]])
    elif isinstance(x, casadi.SX):
        xs = SX(N, x.shape[0])
    elif isinstance(x, casadi.MX):
        xs = MX(N, x.shape[0])
    # xs = np.zeros(N,x.shape[0])
    for i in range(0, N):
        x = x + dyn(x) * ts
        xs[i, :] = x
    return xs


def lane_bdry_h(x, lb=0, ub=7.2):
    if isinstance(x, casadi.SX):
        h = SX(x.shape[0], 1)
        for i in range(0, x.shape[0]):
            h[i] = softmin(vertcat(x[i, 1] - lb, ub - x[i, 1]), 5)
        return h
    elif isinstance(x, casadi.MX):
        h = MX(x.shape[0], 1)
        for i in range(0, x.shape[0]):
            h[i] = softmin(vertcat(x[i, 1] - lb, ub - x[i, 1]), 5)
        return h
    else:
        if x.ndim == 1:
            return softmin(np.array([x[1] - lb, ub - x[1]]), 5)
        else:
            h = np.zeros(x.shape[0])
            for i in range(0, x.shape[0]):
                h[i] = softmin(np.array([x[i, 1] - lb, ub - x[i, 1]]), 5)
            return h


def veh_col(x1, x2, size, alpha=1):
    '''
    vehicle collision constraints: h>=0 means no collision
    implemented via a softmax function
    '''
    if isinstance(x1, casadi.SX):
        h = SX(x1.shape[0])
        for i in range(0, x1.shape[0]):
            dx = (fabs(x1[i, 0] - x2[i, 0]) - size[0])
            dy = (fabs(x1[i, 1] - x2[i, 1]) - size[1])
            h[i] = (dx * np.exp(alpha * dx) + dy * np.exp(dy * alpha)) / (np.exp(alpha * dx) + np.exp(dy * alpha))
        return h
    elif isinstance(x1, casadi.MX):
        h = MX(x1.shape[0])
        for i in range(0, x1.shape[0]):
            dx = (fabs(x1[i, 0] - x2[i, 0]) - size[0])
            dy = (fabs(x1[i, 1] - x2[i, 1]) - size[1])
            h[i] = (dx * np.exp(alpha * dx) + dy * np.exp(dy * alpha)) / (np.exp(alpha * dx) + np.exp(dy * alpha))
        return h
    else:
        if x1.ndim == 1:
            dx = np.clip((abs(x1[0] - x2[0]) - size[0]), -5, 5)
            dy = np.clip((abs(x1[1] - x2[1]) - size[1]), -5, 5)
            return (dx * np.exp(alpha * dx) + dy * np.exp(dy * alpha)) / (np.exp(alpha * dx) + np.exp(dy * alpha))
        else:
            h = np.zeros(x1.shape[0])
            for i in range(0, x1.shape[0]):
                dx = np.clip((abs(x1[i][0] - x2[i][0]) - size[0]), -5, 5)
                dy = np.clip((abs(x1[i][1] - x2[i][1]) - size[1]), -5, 5)
                h[i] = (dx * np.exp(alpha * dx) + dy * np.exp(dy * alpha)) / (np.exp(alpha * dx) + np.exp(dy * alpha))
            return h


class PredictiveModel:
    def __init__(self, n, d, N, backupcons, dt, cons, N_lane=3):
        self.n = n  # state dimension
        self.d = d  # input dimension
        self.N = N  # number of prediction steps
        self.m = len(backupcons)  # number of policies
        self.dt = dt
        self.cons = cons  # parameters
        self.Asym = None
        self.Bsym = None
        self.xpsym = None
        self.hsym = None
        self.zpred = None
        self.xpred = None
        self.u0sym = None
        self.Jh = None
        self.LB = [cons.W / 2, N_lane * 3.6 - cons.W / 2]  # lane boundary
        self.backupcons = backupcons
        self.calc_xp_expr()

    def dyn_linearization(self, x, u):
        # linearizing the dynamics x^+=Ax+Bu+C
        A = self.Asym(x, u)
        B = self.Bsym(x, u)
        xp = self.xpsym(x, u)
        C = xp - A @ x - B @ u

        return np.array(A), np.array(B), np.squeeze(np.array(C)), np.squeeze(np.array(xp))

    def branch_eval(self, x, z):
        p = self.psym(x, z)
        dp = self.dpsym(x, z)
        return np.array(p).flatten(), np.array(dp)

    def zpred_eval(self, z):
        return np.array(self.zpred(z))

    def xpred_eval(self, x):
        return self.xpred(x), self.u0sym(x)

    def col_eval(self, x, z):
        dh = np.squeeze(np.array(self.dhsym(x, z)))
        h = np.squeeze(np.array(self.hsym(x, z)))
        return h - np.dot(dh, x), dh

    def update_backup(self, backupcons):
        self.backupcons = backupcons
        self.m = len(backupcons)
        self.calc_xp_expr()

    def BF_traj(self, x1, x2):
        if isinstance(x1, casadi.SX):
            h = SX(x1.shape[0] * 2, 1)
        elif isinstance(x1, casadi.MX):
            h = MX(x1.shape[0] * 2, 1)
        for i in range(0, x1.shape[0]):
            h[i] = veh_col(x1[i, :], x2[i, :], [self.cons.L + 2, self.cons.W + 0.2])
            h[i + x1.shape[0]] = lane_bdry_h(x1[i, :], self.LB[0], self.LB[1])
        return softmin(h, 5)

    def branch_prob(self, h):
        # branching probability as a function of the safety function
        h = softsat(h, 1)
        m = exp(self.cons.s1 * h)
        return m / sum1(m)

    def calc_xp_expr(self):
        u = SX.sym('u', self.d)
        x = SX.sym('x', self.n)
        z = SX.sym('z', self.n)
        zdot = SX.sym('zdot', self.m, self.n)

        xp = x + dubin(x, u) * self.dt

        dyn = lambda x: dubin(x, self.backupcons[0](x))
        x1 = propagate_backup(x, dyn, self.N, self.dt)
        x2 = SX(self.N, self.n * self.m)
        for i in range(0, self.m):
            dyn = lambda x: dubin(x, self.backupcons[i](x))
            x2[:, i * self.n:(i + 1) * self.n] = propagate_backup(z, dyn, self.N, self.dt)
        hi = SX(self.m, 1)
        for i in range(0, self.m):
            hi[i] = self.BF_traj(x2[:, self.n * i:self.n * (i + 1)], x1)
        p = self.branch_prob(hi)

        h = veh_col(x.T, z.T, [self.cons.L + 1, self.cons.W + 0.2], 1)

        self.xpsym = Function('xp', [x, u], [xp])
        self.Asym = Function('A', [x, u], [jacobian(xp, x)])
        self.Bsym = Function('B', [x, u], [jacobian(xp, u)])
        self.dhsym = Function('dh', [x, z], [jacobian(h, x)])
        self.hsym = Function('h', [x, z], [h])
        self.zpred = Function('zpred', [z], [x2])
        self.xpred = Function('xpred', [x], [x1])
        self.psym = Function('p', [x, z], [p])
        self.dpsym = Function('dp', [x, z], [jacobian(p, x)])
        self.u0sym = Function('u0', [x], [self.backupcons[0](x)])
