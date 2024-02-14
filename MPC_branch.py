import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from dataclasses import dataclass, field
import ecos

solvers.options['show_progress'] = False


@dataclass
class PythonMsg:
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise TypeError('Cannot add new field "%s" to frozen class %s' % (key, self))
        else:
            object.__setattr__(self, key, value)


@dataclass
class BranchMPCParams(PythonMsg):
    n: int = field(default=None)  # dimension state space
    d: int = field(default=None)  # dimension input space
    NB: int = field(default=None)  # number of branching
    N: int = field(default=None)  # number of time steps in between branches

    A: np.array = field(default=None)  # prediction matrices. Single matrix for LTI and list for LTV
    B: np.array = field(default=None)  # prediction matrices. Single matrix for LTI and list for LTV

    Q: np.array = field(default=np.array((n, n)))  # quadratic state cost
    R: np.array = field(default=None)  # quadratic input cost
    Qf: np.array = field(default=None)  # quadratic state cost final
    dR: np.array = field(default=None)  # Quadratic rate cost

    Qslack: float = field(default=None)  # it has to be a vector. Qslack = [linearSlackCost, quadraticSlackCost]
    Fx: np.array = field(default=None)  # State constraint Fx * x <= bx
    bx: np.array = field(default=None)
    Fu: np.array = field(default=None)  # State constraint Fu * u <= bu
    bu: np.array = field(default=None)
    xRef: np.array = field(default=None)

    slacks: bool = field(default=False)
    timeVarying: bool = field(default=False)

    def __post_init__(self):
        if self.Qf is None: self.Qf = self.Q
        if self.dR is None: self.dR = np.zeros(self.d)
        if self.xRef is None: self.xRef = np.zeros(self.n)


############################################################################################
####################################### MPC CLASS ##########################################
############################################################################################
class BranchTree:
    def __init__(self, xtraj, ztraj, utraj, w, depth=0):
        self.xtraj = xtraj
        self.ztraj = ztraj
        self.utraj = utraj
        self.dynmatr = [None] * xtraj.shape[0]
        self.w = w
        self.children = []
        self.depth = depth
        self.p = None
        self.dp = None
        self.J = 0

    def addchild(self, BT):
        self.children.append(BT)


class BranchMPC:

    def __init__(self, mpcParameters, predictiveModel):
        """Initialization
        Arguments:
            mpcParameters: struct containing MPC parameters
            predictiveModel: containing CasADi functions about z prediction, collision constraints, and branching probability
        """
        self.N = mpcParameters.N
        self.NB = mpcParameters.NB
        self.Qslack = mpcParameters.Qslack
        self.Q = mpcParameters.Q
        self.Qf = mpcParameters.Qf
        self.R = mpcParameters.R
        self.dR = mpcParameters.dR
        self.n = mpcParameters.n
        self.d = mpcParameters.d
        self.Fx = mpcParameters.Fx
        self.Fu = mpcParameters.Fu
        self.bx = mpcParameters.bx
        self.bu = mpcParameters.bu
        self.xRef = mpcParameters.xRef
        self.m = predictiveModel.m

        self.slacks = mpcParameters.slacks
        self.slackweight = None
        self.timeVarying = mpcParameters.timeVarying
        self.predictiveModel = predictiveModel
        self.osqp = None
        self.BT = None
        self.totalx = 0
        self.totalu = 0
        self.ndx = {}
        self.ndu = {}

        self.xPred = None
        self.uPred = None
        self.xLin = None
        self.uLin = None
        self.OldInput = np.zeros(self.d)

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0

    def inittree(self, x, z):
        '''
        Initialize the scenario tree, each node contains the z prediction, x prediction (from last iteration) and u prediction (from last iteration), and weight
        '''
        u = np.zeros(self.d)
        self.BT = BranchTree(np.reshape(x, [1, self.n]), np.reshape(z, [1, self.n]), np.reshape(u, [1, self.d]), 1, 0)
        q = [self.BT]
        countx = 0
        countu = 0
        self.uLin = np.reshape(u, [1, self.d])
        self.xLin = np.reshape(x, [1, self.n])

        self.ndx[self.BT] = countx
        self.ndu[self.BT] = countu
        A, B, C, xp = self.predictiveModel.dyn_linearization(x, u)
        self.BT.dynmatr[0] = (A, B, C)
        countx += self.BT.xtraj.shape[0]
        countu += self.BT.xtraj.shape[0]

        while len(q) > 0:
            currentbranch = q.pop(0)

            if currentbranch.depth < self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p, dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1], currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp = dp
                for i in range(0, self.m):
                    xtraj = np.zeros((self.N, self.n))
                    utraj = np.zeros((self.N, self.d))
                    newbranch = BranchTree(xtraj, zPred[:, self.n * i:self.n * (i + 1)], utraj, p[i] * currentbranch.w,
                                           currentbranch.depth + 1)
                    A, B, C, xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],
                                                                         currentbranch.utraj[-1])
                    newbranch.xtraj[0] = xp
                    for t in range(0, self.N):
                        A, B, C, xp = self.predictiveModel.dyn_linearization(newbranch.xtraj[t], newbranch.utraj[t])
                        newbranch.dynmatr[t] = (A, B, C)
                        if t < self.N - 1:
                            newbranch.xtraj[t + 1] = xp

                    self.ndx[newbranch] = countx
                    self.ndu[newbranch] = countu

                    self.xLin = np.vstack((self.xLin, newbranch.xtraj))
                    self.uLin = np.vstack((self.uLin, newbranch.utraj))
                    if newbranch.depth == self.NB:
                        countx += (newbranch.xtraj.shape[0] + 1)
                    else:
                        countx += newbranch.xtraj.shape[0]
                    countu += newbranch.xtraj.shape[0]
                    currentbranch.addchild(newbranch)
                    q.append(newbranch)
        self.totalx = countx
        self.totalu = countu
        self.slackweight = np.zeros(self.totalx * (self.Fx.shape[0] + 1))

    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.totalx * self.n)
        Gu = np.zeros((self.totalx * self.n, self.totalu * self.d))

        E = np.zeros((self.totalx * self.n, self.n))
        E[0:self.n] = np.eye(self.n)

        L = np.zeros(self.totalx * self.n)
        self.E = E

        totalxdim = self.totalx * self.n
        for branch in self.ndx:
            l = branch.xtraj.shape[0]
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            for t in range(1, l):
                A, B, C = branch.dynmatr[t - 1]
                Gx[(ndx + t) * self.n:(ndx + t + 1) * self.n, (ndx + t - 1) * self.n:(ndx + t) * self.n] = -A
                Gu[(ndx + t) * self.n:(ndx + t + 1) * self.n, (ndu + t - 1) * self.d:(ndu + t) * self.d] = -B
                L[(ndx + t) * self.n:(ndx + t + 1) * self.n] = C
            A, B, C = branch.dynmatr[-1]
            if branch.depth < self.NB:
                for child in branch.children:
                    ndxc = self.ndx[child]
                    Gx[ndxc * self.n:(ndxc + 1) * self.n, (ndx + l - 1) * self.n:(ndx + l) * self.n] = -A
                    Gu[ndxc * self.n:(ndxc + 1) * self.n, (ndu + l - 1) * self.d:(ndu + l) * self.d] = -B
                    L[ndxc * self.n:(ndxc + 1) * self.n] = C
            else:
                Gx[(ndx + l) * self.n:(ndx + l + 1) * self.n, (ndx + l - 1) * self.n:(ndx + l) * self.n] = -A
                Gu[(ndx + l) * self.n:(ndx + l + 1) * self.n, (ndu + l - 1) * self.d:(ndu + l) * self.d] = -B
                L[(ndx + l) * self.n:(ndx + l + 1) * self.n] = C
        self.L = L

        if self.slacks == True:
            self.G = np.hstack((Gx, Gu, np.zeros((Gx.shape[0], self.slackweight.shape[0]))))
        else:
            self.G = np.hstack((Gx, Gu))

    def updatetree(self, x, z):
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            branch.utraj[0:l - 1] = self.uLin[self.ndu[branch] + 1:self.ndu[branch] + l]
            if branch.depth < self.NB:
                idx = np.argmax(branch.p)
                ndu = self.ndu[branch.children[idx]]
                branch.utraj[-1] = self.uLin[ndu]
            else:
                branch.utraj[-1] = branch.utraj[-2]
        self.BT.ztraj = np.reshape(z, [1, self.n])
        self.BT.xtraj = np.reshape(x, [1, self.n])
        for i in range(0, self.BT.xtraj.shape[0]):
            A, B, C, xp = self.predictiveModel.dyn_linearization(self.BT.xtraj[i], self.BT.utraj[i])
            self.BT.dynmatr[i] = (A, B, C)
        q = [self.BT]

        while len(q) > 0:
            currentbranch = q.pop(0)
            if currentbranch.depth < self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p, dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1], currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp = dp
                for i in range(0, self.m):
                    child = currentbranch.children[i]
                    child.w = currentbranch.w * p[i]
                    child.ztraj = zPred[:, i * self.n:(i + 1) * self.n]
                    xtraj = np.zeros((self.N, self.n))
                    A, B, C, xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],
                                                                         currentbranch.utraj[-1])
                    child.xtraj[0] = xp
                    for t in range(0, self.N):
                        A, B, C, xp = self.predictiveModel.dyn_linearization(child.xtraj[t], child.utraj[t])
                        child.dynmatr[t] = (A, B, C)
                        if t < self.N - 1:
                            child.xtraj[t + 1] = xp

                    q.append(child)

    def buildCost(self):
        totalxdim = self.totalx * self.n
        listQ = [None] * (self.totalx)
        Hu = np.zeros([self.totalu * self.d, self.totalu * self.d])
        dRmat = np.diag(self.dR)
        qx = np.zeros(self.totalx * self.n)
        dQ = self.Q * 0.5
        for branch in self.ndx:
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            l = branch.utraj.shape[0]
            for i in range(0, l - 1):
                t = 1 + self.N * (branch.depth - 1) + i
                listQ[ndx + i] = (dQ + self.Q) * branch.w
                qx[(ndx + i) * self.n:(ndx + i + 1) * self.n] = -2 * branch.w * (
                        np.dot(self.xRef, self.Q) + np.dot(branch.xtraj[i], dQ))
                Hu[(ndu + i) * self.d:(ndu + i + 1) * self.d,
                (ndu + i) * self.d:(ndu + i + 1) * self.d] = branch.w * self.R

            if branch.depth < self.NB:
                Hu[(ndu + l - 1) * self.d:(ndu + l) * self.d,
                (ndu + l - 1) * self.d:(ndu + l) * self.d] = branch.w * self.R

                listQ[ndx + l - 1] = (dQ + self.Q) * branch.w
                childJ = np.zeros(self.m)
                for j in range(0, self.m):
                    childJ[j] = branch.children[j].J

                qx[(ndx + l - 1) * self.n:(ndx + l) * self.n] = branch.w * (
                        -2 * np.dot(self.xRef, self.Q) - 2 * np.dot(branch.xtraj[-1], dQ) + np.dot(childJ,
                                                                                                   branch.dp))

            else:
                Hu[(ndu + l - 1) * self.d:(ndu + l) * self.d,
                (ndu + l - 1) * self.d:(ndu + l) * self.d] = branch.w * self.R
                listQ[ndx + l - 1] = (dQ + self.Q) * branch.w
                listQ[ndx + l] = self.Qf * branch.w
                qx[(ndx + l - 1) * self.n:(ndx + l) * self.n] = -2 * branch.w * (
                        np.dot(self.xRef, self.Qf) + np.dot(branch.xtraj[-1], dQ))

        Hx = linalg.block_diag(*listQ)
        qu = np.zeros(self.totalu * self.d)
        qu[0:self.d] = -2 * self.OldInput @ self.dR

        # Cost linear term for state and input
        q = np.append(qx, qu)

        if self.slacks == True:
            quadSlack = self.Qslack[0] * np.eye(self.slackweight.shape[0])
            linSlack = self.Qslack[1] * self.slackweight
            self.H = linalg.block_diag(Hx, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else:
            self.H = linalg.block_diag(Hx, Hu)
            self.q = q
        self.H = 2 * self.H  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        Nc = self.Fx.shape[0] + 1
        slackweight_x = np.zeros(self.totalx * Nc)

        Fxtot = np.zeros([Nc * self.totalx, self.totalx * self.n])
        bxtot = np.zeros(Nc * self.totalx)
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0, l):
                # collision constraint linearized
                h, dh = self.predictiveModel.col_eval(branch.xtraj[i], branch.ztraj[i])
                idx = self.ndx[branch] + i
                Fxtot[idx * Nc:(idx + 1) * Nc, idx * self.n:(idx + 1) * self.n] = np.vstack((-dh, self.Fx))
                bxtot[idx * Nc:(idx + 1) * Nc] = np.append(h, self.bx)
                slackweight_x[idx * Nc:(idx + 1) * Nc] = branch.w

        self.slackweight = slackweight_x
        rep_b = [self.Fu] * (self.totalu)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.totalu)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need

        if self.slacks == True:
            nc_x = Fxtot.shape[0]  # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x))
            addSlack[0:nc_x, 0:nc_x] = -np.eye(nc_x)
            # Now constraint slacks >= 0
            I = - np.eye(nc_x);
            Zeros = np.zeros((nc_x, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack((np.hstack((F_hard, addSlack)), Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))

    def updateIneqConstr(self):
        # for warm_start, faster than building the constraints from scratch
        Nc = self.Fx.shape[0] + 1
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0, l):
                h, dh = self.predictiveModel.col_eval(branch.xtraj[i], branch.ztraj[i])
                idx = self.ndx[branch] + i
                self.F[idx * Nc, idx * self.n:(idx + 1) * self.n] = -dh
                self.b[idx * Nc] = h
                self.slackweight[idx * Nc:(idx + 1) * Nc] = branch.w

    def solve(self, x, z, xRef=None):
        """Computes control action
        Arguments:
            x: ego vehicle state
            z: uncontrolled vehicle state
        """

        if not xRef is None:
            self.xRef = xRef
        if self.BT is None:
            self.inittree(x, z)
            # self.buildIneqConstr()
        else:
            self.updatetree(x, z)
            # self.updateIneqConstr()
        '''
        self.buildCost()
        self.buildEqConstr()

        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP,
                           np.add(np.dot(self.E_FTOCP, x), self.L_FTOCP))
        self.unpackSolution()
        endTimer = datetime.datetime.now();
        deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        # print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")

        # update applied input
        self.OldInput = self.uPred[0, :]
        '''
        self.timeStep += 1

    def addTerminalComponents(self):
        # TO DO: ....
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L

    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
        if self.feasible:
            self.xPred = np.squeeze(
                np.transpose(np.reshape((self.Solution[np.arange(self.totalx * self.n)]), (-1, self.n)))).T
            self.uPred = np.squeeze(np.transpose(
                np.reshape((self.Solution[self.totalx * self.n + np.arange(self.totalu * self.d)]), (-1, self.d)))).T
            self.xLin = self.xPred
            self.uLin = self.uPred
            self.uLin = np.vstack((self.uLin, self.uLin[-1]))

    def BT2array(self):
        # for plotting and dubug maining, outputs the scenario tree as x and z trajectories
        ztraj = []
        xtraj = []
        utraj = []
        branch_w = []
        q = [self.BT]
        while (len(q) > 0):
            curr = q.pop(0)
            for child in curr.children:
                branch_w.append(child.w)
                ztraj.append(np.vstack((curr.ztraj[-1], child.ztraj)))
                xtraj.append(np.vstack((curr.xtraj[-1], child.xtraj)))
                utraj.append(np.vstack((curr.utraj[-1], child.utraj)))
                q.append(child)
        return xtraj, ztraj, utraj, branch_w

    def osqp_solve_qp(self, P, q, G=None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """
        qp_A = vstack([G, A]).tocsc()
        l = -inf * ones(len(h))
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])

        self.osqp = OSQP()

        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0

        self.Solution = res.x
