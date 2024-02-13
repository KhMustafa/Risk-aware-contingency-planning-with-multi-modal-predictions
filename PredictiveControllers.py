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

solvers.options['show_progress'] = False

@dataclass
class PythonMsg:
    def __setattr__(self,key,value):
        if not hasattr(self,key):
            raise TypeError ('Cannot add new field "%s" to frozen class %s' %(key,self))
        else:
            object.__setattr__(self,key,value)

@dataclass
class MPCParams(PythonMsg):
    n: int = field(default=None) # dimension state space
    d: int = field(default=None) # dimension input space
    N: int = field(default=None) # horizon length

    A: np.array = field(default=None) # prediction matrices. Single matrix for LTI and list for LTV
    B: np.array = field(default=None) # prediction matrices. Single matrix for LTI and list for LTV

    Q: np.array = field(default=np.array((n, n))) # quadratic state cost
    R: np.array = field(default=None) # quadratic input cost
    Qf: np.array = field(default=None) # quadratic state cost final
    dR: np.array = field(default=None) # Quadratic rate cost

    Qslack: float = field(default=None) # it has to be a vector. Qslack = [linearSlackCost, quadraticSlackCost]
    Fx: np.array = field(default=None) # State constraint Fx * x <= bx
    bx: np.array = field(default=None)
    Fu: np.array = field(default=None) # State constraint Fu * u <= bu
    bu: np.array = field(default=None)
    xRef: np.array = field(default=None)

    slacks: bool = field(default=True)
    timeVarying: bool = field(default=False)

    def __post_init__(self):
        if self.Qf is None: self.Qf = np.zeros((self.n, self.n))
        if self.dR is None: self.dR = np.zeros(self.d)
        if self.xRef is None: self.xRef = np.zeros(self.n)

############################################################################################
####################################### MPC CLASS ##########################################
############################################################################################
class MPC():
    """Model Predicitve Controller class
    Methods (needed by user):
        solve: given system's state xt compute control action at
    Arguments:
        mpcParameters: model paramters
    """
    def __init__(self,  mpcParameters, predictiveModel):
        """Initialization
        Arguments:
            mpcParameters: struct containing MPC parameters
        """
        self.N      = mpcParameters.N
        self.Qslack = mpcParameters.Qslack
        self.Q      = mpcParameters.Q
        self.Qf     = mpcParameters.Qf
        self.R      = mpcParameters.R
        self.dR     = mpcParameters.dR
        self.n      = mpcParameters.n
        self.d      = mpcParameters.d
        self.A      = mpcParameters.A
        self.B      = mpcParameters.B
        self.Fx     = mpcParameters.Fx
        self.Fu     = mpcParameters.Fu
        self.bx     = mpcParameters.bx
        self.bu     = mpcParameters.bu
        self.xRef   = mpcParameters.xRef
        self.M      = predictiveModel.M
        self.m      = predictiveModel.m
        self.nx     = self.n-self.M*self.m
        self.h0     = []
        self.Jh     = []
        self.thres  = 0.1
        self.alphad = np.exp(-predictiveModel.alpha*predictiveModel.dt)

        self.slacks          = mpcParameters.slacks
        self.slackdim        = self.Fx.shape[0]*self.N+(self.N-1)*self.M*self.m
        self.timeVarying     = mpcParameters.timeVarying
        self.predictiveModel = predictiveModel
        self.osqp = None

        # if self.timeVarying == True:
        #     self.xLin = self.predictiveModel.xStored[-1][0:self.N+1,:]
        #     self.uLin = self.predictiveModel.uStored[-1][0:self.N,:]
        #

        self.OldInput = np.zeros((1,2)) # TO DO fix size



        self.xPred = None
        self.uLin = None

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0

    def get_xLin(self,x0,xbackup,b0):
        if self.uLin is None:
            self.uLin = np.zeros([self.N,self.d])
        self.uLin = np.vstack((self.uLin,self.uLin[-1]))
        self.xLin = np.zeros([self.N+1,self.n])
        xb = np.append(x0,np.reshape(b0,-1,1))

        self.xLin[0] = xb
        for i in range(0,self.N):
            A,B,C,h0,Jh=self.predictiveModel.regressionAndLinearization(xb,xbackup[:,i*self.nx:(i+1)*self.nx], self.uLin[i])
            xbp = C+A.dot(xb)+B.dot(self.uLin[i])
            self.xLin[i+1] = xbp
            xb = xbp

    def solve(self, x0,b0,xbackup,xRef=None):
        """Computes control action
        Arguments:
            x0: current state
        """
        # If LTV active --> identify system model
        if not xRef is None:
            self.xRef = np.append(xRef,np.zeros(self.M*self.m))


        self.get_xLin(x0,xbackup,b0)
        self.computeLTVdynamics(xbackup)
        self.buildIneqConstr()
        self.buildCost()
        self.buildEqConstr()

        xb0 = np.append(x0,np.reshape(b0,[-1,1]))
        self.addTerminalComponents(xb0)
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,xb0),self.L_FTOCP))
        self.unpackSolution()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        # If LTV active --> compute state-input linearization trajectory
        self.feasibleStateInput()
        if self.timeVarying == True:
            self.xLin = np.vstack((self.xPred[1:, :], self.zt))
            self.uLin = np.vstack((self.uPred[1:, :], self.zt_u))

        # update applied input
        self.OldInput = self.uPred[0,:]
        self.timeStep += 1


    def computeLTVdynamics(self,xbackup):
        # Estimate system dynamics
        self.A = []; self.B = []; self.C =[]; self.h0 = []; self.Jh = []
        for i in range(0, self.N):
            Ai, Bi, Ci, h0i, Jhi = self.predictiveModel.regressionAndLinearization(self.xLin[i+1], xbackup[:,i*self.nx:(i+1)*self.nx], self.uLin[i+1])
            self.A.append(Ai); self.B.append(Bi); self.C.append(Ci); self.h0.append(h0i); self.Jh.append(Jhi)

    def addTerminalComponents(self, x0):
        # TO DO: ....
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L

    def feasibleStateInput(self):
        self.zt   = self.xPred[-1,:]
        self.zt_u = self.uPred[-1,:]

    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
        self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.n*(self.N+1))]),(self.N+1,self.n)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.n*(self.N+1)+np.arange(self.d*self.N)]),(self.N, self.d)))).T
        self.xLin = self.xPred
        self.uLin = self.uPred
        self.uLin = np.vstack((self.uLin,self.uLin[-1]))

    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        rep_a = [self.Fx] * (self.N)
        Mat = linalg.block_diag(*rep_a)
        NoTerminalConstr = np.zeros((np.shape(Mat)[0], self.n))  # The last state is unconstrained. There is a specific function add the terminal constraints (so that more complicated terminal constrains can be handled)
        Fxtot = np.hstack((Mat, NoTerminalConstr))
        bxtot = np.tile(np.squeeze(self.bx), self.N)

        Fxbackup = np.zeros([(self.N-1)*self.M*self.m,self.n*(self.N+1)])
        bxbackup = np.zeros((self.N-1)*self.M*self.m)
        counter = 0
        for i in range(0,self.N-1):
            b = np.reshape(self.xLin[i+1][self.nx:],[self.M,self.m])
            for j in range(0,self.M):
                for k in range(0,self.m):
                    if b[j,k]>self.thres:
                        Fxbackup[counter][(i+1)*self.n:(i+2)*self.n] = -self.Jh[i+1][j][k]
                        bxbackup[counter] = self.h0[i+1][j][k]
                        # if self.h0[i][j][k]+self.Jh[i][j][k]@self.xLin[i]>0:
                        #     Fxbackup[counter][(i+1)*self.n:(i+2)*self.n] = -self.Jh[i+1][j][k]
                        #     Fxbackup[counter][i*self.n:(i+1)*self.n] = self.alphad*self.Jh[i][j][k]
                        #     bxbackup[counter] = self.h0[i+1][j][k]-self.alphad*self.h0[i][j][k]
                        # else:
                        #     Fxbackup[counter][(i+1)*self.n:(i+2)*self.n] = -self.Jh[i+1][j][k]
                        #     bxbackup[counter] = self.h0[i+1][j][k]
                        counter+=1

        Fxtot = np.vstack((Fxtot,Fxbackup[0:counter]))
        bxtot = np.append(bxtot,bxbackup[0:counter])
        self.slackdim = Fxtot.shape[0]
        # Let's start by computing the submatrix of F relates with the input
        rep_b = [self.Fu] * (self.N)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.N)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need
        if self.slacks == True:
            nc_x = Fxtot.shape[0] # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x))
            addSlack[0:nc_x, 0:nc_x] = -np.eye(nc_x)
            # Now constraint slacks >= 0
            I = - np.eye(nc_x); Zeros = np.zeros((nc_x, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack(( np.hstack((F_hard, addSlack)) , Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))

    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.n * (self.N + 1))
        Gu = np.zeros((self.n * (self.N + 1), self.d * (self.N)))

        E = np.zeros((self.n * (self.N + 1), self.n))
        E[np.arange(self.n)] = np.eye(self.n)

        L = np.zeros(self.n * (self.N + 1))

        for i in range(0, self.N):
            if self.timeVarying == True:
                Gx[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.n):(i*self.n + self.n)] = -self.A[i]
                Gu[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.d):(i*self.d + self.d)] = -self.B[i]
                L[(self.n + i*self.n):(self.n + i*self.n + self.n)]                                  =  self.C[i]
            else:
                Gx[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.n):(i*self.n + self.n)] = -self.A
                Gu[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.d):(i*self.d + self.d)] = -self.B

        if self.slacks == True:
            self.G = np.hstack( (Gx, Gu, np.zeros( ( Gx.shape[0], self.slackdim) ) ) )
        else:
            self.G = np.hstack((Gx, Gu))

        self.E = E
        self.L = L

    def buildCost(self):
        # The cost is: (1/2) * z' H z + q' z
        listQ = [self.Q] * (self.N)
        Hx = linalg.block_diag(*listQ)

        listTotR = [self.R + 2 * np.diag(self.dR)] * (self.N) # Need to add dR for the derivative input cost
        Hu = linalg.block_diag(*listTotR)
        # Need to condider that the last input appears just once in the difference
        for i in range(0, self.d):
            Hu[ i - self.d, i - self.d] = Hu[ i - self.d, i - self.d] - self.dR[i]

        # Derivative Input Cost
        OffDiaf = -np.tile(self.dR, self.N-1)
        np.fill_diagonal(Hu[self.d:], OffDiaf)
        np.fill_diagonal(Hu[:, self.d:], OffDiaf)

        # Cost linear term for state and input
        q = - 2 * np.dot(np.append(np.tile(self.xRef, self.N + 1), np.zeros(self.R.shape[0] * self.N)), linalg.block_diag(Hx, self.Qf, Hu))
        # Derivative Input (need to consider input at previous time step)
        q[self.n*(self.N+1):self.n*(self.N+1)+self.d] = -2 * np.dot( self.OldInput, np.diag(self.dR) )
        if self.slacks == True:
            quadSlack = self.Qslack[0] * np.eye(self.slackdim)
            linSlack  = self.Qslack[1] * np.ones(self.slackdim )
            self.H = linalg.block_diag(Hx, self.Qf, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else:
            self.H = linalg.block_diag(Hx, self.Qf, Hu)
            self.q = q

        self.H = 2 * self.H  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
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
        # if self.osqp is None:
        #     self.osqp = OSQP()
        #     self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        # else:
        #     self.osqp.update(Px=P.data,Ax=qp_A.data,q=q,l=qp_l, u=qp_u)
            # self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
            # self.osqp.update(Ax=qp_A,Ax_idx = qp_A.indices,l=qp_l, u=qp_u)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0
        self.Solution = res.x
