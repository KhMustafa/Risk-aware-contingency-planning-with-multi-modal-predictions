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

@dataclass
class PythonMsg:
    def __setattr__(self,key,value):
        if not hasattr(self,key):
            raise TypeError ('Cannot add new field "%s" to frozen class %s' %(key,self))
        else:
            object.__setattr__(self,key,value)
@dataclass
class Branch_constants:
     s1: float = field(default=None)
     s2: float = field(default=None)
     c2: float = field(default=None)
     tran_diag: float = field(default=None)
     alpha: float = field(default=None)
     R: float = field(default=None)
     am: float = field(default=None)
     rm: float = field(default=None)
     J_c: float = field(default=None)
     s_c: float = field(default=None)
     ylb: float = field(default=None)
     yub: float = field(default=None)
     W: float = field(default=None)
     L: float = field(default=None)
     col_alpha:float = field(default=None)
     Kpsi: float = field(default=None)

@dataclass
class Quad_constants:
     s1: float = field(default=None)
     s2: float = field(default=None)
     c2: float = field(default=None)
     alpha: float = field(default=None)
     R: float = field(default=None)
     vxm: float = field(default=None)
     vym: float = field(default=None)
     rm: float = field(default=None)
     W1: float = field(default=None)
     L1: float = field(default=None)
     W2: float = field(default=None)
     L2: float = field(default=None)
     col_tol: float = field(default=None)
     col_alpha:float = field(default=None)

@dataclass
class MPCParams(PythonMsg):
    n: int = field(default=None) # dimension state space
    d: int = field(default=None) # dimension input space
    N: int = field(default=None) # horizon length
    M: int = field(default=None) # number of other agents
    m: int = field(default=None) # number of backup policies

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
