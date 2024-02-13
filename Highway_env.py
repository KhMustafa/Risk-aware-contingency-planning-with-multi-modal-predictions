import pdb
import osqp
import argparse
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
from HMM_backup_dyn import *

v0 = 15
lane_width = 3.6
lm = [0,3.6,7.2,10.8,14.4,18,21.6]
f0 = np.array([v0,0,0,0])


def with_probability(P=1):
    return np.random.uniform() <= P

class vehicle():
    def __init__(self, state=[0,0,v0,0],v_length=4,v_width=2.4,dt=0.05,backupidx=0,laneidx=0):
        self.state = np.array(state)
        self.dt = dt
        self.v_length = v_length
        self.v_width = v_width
        self.x_pred = []
        self.y_pred = []
        self.obs_rec_x = []
        self.obs_rec_y = []
        self.xbackup = None
        self.backupidx = backupidx
        self.laneidx = laneidx
    def step(self,u): # controlled vehicle
        dxdt = np.array([self.state[2]*np.cos(self.state[3]),self.state[2]*np.sin(self.state[3]),u[0],u[1]])
        self.state = self.state + dxdt*self.dt





class Highway_env():
    def __init__(self,NV,mpc,N_lane=6):
        self.dt = mpc.predictiveModel.dt
        self.veh_set = []
        self.NV = NV
        self.N_lane = N_lane
        self.desired_x = [None]*NV
        self.mpc = mpc
        self.backupcons = mpc.predictiveModel.backupcons
        self.b = np.ones([NV-1,len(self.backupcons)])/len(self.backupcons)
        self.m = len(self.backupcons)
        self.cons = mpc.predictiveModel.cons




        UB = 30
        LB = 0
        for i in range(0,NV):
            lane_number = math.floor(random.random()*N_lane)
            success = False
            while not success:
                Y = (lane_number+0.5)*lane_width+np.random.normal(0,0.1)
                X = random.random()*(UB-LB)+LB
                collision = False
                for veh in self.veh_set:
                    if abs(Y-veh.state[1])<=3 and abs(X-veh.state[0])<=8:
                        collision=True
                        break
                if not collision:
                    success = True
            self.veh_set.append(vehicle([X,Y,v0,0],dt=self.dt,backupidx = 0,laneidx = lane_number))
            lane_des = np.random.choice(N_lane)
            v_des = v0 + np.random.normal(0,5)
            if i==0:
                v_des = v0
            self.desired_x[i] = np.array([0,  lm[lane_des]+lane_width/2,v_des,0])




    def step(self):
        u_set = [None]*self.NV
        xx_set = [None]*self.NV
        QQ_set = [None]*self.NV
        u0_set = [None]*self.NV
        Qt_set = [None]*self.NV
        x_set = [None]*self.NV

        umax = np.array([self.cons.am,self.cons.rm])
        # generate backup trajectories
        self.xbackup = np.empty([0,(self.mpc.N+1)*4])
        for i in range(0,self.NV):
            xx_set[i] = [None]*self.m
            QQ_set[i] = [None]*self.m
            Qt_set[i] = [None]*self.m

            if abs(self.veh_set[i].state[1]-(1.8+self.veh_set[i].laneidx*3.6))<0.4:
                if i==0:
                    mindis = 1000
                    idx = 0
                    for ii in range(1,self.NV):
                        if self.veh_set[ii].laneidx!=self.veh_set[0].laneidx and abs(self.veh_set[ii].state[0]-self.veh_set[0].state[0])<mindis:
                            mindis = abs(self.veh_set[ii].state[0]-self.veh_set[0].state[0])
                            idx = ii
                    if mindis<4:
                        self.veh_set[0].laneidx = self.veh_set[idx].laneidx

                else:
                    if with_probability(0.05):
                        if self.veh_set[i].laneidx==0:
                            self.veh_set[i].laneidx = 1
                        elif self.veh_set[i].laneidx==self.N_lane-1:
                            self.veh_set[i].laneidx = self.N_lane-2
                        else:
                            if with_probability(0.5):
                                self.veh_set[i].laneidx +=1
                            else:
                                self.veh_set[i].laneidx-=1

            x0 = self.veh_set[i].state.copy()
            x0[1] = 1.8+self.veh_set[i].laneidx*3.6
            x0[2] = self.veh_set[0].state[2]+0.5*(self.veh_set[0].state[0]-self.veh_set[i].state[0])
            x0[3]=0

            con = lambda x:veh_con(x,x0,umax)
            u0_set[i]=con(self.veh_set[i].state)
            stop_crit = lambda x,t:t>self.dt*self.mpc.N+2
            for j in range(0,self.m):
                tt,xx,uu,QQ,Qt=generate_backup_traj(self.veh_set[i].state,self.backupcons[j],stop_crit,f0,self.dt,True)
                xx_set[i][j] = xx
                Qt_set[i][j] = Qt
                QQ_set[i][j] = QQ
                if i>0:
                    self.xbackup = np.vstack((self.xbackup,np.reshape(np.array(xx[0:self.mpc.N+1]),[1,-1])))


        Ydes = 1.8+self.veh_set[0].laneidx*3.6
        vdes = v0
        xRef = np.array([0,Ydes,vdes,0])
        # print(self.veh_set[1].backupidx)
        self.mpc.solve(self.veh_set[0].state,self.b,self.xbackup,xRef)

        u_set[0] = self.mpc.uPred[0]
        self.veh_set[0].step(u_set[0])
        x_set[0] = self.veh_set[0].state

        ## no control
        # u_set[0]=np.zeros(2)
        # self.veh_set[0].step(u_set[0])
        # x_set[0] = self.veh_set[0].state
        ## backup CBF QP
        for i in range(1,self.NV):
            A = []
            b = []

            x = self.veh_set[i].state
            fi,g = dubin_fg(x)
            for t in range(0,len(xx_set[i][self.veh_set[i].backupidx])):
                if t%3==0:
                    xi = xx_set[i][self.veh_set[i].backupidx][t]
                    h,dh = X_bdry(xi,[0,lm[self.N_lane]],self.veh_set[i].v_width)
                    if h<0.5:
                        dhdx = np.matmul(dh,QQ_set[i][self.veh_set[i].backupidx][t])
                        if norm(dhdx.dot(g))>1e-6:
                            A.append(-dhdx.dot(g))
                            b.append(dhdx.dot(fi-f0)-np.matmul(dh,Qt_set[i][self.veh_set[i].backupidx][t])+self.cons.alpha*h)

                    for j in range(0,self.NV):
                        if j!=i:
                            if t>=len(xx_set[j][self.veh_set[j].backupidx]):
                                xj = xx_set[j][self.veh_set[j].backupidx][-1]+f0*self.dt*(t-len(xx_set[j][self.veh_set[j].backupidx])+1)
                            else:
                                xj = xx_set[j][self.veh_set[j].backupidx][t]
                                eps = 1e-6
                                h = veh_col(xi,xj,[(self.veh_set[i].v_length+self.veh_set[j].v_length)/2+1,(self.veh_set[i].v_width+self.veh_set[j].v_width)/2+0.2])

                                if h<2:
                                    dh = np.zeros(4)
                                    dh[0] = (veh_col(xi+[eps,0,0,0],xj,[(self.veh_set[i].v_length+self.veh_set[j].v_length)/2+1,(self.veh_set[i].v_width+self.veh_set[j].v_width)/2+0.2])-h)/eps
                                    dh[1] = (veh_col(xi+[0,eps,0,0],xj,[(self.veh_set[i].v_length+self.veh_set[j].v_length)/2+1,(self.veh_set[i].v_width+self.veh_set[j].v_width)/2+0.2])-h)/eps
                                    dhdx = np.matmul(dh,QQ_set[i][self.veh_set[i].backupidx][t])
                                    if norm(dhdx.dot(g))>1e-6:
                                        A.append(-dhdx.dot(g))
                                        b.append(dhdx.dot(fi-f0)+self.cons.alpha*h-np.matmul(dh,Qt_set[i][self.veh_set[i].backupidx][t]))
            if A:
                A = np.array(A)
                A = np.append(A, -np.ones([A.shape[0],1]), 1)
                AA = np.concatenate((A,np.identity(3)))
                AA = sparse.csc_matrix(AA)

                ub = np.append(np.append(np.array(b),np.array(umax)),np.inf)
                lb = np.append(np.append(-np.inf*np.ones(len(b)),np.array(-umax)),0.0)
                P = np.eye(3)
                P[2][2]=0
                P = sparse.csc_matrix(P)
                q = np.append(-u0_set[i],1e6)
                prob = osqp.OSQP()
                prob.setup(P, q, AA, lb, ub, alpha=1.0,verbose=False)
                res = prob.solve()


                if res.info.status_val == 1 or res.info.status_val == 2:
                    u_set[i] = res.x[0:2]
                else:
                    if res.x.shape[0]>0:
                        u_set[i] = res.x[0:2]
                    else:
                        u_set[i] = u0_set[i]

            else:
                uu = np.maximum(u0_set[i],-umax)
                u_set[i] = np.minimum(uu,umax)
            self.veh_set[i].step(u_set[i])
            x_set[i] = self.veh_set[i].state

            ## update belief and replace vehicles far away from host vehicle
            if self.veh_set[i].state[0]-self.veh_set[0].state[0]>15 or self.veh_set[i].state[0]-self.veh_set[0].state[0]<-15:
                succ = self.replace_veh(i,0)
                if not succ:
                    self.replace_veh(i,2)
            else:
                cbfcond = np.zeros(self.m)
                xdot = dubin(x_set[i],u_set[i])
                hi = np.zeros(self.m)
                for j in range(0,self.m):
                    hij = np.zeros(self.mpc.N)
                    dhij = np.zeros(self.mpc.N)

                    for tt in range(0,self.mpc.N):
                        hij[tt] = veh_col(xx_set[i][j][tt],self.mpc.xPred[tt][0:4],[(self.veh_set[i].v_length+self.veh_set[0].v_length)/2,\
                        (self.veh_set[i].v_width+self.veh_set[0].v_width)/2],self.cons.col_alpha)
                        dh = np.zeros(4)
                        dh[0] = (veh_col(xx_set[i][j][tt]+[eps,0,0,0],self.mpc.xPred[tt][0:4],[(self.veh_set[i].v_length+self.veh_set[0].v_length)/2,\
                        (self.veh_set[i].v_width+self.veh_set[0].v_width)/2],self.cons.col_alpha)-hij[tt])/eps
                        dh[0] = (veh_col(xx_set[i][j][tt]+[0,eps,0,0],self.mpc.xPred[tt][0:4],[(self.veh_set[i].v_length+self.veh_set[0].v_length)/2,\
                        (self.veh_set[i].v_width+self.veh_set[0].v_width)/2],self.cons.col_alpha)-hij[tt])/eps
                        dhij[tt] = dh@(QQ_set[i][j][tt]@(xdot-f0)-Qt_set[i][j][tt])
                    hi[j] = np.min(hij)
                    cbfcond[j] = np.mean(hij+dhij)

                bi = self.b[i-1].copy()
                H = backup_trans(hi,self.cons)
                bi = bi@H
                for j in range(0,self.m):
                    bi[j] = bi[j]*backup_input_prob(cbfcond[j],self.cons)
                self.b[i-1]=bi/np.sum(bi)

                ## change backup
                self.veh_set[i].backupidx = np.random.choice(range(0,self.m), 1, p=H[self.veh_set[i].backupidx])[0]
                if np.isnan(self.b).any():
                    pdb.set_trace()
        return u_set,x_set,xx_set,self.mpc.xPred[1:,0:4]

    def replace_veh(self,idx,dir = 2):
        if idx==0:
            return
        if dir ==0:
            UB = self.veh_set[0].state[0]+13
            LB = self.veh_set[0].state[0]+8
        elif dir==1:
            UB = self.veh_set[0].state[0]-5
            LB = self.veh_set[0].state[0]-13
        else:
            UB = self.veh_set[0].state[0]+15
            LB = self.veh_set[0].state[0]-15

        if self.veh_set[0].laneidx==0:
            laneidx = 1
        elif self.veh_set[0].laneidx==self.N_lane-1:
            laneidx = self.N_lane-2
        else:
            if with_probability(0.5):
                laneidx = self.veh_set[0].laneidx-1
            else:
                laneidx = self.veh_set[0].laneidx+1
        success = False
        count = 0
        while not success:
            count+=1
            Y = (laneidx+0.5)*lane_width+np.random.normal(0,0.1)
            X = random.random()*(UB-LB)+LB
            collision = False
            for i in range(0,self.NV):
                if i!=idx:
                    if abs(Y-self.veh_set[i].state[1])<=2.2 and abs(X-self.veh_set[i].state[0])<=5:
                        collision=True
                        break
            if not collision:
                success = True
            if count>20:
                return False
        self.veh_set[idx] = vehicle([X,Y,self.veh_set[0].state[2],0],dt=self.dt,backupidx = 0,laneidx = laneidx)
        self.b[idx-1] = np.ones(self.m)/self.m
        return True





def Highway_sim(env,T):
    collision = False
    dt = env.dt
    t=0
    Ts_update = 4
    N_update = int(round(Ts_update/dt))
    N = int(round(T/dt))
    state_rec = np.zeros([env.NV,N,4])
    b_rec = [None]*N
    backup_rec = [None]*env.NV
    backup_choice_rec = [None]*env.NV
    xPred_rec = [None]*N
    for i in range(0,env.NV):
        backup_rec[i]=[None]*N
        backup_choice_rec[i] = [None]*N
    input_rec = np.zeros([env.NV,N,2])
    f0 = np.array([v0,0,0,0])
    for i in range(0,len(env.veh_set)):
        state_rec[i][t]=env.veh_set[i].state
    y_des = [None]*env.NV
    for i in range(0,env.NV):
        y_des[i]=env.desired_x[i][1]
    xx_set = []
    dis = 100
    while t<N:
        if not collision:
            for i in range(0,env.NV):
                for j in range(0,env.NV):
                    if i!=j:
                        dis = max(abs(env.veh_set[i].state[0]-env.veh_set[j].state[0])-0.5*(env.veh_set[i].v_length+env.veh_set[j].v_length),\
                        abs(env.veh_set[i].state[1]-env.veh_set[j].state[1])-0.5*(env.veh_set[i].v_width+env.veh_set[j].v_width))
                if dis<0:
                    # pdb.set_trace()
                    collision = True

        print("t=",t*env.dt)
        if t%N_update==0:
            for i in range(0,env.NV):
                p = np.random.random()
                if p>0.5:
                    lane_des = np.random.choice(env.N_lane)
                else:
                    lane_des = int(env.veh_set[i].state[1]/3.6)
                    lane_des = max(lane_des,0)
                    lane_des = min(lane_des,env.N_lane-1)

                if i==0:
                    v_des = v0 + np.random.randn()*8
                    env.desired_x[i] = np.array([0, lm[lane_des]+lane_width/2,v_des,0])
                else:
                    if env.veh_set[i].state[0]>env.veh_set[0].state[0]+6:
                        v_des = env.desired_x[0][2] - np.random.random()*4

                    elif env.veh_set[i].state[0]<env.veh_set[0].state[0]-6:
                        v_des = env.desired_x[0][2] + np.random.random()*4
                    else:
                        v_des = env.desired_x[i][2]+np.random.randn()*4
                    env.desired_x[i][1]=lm[lane_des]+lane_width/2
                    env.desired_x[i][2] = v_des
                y_des[i]=env.desired_x[i][1]




        u_set,x_set,xx_set,xPred=env.step()
        xPred_rec[t]=xPred
        for i in range(0,env.NV):
            input_rec[i][t]=u_set[i]
            state_rec[i][t]=x_set[i]
            backup_rec[i][t]=xx_set[i]
            backup_choice_rec[i][t] = env.veh_set[i].backupidx
        b_rec[t] = env.b.copy()
        t=t+1

    return state_rec,input_rec,backup_rec,backup_choice_rec,b_rec,xPred_rec,collision
def animate_scenario(env,state_rec,backup_rec,backup_choice_rec,b_rec,xPred_rec,lm,output=None):
    if output:
        matplotlib.use("Agg")
    ego_idx = 0
    fig = plt.figure(figsize=(10,4))
    # plt.xlim(0, env.N_lane*lane_width)
    ax = fig.add_subplot(111)
    plt.grid()

    nframe = len(state_rec[0])
    ego_veh = env.veh_set[ego_idx]
    veh_patch = [None]*env.NV
    for i in range(0,env.NV):
        if i==ego_idx:
            veh_patch[i]=plt.Rectangle((ego_veh.state[0]-ego_veh.v_length/2,ego_veh.state[1]-ego_veh.v_width/2), ego_veh.v_length,ego_veh.v_width, fc='r', zorder=0)
        else:
            veh_patch[i]=plt.Rectangle((ego_veh.state[0]-ego_veh.v_length/2,ego_veh.state[1]-ego_veh.v_width/2), ego_veh.v_length,ego_veh.v_width, fc='b', zorder=0)

    for patch in veh_patch:
        ax.add_patch(patch)

    def animate(t,veh_patch,state_rec,backup_rec,backup_choice_rec,b_rec,xPred_rec,env,ego_idx=0):

        N_veh = len(state_rec)
        ego_y = state_rec[ego_idx][t][1]
        ego_x = state_rec[ego_idx][t][0]
        ax.clear()

        xmin = ego_x-20
        xmax = ego_x+30
        ymin = ego_y-10
        ymax = ego_y+10
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ts = ax.transData
        # props = dict(boxstyle='none', facecolor='wheat', alpha=0.5)
        for i in range(0,N_veh):
            coords = ts.transform([state_rec[i][t][0],state_rec[i][t][1]])
            tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], state_rec[i][t][3])
            te= ts + tr
            veh_patch[i].set_xy([state_rec[i][t][0]-env.veh_set[i].v_length/2,state_rec[i][t][1]-env.veh_set[i].v_width/2])
            veh_patch[i].set_transform(te)
            ax.add_patch(veh_patch[i])
            idx = backup_choice_rec[i][t]
            # xp = (state_rec[i][t][1]-xmin)/(xmax-xmin)
            # yp = (state_rec[i][t][0]-ymin)/(ymax-ymin)
            if i!=0:
                ax.text(state_rec[i][t][0], state_rec[i][t][1], str(b_rec[t][i-1][0])[0:4], fontsize=14,verticalalignment='top',ha='center')
            if i==0:
                for j in range(0,len(xPred_rec[t])):
                    if j%2==0:
                        newpatch = plt.Rectangle((xPred_rec[t][j][0]-env.veh_set[ego_idx].v_length/2,xPred_rec[t][j][1]-env.veh_set[ego_idx].v_width/2), env.veh_set[ego_idx].v_length,env.veh_set[ego_idx].v_width, fc='y',alpha=0.3, zorder=0)
                        coords = ts.transform([xPred_rec[t][j][0],xPred_rec[t][j][1]])
                        tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], xPred_rec[t][j][3])
                        newpatch.set_transform(ts+tr)
                        ax.add_patch(newpatch)
            else:
                for j in range(1,len(backup_rec[i][t][0])):
                    if (j-1)%3==0:

                        newpatch = plt.Rectangle((backup_rec[i][t][idx][j][0]-env.veh_set[i].v_length/2,backup_rec[i][t][idx][j][1]-env.veh_set[i].v_width/2), env.veh_set[i].v_length,env.veh_set[i].v_width, fc='m',alpha=0.3, zorder=0)
                        coords = ts.transform([backup_rec[i][t][idx][j][0],backup_rec[i][t][idx][j][1]])
                        tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], backup_rec[i][t][idx][j][3])
                        newpatch.set_transform(ts+tr)
                        ax.add_patch(newpatch)

        plt.plot([xmin-50, xmax+50],[lm[0], lm[0]], 'g', linewidth=2)
        for j in range(1, env.N_lane):
            plt.plot([xmin-50, xmax+50],[lm[j], lm[j]], 'g--', linewidth=1)
        plt.plot([xmin-50, xmax+50],[lm[env.N_lane], lm[env.N_lane]], 'g', linewidth=2)



        return veh_patch
    anim = animation.FuncAnimation(fig, animate, fargs=(veh_patch,state_rec,backup_rec,backup_choice_rec,b_rec,xPred_rec,env,ego_idx,),
                                   frames=nframe,
                                   interval=50,
                                   blit=False, repeat=False)


    if output:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1/env.dt), metadata=dict(artist='Me'), bitrate=1800)
        anim_name = output
        anim.save(anim_name,writer=writer)
    else:
        plt.show()


def sim(mpc,N_lane):
    env = Highway_env(NV=mpc.M+1,mpc = mpc,N_lane=N_lane)
    state_rec,input_rec,backup_rec,backup_choice_rec,b_rec,xPred_rec,collision = Highway_sim(env,15)
    animate_scenario(env,state_rec,backup_rec,backup_choice_rec,b_rec,xPred_rec,lm,'movie1.mp4')
