# %%
import numpy as np
import pandas as pd
import sys
import itertools
from scipy.interpolate import interpn
import os
import subprocess
import time
import pyvista as pv

def Bspline(u,n):#u is a vector, n is a scalar
	if n==0:
		return ((1-u)**3)/6
	elif n==1:
		return (3*u**3-6*u**2+4)/6
	elif n==2:
		return (-3*u**3+3*u**2+3*u+1)/6
	elif n==3:
		return (u**3)/6
def dBspline(u,n):
	if n==0:
		return (-(1-u)**2)/2
	elif n==1:
		return (3*u**2-4*u)/2
	elif n==2:
		return (-3*u**2+2*u+1)/2
	elif n==3:
		return (u**2)/2
def multDefGrads(F1,F2):
	indmap = np.array([[0,1,2],[3,4,5],[6,7,8]])
	Fmult = np.zeros(F1.shape)
	for alpha in range(3):
		for beta in range(3):
			for delta in range(3):
				Fmult[:,indmap[alpha,beta]]+=F1[:,indmap[alpha,delta]]*F2[:,indmap[delta,beta]]
	return Fmult
# %%
class PV:
    def __init__(self, points) -> None:
        self._points = points
    
    @property
    def points(self):
        return self._points
# %%
gth = 0
df = pd.read_csv('disp_32vox.csv')
Rcv = df[['Pos. x [vox]','Pos. y [vox]','Pos. z [vox]']].to_numpy(dtype=np.float32)
print(Rcv.shape)
# ptcv = PV(Rcv)
ptcv = pv.PolyData(Rcv)
gv = np.ones_like(Rcv[:,0])
# ptcv.point_data['grayscale'] = gv
# %%
offy = -0.5*1000
mplowX = -0.5*1000
mplowY = -0.5*1000
mplowZ = -0.1*1000
mphighX = 0.5*1000
mphighY = 0.5*1000
mphighZ = 0.1*1000

# maskmp = (gv>=gth) & (ptcv.points[:,1]>mplowY-offy) & (ptcv.points[:,1]<mphighY+offy) & (ptcv.points[:,2]>mplowZ-offy) & (ptcv.points[:,2]<mphighZ+offy) & (ptcv.points[:,0]>mplowX-offy) & (ptcv.points[:,0]<mphighX+offy)
maskmp = np.arange(Rcv.shape[0])
Rmp = Rcv[maskmp,:] # coordinates of material points
# %%
gridsize = np.linalg.norm(Rcv[0,:]-Rcv[1,:]) # need uniform grid in all directions
dimtol = gridsize/4

Zmaxnumber = int(np.round((np.max(ptcv.points[:,2])-np.min(ptcv.points[:,2]))/gridsize))+1
Ymaxnumber = int(np.round((np.max(ptcv.points[:,1])-np.min(ptcv.points[:,1]))/gridsize))+1
Xmaxnumber = int(np.round((np.max(ptcv.points[:,0])-np.min(ptcv.points[:,0]))/gridsize))+1

condtrim = (Rcv[:,0]<(Rcv[0,0]+(Xmaxnumber-1)*gridsize-dimtol))&(Rcv[:,1]<(Rcv[0,1]+(Ymaxnumber-1)*gridsize-dimtol))&(Rcv[:,2]<(Rcv[0,2]+(Zmaxnumber-1)*gridsize-dimtol))

elemcentroid = Rcv[condtrim,:] + np.array([gridsize/2,gridsize/2,gridsize/2])

Xnumberfloor = (np.floor((elemcentroid[:,0]-ptcv.points[0,0])/gridsize))
Xnumberceil = (np.ceil((elemcentroid[:,0]-ptcv.points[0,0])/gridsize))
Ynumberfloor = (np.floor((elemcentroid[:,1]-ptcv.points[0,1])/gridsize))
Ynumberceil = (np.ceil((elemcentroid[:,1]-ptcv.points[0,1])/gridsize))
Znumberfloor = (np.floor((elemcentroid[:,2]-ptcv.points[0,2])/gridsize))
Znumberceil = (np.ceil((elemcentroid[:,2]-ptcv.points[0,2])/gridsize))
# %%
connectivity =[]

for Z in [Znumberfloor,Znumberceil]:
    for Y in [Ynumberfloor,Ynumberceil]:
        for X in [Xnumberfloor,Xnumberceil]:
            connectivity.append(X+Y*Xmaxnumber+Z*Xmaxnumber*Ymaxnumber)

connectivity = np.array(connectivity,dtype=np.int32)
connectivity = np.transpose(connectivity)
# %%
nodesperlelem = connectivity.shape[1]
connectivity= connectivity[:,[0,1,3,2,4,5,7,6]]
nelems = 8*np.ones(connectivity.shape[0],dtype=np.int32)
connectivity = np.column_stack((nelems,connectivity))
cells = connectivity.flatten()
celltypes= 12*np.ones(connectivity.shape[0],dtype=np.int32)
ptcv = pv.UnstructuredGrid(cells, celltypes, Rcv)
L = gridsize
disp_calc = np.zeros(Rcv.shape)
disp_calc[~maskmp,:]=np.nan
offset =0
# %%
for i in [1]:
    Xnumberfloor = (np.floor((Rmp[:,0]+offset-ptcv.points[0,0])/gridsize))
    Xnumberceil = (np.ceil((Rmp[:,0]+offset-ptcv.points[0,0])/gridsize))
    Ynumberfloor = (np.floor((Rmp[:,1]+offset-ptcv.points[0,1])/gridsize))
    Ynumberceil = (np.ceil((Rmp[:,1]+offset-ptcv.points[0,1])/gridsize))
    Znumberfloor = (np.floor((Rmp[:,2]+offset-ptcv.points[0,2])/gridsize))
    Znumberceil = (np.ceil((Rmp[:,2]+offset-ptcv.points[0,2])/gridsize))

    connectivitymp =[]

    for Z in [Znumberfloor,Znumberceil]:
        for Y in [Ynumberfloor,Ynumberceil]:
            for X in [Xnumberfloor,Xnumberceil]:
                connectivitymp.append(X+Y*Xmaxnumber+Z*Xmaxnumber*Ymaxnumber)

    connectivitymp = np.array(connectivitymp)
    connectivitymp = np.transpose(connectivitymp)

    nanpoints = np.isnan(connectivitymp)
    nanpoints = np.sum(nanpoints,axis=1,dtype=np.bool_)
    connectivitymp = connectivitymp.astype(np.int32)
    elemcentroidmpx = np.array([np.nan]*Rmp.shape[0])
    elemcentroidmpy = np.array([np.nan]*Rmp.shape[0])
    elemcentroidmpz = np.array([np.nan]*Rmp.shape[0])
    elemcentroidmpx[~nanpoints] = np.mean(Rcv[connectivitymp[~nanpoints,:],0],axis=1)
    elemcentroidmpy[~nanpoints] = np.mean(Rcv[connectivitymp[~nanpoints,:],1],axis=1)
    elemcentroidmpz[~nanpoints] = np.mean(Rcv[connectivitymp[~nanpoints,:],2],axis=1)

    ump_spline = np.zeros((Rmp.shape[0],3))
    incFmp_iX = np.zeros((Rmp.shape[0],3))
    incFmp_iY = np.zeros((Rmp.shape[0],3))
    incFmp_iZ = np.zeros((Rmp.shape[0],3))
    # df = pd.read_csv('Al_60vox_normal.csv')
    df = pd.read_csv(f'./disp_32vox.csv')
    Ucv = df[['Displacement x [µm]','Displacement y [µm]','Displacement z [µm]']].to_numpy(dtype=np.float32)
    # strains = df[['Strain tensor xx','Strain tensor yy','Strain tensor zz','Strain tensor xy','Strain tensor yz','Strain tensor xz']].to_numpy(dtype=np.float32)

    icvpoints = np.mod(np.mod(connectivitymp[~nanpoints,0],Xmaxnumber*Ymaxnumber),Xmaxnumber)-1
    jcvpoints = np.int32(np.floor(np.mod(connectivitymp[~nanpoints,0],Xmaxnumber*Ymaxnumber)/Xmaxnumber))-1
    kcvpoints = np.int32(np.floor(connectivitymp[~nanpoints,0]/(Xmaxnumber*Ymaxnumber)))-1

    splinepoints = np.ones(icvpoints.shape,dtype=np.bool_).reshape(-1,1)

    for l in range(4):
        for m in range(4):
            for n in range(4):
                splinepoints = splinepoints & np.isnan(Ucv[(icvpoints+l)+(jcvpoints+m)*Xmaxnumber+(kcvpoints+n)*Xmaxnumber*Ymaxnumber,0]).reshape(-1,1)
    nanpoints = nanpoints | splinepoints.squeeze()
    icvpoints = icvpoints[~splinepoints.squeeze()]
    jcvpoints = jcvpoints[~splinepoints.squeeze()]
    kcvpoints = kcvpoints[~splinepoints.squeeze()] # Removing all material-points where entire cube has nan values.
    splinepoints2 = np.zeros(icvpoints.shape,dtype=np.bool_).reshape(-1,1)
    for l in range(4):
        for m in range(4):
            for n in range(4):
                splinepoints2 = splinepoints2 | np.isnan(Ucv[(icvpoints+l)+(jcvpoints+m)*Xmaxnumber+(kcvpoints+n)*Xmaxnumber*Ymaxnumber,0]).reshape(-1,1)
    splinepoints2 = splinepoints2.squeeze()
    badicvpoints = icvpoints[splinepoints2]
    badjcvpoints = jcvpoints[splinepoints2]
    badkcvpoints = kcvpoints[splinepoints2]
    for count in range(badicvpoints.shape[0]):
        replmn = np.array([],dtype=np.int32)
        for l in range(4):
            for m in range(4):
                for n in range(4):
                    if ~np.isnan(Ucv[(badicvpoints[count]+l)+(badjcvpoints[count]+m)*Xmaxnumber+(badkcvpoints[count]+n)*Xmaxnumber*Ymaxnumber,0]):
                        replmn = np.array([l,m,n],dtype=np.int32)
        for l in range(4):
            for m in range(4):
                for n in range(4):
                    if np.isnan(Ucv[(badicvpoints[count]+l)+(badjcvpoints[count]+m)*Xmaxnumber+(badkcvpoints[count]+n)*Xmaxnumber*Ymaxnumber,0]):
                        Ucv[(badicvpoints[count]+l)+(badjcvpoints[count]+m)*Xmaxnumber+(badkcvpoints[count]+n)*Xmaxnumber*Ymaxnumber,:] = Ucv[(badicvpoints[count]+replmn[0])+(badjcvpoints[count]+replmn[1])*Xmaxnumber+(badkcvpoints[count]+replmn[2])*Xmaxnumber*Ymaxnumber,:]

    uvw = (Rmp[~nanpoints,:]-Rcv[connectivitymp[~nanpoints,0],:])/gridsize
    for l in range(4):
        for m in range(4):
            for n in range(4):
                temp = Bspline(uvw[:,0],l)*Bspline(uvw[:,1],m)*Bspline(uvw[:,2],n)
                ump_spline[~nanpoints,:] += temp.reshape((-1,1))*Ucv[(icvpoints+l)+(jcvpoints+m)*Xmaxnumber+(kcvpoints+n)*Xmaxnumber*Ymaxnumber,:]
                temp = dBspline(uvw[:,0],l)*Bspline(uvw[:,1],m)*Bspline(uvw[:,2],n)
                incFmp_iX[~nanpoints,:] += temp.reshape((-1,1))/gridsize*Ucv[(icvpoints+l)+(jcvpoints+m)*Xmaxnumber+(kcvpoints+n)*Xmaxnumber*Ymaxnumber,:]
                temp = Bspline(uvw[:,0],l)*dBspline(uvw[:,1],m)*Bspline(uvw[:,2],n)
                incFmp_iY[~nanpoints,:] += temp.reshape((-1,1))/gridsize*Ucv[(icvpoints+l)+(jcvpoints+m)*Xmaxnumber+(kcvpoints+n)*Xmaxnumber*Ymaxnumber,:]
                temp = Bspline(uvw[:,0],l)*Bspline(uvw[:,1],m)*dBspline(uvw[:,2],n)
                incFmp_iZ[~nanpoints,:] += temp.reshape((-1,1))/gridsize*Ucv[(icvpoints+l)+(jcvpoints+m)*Xmaxnumber+(kcvpoints+n)*Xmaxnumber*Ymaxnumber,:]
    incFmp = np.column_stack((incFmp_iX,incFmp_iY,incFmp_iZ))
    incFmp = incFmp[:,[0,3,6,1,4,7,2,5,8]]
    incFmp = incFmp + np.array([1.,0.,0.,0.,1.,0.,0.,0.,1.])
    incFcv = np.ones((Rcv.shape[0],9))
    incFcv[:,:] = np.nan
    # indmp = np.arange(Rcv.shape[0])
    # indmp = indmp[maskmp]
    incFcv[maskmp,:] = incFmp
    Ftotal = np.column_stack((np.ones((Rcv.shape[0],1)),np.zeros((Rcv.shape[0],1)),np.zeros((Rcv.shape[0],1)),np.zeros((Rcv.shape[0],1)),np.ones((Rcv.shape[0],1)),np.zeros((Rcv.shape[0],1)),np.zeros((Rcv.shape[0],1)),np.zeros((Rcv.shape[0],1)),np.ones((Rcv.shape[0],1))))
    prev_disp = np.zeros(Rcv.shape)
    if i>=2:
        ptcvprev = pv.read(f'./New_version/output_{i-1}_clip.vtk')
        prev_disp = ptcvprev.point_data['displacement']
        Ftotal = ptcvprev.point_data['DefGrad']
    Ftotal = multDefGrads(incFcv,Ftotal)

    disp_calc[maskmp,:] = prev_disp[maskmp,:] + ump_spline
    Rmp = Rmp + ump_spline #updating Rmp
    ptcv.point_data['displacement'] = disp_calc

    # nanlocator = np.prod(~np.isnan(Ftotal),axis=1,dtype=np.bool_)
    # gv[(~maskmp)|(~nanlocator)] = 0
    # ptcv.point_data['grayscale'] = gv
    # detF = Ftotal[:,0]*(Ftotal[:,4]*Ftotal[:,8]-Ftotal[:,7]*Ftotal[:,5]) - Ftotal[:,1]*(Ftotal[:,3]*Ftotal[:,8]-Ftotal[:,6]*Ftotal[:,5]) + Ftotal[:,2]*(Ftotal[:,3]*Ftotal[:,7]-Ftotal[:,4]*Ftotal[:,6])

    # Ebrick = 0.5*(multDefGrads(Ftotal[:,[0,3,6,1,4,7,2,5,8]],Ftotal)-np.array([1.,0.,0.,0.,1.,0.,0.,0.,1.]))
    # ptcv.point_data['DefGrad'] = Ftotal
    # ptcv.point_data['DetF'] = detF
    # ptcv.point_data['Ebrick'] = Ebrick
    # ptcv.point_data['strains'] = strains
    # ptcv.save('Al_60vox_normal.vtk')
    # ptcv.save(f'./output_{i}_clip.vtk')
# %%
