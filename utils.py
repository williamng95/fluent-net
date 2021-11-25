import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
from scipy.interpolate import griddata

numRLN=7

def dataExtract(file_path,**kwargs):
    with h5py.File(file_path,'r') as inputF:
        pressure=np.asarray(inputF['results']['1']['phase-1']['cells']['SV_P']['1'])
        xVel=np.asarray(inputF['results']['1']['phase-1']['cells']['SV_U']['1'])
        yVel=np.asarray(inputF['results']['1']['phase-1']['cells']['SV_V']['1'])
        rho=np.asarray(inputF['results']['1']['phase-1']['cells']['SV_DENSITY']['1'])
        mu=np.asarray(inputF['results']['1']['phase-1']['cells']['SV_MU_LAM']['1'])
        flux=np.asarray(inputF['results']['1']['phase-1']['faces']['SV_FLUX']['1'])
        u=(xVel**2+yVel**2)**0.5
    uInf=kwargs.get('uInf',u)
    fluidCode=kwargs.get('fluidCode',0)
    return np.column_stack((pressure/(rho*(uInf**2)),xVel/uInf,yVel/uInf)),np.column_stack((np.zeros_like(rho)+fluidCode,mu/rho,mu/rho))

def dataGen(path,timeStep,stepSize,numT,shuffle=True):
    fluidbase={'air':1,'oil':2}
    #path,cellidx,timeStep,stepSize
    params=[]
    for x in path:
        for tS in timeStep:
            for sz in stepSize:
                params.append([x,tS,sz])
    if (shuffle):
        np.random.shuffle(params)
    with h5py.File('./assets/meshGraph2.h5','r') as graph:
        Re=np.asarray(graph['edgeList'])
        Rr=np.asarray(graph['receiveList'])
        Rs=np.asarray(graph['sendList'])
        cellRln=np.asarray(graph['cellRelation'])
        
    for fluidb in params:
        fluid=fluidb[0]
        if (type(fluid)==bytes):
            fluid=fluid.decode('utf-8')
        fluidCode=fluidbase.get(fluid,0)
        with h5py.File('E:/UROP/dat/karman2d'+fluid+'.dat.h5','r') as initF:

            
            xVel=np.asarray(initF['results']['1']['phase-1']['faces']['SV_U']['1'])
            yVel=np.asarray(initF['results']['1']['phase-1']['faces']['SV_V']['1'])
            uInf=((xVel**2+yVel**2)**0.5).max()
        time=fluidb[1]
        flowP=[]
        globalP=[]
        for x in range(numT):
            pathStr='E:/UROP/dat/karman2d'+fluid+'-1-'+str(time+x*fluidb[2])+'.dat.h5'
#             pathStr='/hpctmp/e0014964/karman2d'+fluid+'/dat/karman2d'+fluid+'-1-'+str(time)+'.dat.h5'
            flowProp,flux=dataExtract(pathStr,uInf=uInf,fluidCode=fluidCode)
            flowP.append(flowProp)
            globalP.append(flux)
#             ,np.ones_like(xVel)*rho*u/mu
        flowP=np.asarray(flowP)
        flowP=np.transpose(flowP,axes=[1,0,2])
        globalP=np.asarray(globalP)
        globalP=np.transpose(globalP,axes=[1,0,2])
         

        outPred,_=dataExtract('E:/UROP/dat/karman2d'+fluid+'-1-'+str(time+fluidb[2]*(x+1))+'.dat.h5',uInf=uInf)
        ls=({'O':flowP,
           'Rr':Rr,
             'Rs':Rs,
            'Re':Re,
             'Ra1':cellRln,
            'global':globalP},
            {'output':outPred})

        yield (ls)
        

@tf.function
def modelLoop(model,inputs,outputs,numSteps):
    predict=tf.squeeze(inputs['O'])
    pred_O=[]
    for steps in range(numSteps):
        carryOver=predict[:,1:,:]
        predData=({'O':predict,'Radj':inputs['Radj'],'Re':inputs['Re'],'Ra1':inputs['Ra1']}) 
        predict=model(predData)  
        predict=tf.concat([carryOver,tf.expand_dims(predict,1)],axis=1)
        pred_O.append(predict[:,-1,:])
    return pred_O


def interstage(fluid,stepNum):
    with h5py.File('E:/UROP/dat/karman2d'+fluid+'.dat.h5','r') as initF:
        xVel=np.asarray(initF['results']['1']['phase-1']['faces']['SV_U']['1'])
        yVel=np.asarray(initF['results']['1']['phase-1']['faces']['SV_V']['1'])
        uInf=((xVel**2+yVel**2)**0.5).max()
    return dataExtract('E:/UROP/dat/karman2d'+fluid+'-1-'+str(stepNum)+'.dat.h5',uInf=uInf)
