# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import utils
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
from scipy.interpolate import griddata
import time

numNODES=43712
numRLN=4
numFace=87985
np.random.seed(seed=0)
times=np.random.choice(range(9500),size=32,replace=False)
numSteps=1
numT=4
sz=[500]
stepSize=sz*numSteps

numVar=3

O_SHAPE=(numNODES,numT,numVar)
#P,vx,vy,reyn
RADJ_SHAPE=(numNODES,numRLN)
#x,y,faceLen
REDG_SHAPE=(numFace,2)
RA_SHAPE=(numFace,4)


varName=['Pressure','X-Velocity','Y-Velocity']

# model_name='model20200626-104248/20200626-104248.h5'
model_name='model20200622-232331/20200622-232331.h5'
path=['']
randtS=[times[0]*10]
# path=['lowreyn']
# randtS=[50]
# path=['hole']
# randtS=[3000]

#multiStep
numStep=25


# %%
with h5py.File('meshGraph2.h5','r') as graph:
    centCoord=np.asarray(graph['centCoord'])


# %%

new_model=tf.keras.models.load_model(model_name,compile=False)


# %%
def plotting_f(varName,initF,opF,predict,numStep):
    if numStep==1:
        numStep=''
   
    for flowVar,name in enumerate(varName):
        figs,axs=plt.subplots(nrows=3,figsize=(10,15),constrained_layout=True)
        print(name+' mse:' + str(tf.keras.losses.MSE(predict[:,flowVar],opF[:,flowVar])))
        print(name+' mae:' + str(tf.keras.losses.MAE(predict[:,flowVar],opF[:,flowVar])))
        pressInit=initF['O'][:,flowVar].numpy()
        pressAft=opF[:,flowVar]
        modelG=predict[:,flowVar]
        sources=['$t$','$t+'+str(numStep)+'\delta t$','$t+'+str(numStep)+'\delta t$']
        prefix=['','Ground Truth ','Model Prediction ']
        xcomp=centCoord[:,0]
        ycomp=centCoord[:,1]
        zcomp=[pressInit,pressAft,modelG]
        diff=(pressAft-modelG)
        delta=(pressAft-pressInit)
        change=(modelG-pressInit)
        p=0
#         axes=axs[flowVar]
#         custom_xlim = (-0.2, 0.5)
#         custom_ylim = (-0.2, 0.2)
#         plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
        for ax in axs:
            ax.title.set_text(prefix[p]+name+' at time='+sources[p])
            plot=ax.tricontour(xcomp,ycomp,zcomp[p],20,cmap='jet')
            p+=1
        figs.colorbar(plot,ax=axs)
        plt.savefig('Tests/single'+name+str(numStep)+'step.svg',format='svg')



    points=np.linspace(-0.25,2,1000)
    points=np.ma.masked_inside(points,-5e-2,5e-2)
    
#     points=np.ma.masked_inside(points,-0.1,0.1)
#     points=np.ma.masked_inside(points,0.5,0.7)
    # points=[points,np.zeros_like(points)]
    fig,ax=plt.subplots(nrows=3,figsize=(10,15),constrained_layout=True)
    init=tf.squeeze(initF['O'])-opF
    predE=tf.squeeze(initF['O'])-predict
    for x in range(3):
        gt=griddata(centCoord,init[:,x],(points,np.zeros_like(points)),method='linear')
        pD=griddata(centCoord,predE[:,x],(points,np.zeros_like(points)),method='linear')
        ax[x].plot(points,gt,label='Ground Truth')
        ax[x].plot(points,pD,'--',label='Model Prediction')
        ax[x].title.set_text('Change in '+varName[x])
        ax[x].legend()
        ax[x].axvline(-5e-2,color="black", linestyle="--")
        ax[x].axvline(5e-2,color="black", linestyle="--")
#         ax[x].axvline(-0.1,color="black", linestyle="--")
#         ax[x].axvline(0.1,color="black", linestyle="--")
        
#         ax[x].axvline(0.5,color="black", linestyle="--")
#         ax[x].axvline(0.7,color="black", linestyle="--")
    plt.savefig('Tests/changeplot'+str(numStep)+'.svg',format='svg')

    fig2,ax2=plt.subplots(nrows=3,figsize=(10,15),constrained_layout=True)
    for x in range(3):
        
        ax2[x].tricontour(xcomp,ycomp,predict[:,x],cmap='jet')
#         ax2[x].title.set_text(varName[x]+' at time='+sources[1])
#         ax2[x].legend()
#         ax2[x].axvline(-5e-2,color="black", linestyle="--")
#         ax2[x].axvline(5e-2,color="black", linestyle="--")

    # points=[points,np.zeros_like(points)]
    fig3,ax3=plt.subplots(nrows=3,figsize=(10,15),constrained_layout=True)
    init=opF
    predE=predict
    for x in range(3):
        gt=griddata(centCoord,init[:,x],(points,np.zeros_like(points)),method='linear')
        pD=griddata(centCoord,predE[:,x],(points,np.zeros_like(points)),method='linear')
        ax3[x].plot(points,gt,label='Ground Truth')
        ax3[x].plot(points,pD,'--',label='Model Prediction')
        ax3[x].title.set_text(varName[x]+' at time='+sources[1])
        ax3[x].legend()
        ax3[x].axvline(-5e-2,color="black", linestyle="--")
        ax3[x].axvline(5e-2,color="black", linestyle="--")
        
#         ax3[x].axvline(-0.1,color="black", linestyle="--")
#         ax3[x].axvline(0.1,color="black", linestyle="--")

#         ax3[x].axvline(0.5,color="black", linestyle="--")
#         ax3[x].axvline(0.7,color="black", linestyle="--")
    plt.savefig('Tests/singleslice'+str(numStep)+'.svg',format='svg')


# %%
inputData=tf.data.Dataset.from_generator(
            utils.dataGen,
output_types=({'O':tf.float32,
                           'Rr':tf.int32,
                           'Rs':tf.int32,
                            'Re':tf.int32,
                             'Ra1':tf.float32,
                          'global':tf.float32},
                        {'output':tf.float32}),
             output_shapes=({'O':O_SHAPE,
                           'Rr':RADJ_SHAPE,
                             'Rs':RADJ_SHAPE,
                            'Re':REDG_SHAPE,
                             'Ra1':RA_SHAPE,
                            'global':O_SHAPE},
                         {'output':(None,numVar)}),
            args=(path,randtS,sz,numT,True,)
        )


# %%
#singlestep
def reform(prev,delta):
    return (delta*sz[0]*1e-5)+prev

for x_Ri,x_Ro in inputData:
    predict=new_model([x_Ri['O'],x_Ri['Rr'],x_Ri['Rs'],x_Ri['Re'],x_Ri['Ra1'],x_Ri['global']])
#     predict=reform(x_Ri['O'][:,-1,:],predict)
#     x_Ro['output']=reform(x_Ri['O'][:,-1,:],x_Ro['output'])
    x_Ri['O']=x_Ri['O'][:,-1,:]

#     predict=predict+tf.transpose(tf.squeeze(x_Ri['O']))
    plotting_f(varName,x_Ri,x_Ro['output'],predict,1)
    


# %%


recurData=tf.data.Dataset.from_generator(
            utils.dataGen,
output_types=({'O':tf.float32,
                           'Rr':tf.int32,
                           'Rs':tf.int32,
                            'Re':tf.int32,
                             'Ra1':tf.float32,
                          'global':tf.float32},
                        {'output':tf.float32}),
             output_shapes=({'O':O_SHAPE,
                           'Rr':RADJ_SHAPE,
                             'Rs':RADJ_SHAPE,
                            'Re':REDG_SHAPE,
                             'Ra1':RA_SHAPE,
                            'global':O_SHAPE},
                         {'output':(None,numVar)}),
            args=(path,randtS,sz,numT,True,)
        )


def modelLoop(model,inputs,outputs,numSteps):
    predict=inputs['O']
    pred_O=[]
    for steps in range(numSteps):
        carryOver=predict[:,1:,:]
#         carryOver=predict[:,:-1,:]
#         predData={'O':predict,'Rr':inputs['Rr'],'Rs':inputs['Rs'],'Re':inputs['Re'],'Ra1':inputs['Ra1'],'global':inputs['global']}
        predData=[predict,inputs['Rr'],inputs['Rs'],inputs['Re'],inputs['Ra1'],inputs['global']]
        predict=model(predData)  
        pred_O.append(predict)
#         predict=reform(prev,predict)
        predict=tf.concat([carryOver,tf.expand_dims(predict,1)],axis=1)
        
    return pred_O

def interstage(fluid,stepNum):
    with h5py.File('E:/UROP/dat/karman2d'+fluid+'.dat.h5','r') as initF:
        xVel=np.asarray(initF['results']['1']['phase-1']['faces']['SV_U']['1'])
        yVel=np.asarray(initF['results']['1']['phase-1']['faces']['SV_V']['1'])
        uInf=((xVel**2+yVel**2)**0.5).max()
    return utils.dataExtract('E:/UROP/dat/karman2d'+fluid+'-1-'+str(stepNum)+'.dat.h5',uInf=uInf)


# %%
for recurIP,recurOp in recurData:
    predict1=modelLoop(new_model,recurIP,recurOp,numStep)
#         predict=predict+tf.transpose(tf.squeeze(predData['O']))
recurIP['O']=recurIP['O'][:,-1,:]
IP=tf.squeeze(recurIP['O'])
# recurOp['output']=reform(recurIP['O'],recurOp['output'])


# %%
errors=[]
error2=[]


for x in range(numStep):
    errors.append(tf.reduce_mean(tf.keras.losses.MSE(predict1[x],interstage(path[0],randtS[0]+(x+5)*sz[0]))))
    error2.append(tf.reduce_mean(tf.keras.losses.MSE(IP,interstage(path[0],randtS[0]+(x+5)*sz[0]))))


# %%
finaldat,_=interstage(path[0],randtS[0]+(x+5)*sz[0])
plotting_f(varName,recurIP,finaldat,predict1[-1],numStep)


# %%
new_model.summary()


# %%
plt.semilogy(np.arange(len(errors)),errors)
plt.title('Mean Squared Error Accumulation')
plt.savefig('multiacc.svg',format='svg')

# %% [markdown]
# for x in range(3):
#     print(tf.keras.losses.MSE(recurIP['O'][-1,x,:],recurOp['output'][:,x]))

# %%

plt.semilogy(np.arange(len(error2)),error2,label='base')
plt.semilogy(np.arange(len(errors)),errors,label='model')
plt.legend()
plt.title('Mean Squared Error Accumulation')
plt.savefig('multiacc2.svg',format='svg')


# %%
fig,ax=plt.subplots(ncols=3,figsize=(30,10))
a=x_Ro['output']
b=predict
for x,ax in enumerate(ax):
    ax.scatter(b[:,x],a[:,x])
    xlim=np.asarray(ax.get_xlim())
    ax.plot(xlim, xlim)
#     ax.plot(xlim, xlim*1.25,'--')
#     ax.plot(xlim, xlim*.75,'--')
    ax.title.set_text(varName[x])
plt.savefig('correl2.png')


# %%
fig,ax=plt.subplots(ncols=3,figsize=(30,10))
a=tf.squeeze(x_Ri['O'])
# b=x_Ro['output']
b=finaldat
for x,ax in enumerate(ax):
    ax.scatter(b[:,x],a[:,x])
    ax.plot(ax.get_xlim(), ax.get_xlim())
    ax.title.set_text(varName[x])
plt.savefig('correl.png')


# %%
interstage(path[0],randtS[0]+(x+5)*sz[0])


# %%
z=0
for a,b in recurData:
    z+=1
    print(z)


# %%
sz


# %%
[5884, 7975,  903, 8996, 3498, 6157, 4695, 3568, 2505, 1134, 5424,
       5565, 7869, 6257, 8334, 3105, 5266, 6742, 8523,  349, 8405, 4336,
       4464,  332, 1004,  399,  209, 4377, 2700, 8337, 3335, 1469]


# %%
import numpy as np
np.random.RandomState(0)
np.random.choice(range(9500),size=32,replace=False)


# %%
import numpy as np
np.random.seed(seed=0)
np.random.choice(range(9500),size=32,replace=False)


# %%
import os
os.listdir("E:/UROP/dat/karman2d-1-5884")


# %%



