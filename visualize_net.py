# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf

numT=4
sz=50
numNODES=43712
numRLN=7

O_SHAPE=(numNODES,numT,4)
#P,vx,vy,reyn
RA_SHAPE=(numNODES,numRLN,3)
#x,y,faceLen
RR_SHAPE=(numNODES,numRLN)


# %%
class MPNN_Class:
    
    def FCN(self,inputS,outputS,layerN,name=None):
    #a basic 3 layer fully connected module
        inputL= tf.keras.Input(shape=(inputS))
        x=tf.keras.layers.Dense(layerN,activation=tf.nn.elu)(inputL)
        x=tf.keras.layers.Dense(layerN,activation=tf.nn.elu)(x)
        x=tf.keras.layers.Dense(outputS,activation=tf.nn.elu)(x)
        return tf.keras.Model(inputs=[inputL],outputs=[x],name=name)

    def encoder(self,inputS,outputS,layerN):
        #modularised time encoder module
        inputL= tf.keras.Input(shape=(inputS))
        x=self.FCN(inputS[1],outputS,layerN)(inputL)
        x=tf.keras.layers.Conv1D(layerN,inputS[0])(x)
        return tf.keras.Model(inputs=[inputL],outputs=[x])
    
    def encoder_block(self,layerN):
        O=tf.keras.Input(shape=(O_SHAPE[1:]),name='O')
        Ra=tf.keras.Input(shape=(RA_SHAPE[1:]),name='Ra')
        #create encoders
        V_encoder=self.encoder([numT, 4],layerN,layerN)
        E_encoder=self.FCN(RA_SHAPE[1:],layerN,layerN)

        #encoded
        V=tf.squeeze(V_encoder(O),axis=1)
        E=E_encoder(Ra)

        return tf.keras.Model(inputs=[O,Ra],outputs=[V,E],name='InputEncoder')
    
    def GNCore(self,layerN,num):
        V=tf.keras.Input(shape=(layerN),name='V')
        E=tf.keras.Input(shape=(RA_SHAPE[1],layerN),name='E')
        Rr=tf.keras.Input(shape=(RR_SHAPE[1:]),name='Rr',dtype=tf.int32)

        #form Vrr
        Vrr=tf.gather(V,Rr)
        Ecat=tf.keras.layers.concatenate([E,Vrr])

        #create update functions
        phiE=self.FCN([RA_SHAPE[1],layerN*2],layerN,layerN)
        phiO=self.FCN([O_SHAPE[1],layerN*2],layerN,layerN)
        e=phiE(Ecat)
        ek=tf.reduce_sum(e,axis=1,keepdims=False)
        ek=tf.keras.layers.concatenate([V,ek])
        Vp=phiO(ek)

        return tf.keras.Model(inputs=[V,E,Rr],outputs=[Vp,e],name='GNCore'+str(num))
    
    def __init__(self, layerN=32,num_msg_pass=4,share_GNCore=False):

        O=tf.keras.Input(shape=(O_SHAPE[1:]),name='O')
        Ra=tf.keras.Input(shape=(RA_SHAPE[1:]),name='Ra')
        Rr=tf.keras.Input(shape=(RR_SHAPE[1:]),name='Rr',dtype=tf.int32)

        #create individual models
        encoder=self.encoder_block(layerN)
        decoder=self.FCN(layerN,3,layerN,'decoder')
        GNC=[]
        for x in range(num_msg_pass):
            GNC.append(self.GNCore(layerN,x))
        if (share_GNCore):
            GNC=[GNC[0]]*(x+1)
        #first, encode
        V,E=encoder([O,Ra])

        #then, process
        for x in range(num_msg_pass):
            V,E=GNC[x]([V,E,Rr])

        #finally, decode
        Vp=decoder(V)
        P=tf.keras.layers.add([O[:,-1,:-1],Vp*1e-5*sz],name='output')

        self.model= tf.keras.Model(inputs=[O,Ra,Rr],outputs=[P])


# %%
test=MPNN_Class(num_msg_pass=10,share_GNCore=True)


# %%

tf.keras.utils.plot_model(test.model,show_shapes=True,to_file='./assets/model.png')


# %%
tf.keras.utils.plot_model(test.GNCore(32,0),show_shapes=True,to_file='./assets/GNCore.png')


# %%
test.model.summary()


# %%
GNC=[]
for x in range(4):
    GNC.append(test.GNCore(32,x))


# %%
GNC


# %%
x


# %%
x=[[1,2],[2,3]]


# %%
import numpy as np
numTime=int(500)
randtS=np.random.choice(range(1,10000), numTime, replace=False)
randtS=randtS*10

splitIdx=int(numTime/10)
path=np.array(["air",'oil','lowreyn'])

params=[]
val_params=[]
for x in path:
    for timeStep in randtS:
        params.append([x,timeStep])
    


# %%
np.random.shuffle(params)


# %%
params


# %%



