??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02v2.0.0-rc2-26-g64c3d388??
?
conv2d_1/kernelVarHandleOp* 
shared_nameconv2d_1/kernel*
dtype0*
_output_shapes
: *
shape:
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:
r
conv2d_1/biasVarHandleOp*
shape:*
shared_nameconv2d_1/bias*
dtype0*
_output_shapes
: 
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:
t
dense/kernelVarHandleOp*
shape
:	 *
shared_namedense/kernel*
dtype0*
_output_shapes
: 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:	 
l

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
: 
x
dense_1/kernelVarHandleOp*
shape
:  *
shared_namedense_1/kernel*
dtype0*
_output_shapes
: 
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:  
p
dense_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
: 
x
dense_2/kernelVarHandleOp*
shared_namedense_2/kernel*
dtype0*
_output_shapes
: *
shape
: 
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes

: 
p
dense_2/biasVarHandleOp*
shared_namedense_2/bias*
dtype0*
_output_shapes
: *
shape:
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
y
dense_9/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	?*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
dtype0*
_output_shapes
:	?
q
dense_9/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:?*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
dtype0*
_output_shapes	
:?
{
dense_10/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	?* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
dtype0*
_output_shapes
:	?
r
dense_10/biasVarHandleOp*
shape:*
shared_namedense_10/bias*
dtype0*
_output_shapes
: 
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
dtype0*
_output_shapes
:

NoOpNoOp
?O
ConstConst"/device:CPU:0*?O
value?NB?N B?N
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer-14
layer-15
layer_with_weights-2
layer-16
layer-17
layer-18
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
a
	constants
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
a
&	constants
'regularization_losses
(	variables
)trainable_variables
*	keras_api
a
+	constants
,regularization_losses
-	variables
.trainable_variables
/	keras_api
a
0	constants
1regularization_losses
2	variables
3trainable_variables
4	keras_api
a
5	constants
6regularization_losses
7	variables
8trainable_variables
9	keras_api
R
:regularization_losses
;	variables
<trainable_variables
=	keras_api
R
>regularization_losses
?	variables
@trainable_variables
A	keras_api
a
B	constants
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
R
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
a
K	constants
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?
Player-0
Qlayer-1
Rlayer-2
Slayer_with_weights-0
Slayer-3
Tlayer_with_weights-1
Tlayer-4
Ulayer_with_weights-2
Ulayer-5
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
h

Zkernel
[bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
a
`	constants
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
R
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
?
ilayer-0
jlayer_with_weights-0
jlayer-1
klayer_with_weights-1
klayer-2
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
a
p	constants
qregularization_losses
r	variables
strainable_variables
t	keras_api
R
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
 
Y
y0
z1
{2
|3
}4
~5
Z6
[7
8
?9
?10
?11
Y
y0
z1
{2
|3
}4
~5
Z6
[7
8
?9
?10
?11
?
 ?layer_regularization_losses
regularization_losses
?metrics
?layers
	variables
trainable_variables
?non_trainable_variables
 
 
 
 
?
regularization_losses
 ?layer_regularization_losses
?metrics
?layers
	variables
trainable_variables
?non_trainable_variables
 
 
 
 
?
regularization_losses
 ?layer_regularization_losses
?metrics
?layers
	variables
 trainable_variables
?non_trainable_variables
 
 
 
?
"regularization_losses
 ?layer_regularization_losses
?metrics
?layers
#	variables
$trainable_variables
?non_trainable_variables
 
 
 
 
?
'regularization_losses
 ?layer_regularization_losses
?metrics
?layers
(	variables
)trainable_variables
?non_trainable_variables
 
 
 
 
?
,regularization_losses
 ?layer_regularization_losses
?metrics
?layers
-	variables
.trainable_variables
?non_trainable_variables
 
 
 
 
?
1regularization_losses
 ?layer_regularization_losses
?metrics
?layers
2	variables
3trainable_variables
?non_trainable_variables
 
 
 
 
?
6regularization_losses
 ?layer_regularization_losses
?metrics
?layers
7	variables
8trainable_variables
?non_trainable_variables
 
 
 
?
:regularization_losses
 ?layer_regularization_losses
?metrics
?layers
;	variables
<trainable_variables
?non_trainable_variables
 
 
 
?
>regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
@trainable_variables
?non_trainable_variables
 
 
 
 
?
Cregularization_losses
 ?layer_regularization_losses
?metrics
?layers
D	variables
Etrainable_variables
?non_trainable_variables
 
 
 
?
Gregularization_losses
 ?layer_regularization_losses
?metrics
?layers
H	variables
Itrainable_variables
?non_trainable_variables
 
 
 
 
?
Lregularization_losses
 ?layer_regularization_losses
?metrics
?layers
M	variables
Ntrainable_variables
?non_trainable_variables
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

ykernel
zbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

{kernel
|bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

}kernel
~bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
*
y0
z1
{2
|3
}4
~5
*
y0
z1
{2
|3
}4
~5
?
 ?layer_regularization_losses
Vregularization_losses
?metrics
?layers
W	variables
Xtrainable_variables
?non_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Z0
[1

Z0
[1
?
\regularization_losses
 ?layer_regularization_losses
?metrics
?layers
]	variables
^trainable_variables
?non_trainable_variables
 
 
 
 
?
aregularization_losses
 ?layer_regularization_losses
?metrics
?layers
b	variables
ctrainable_variables
?non_trainable_variables
 
 
 
?
eregularization_losses
 ?layer_regularization_losses
?metrics
?layers
f	variables
gtrainable_variables
?non_trainable_variables
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
m

kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 

0
?1
?2
?3

0
?1
?2
?3
?
 ?layer_regularization_losses
lregularization_losses
?metrics
?layers
m	variables
ntrainable_variables
?non_trainable_variables
 
 
 
 
?
qregularization_losses
 ?layer_regularization_losses
?metrics
?layers
r	variables
strainable_variables
?non_trainable_variables
 
 
 
?
uregularization_losses
 ?layer_regularization_losses
?metrics
?layers
v	variables
wtrainable_variables
?non_trainable_variables
HF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_9/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_9/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_10/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_10/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 

y0
z1

y0
z1
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 

{0
|1

{0
|1
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 

}0
~1

}0
~1
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 
 
*
P0
Q1
R2
S3
T4
U5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 

0
?1

0
?1
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
 
 

i0
j1
k2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0*
_output_shapes
: 
?
serving_default_ORrPlaceholder*
dtype0*/
_output_shapes
:?????????*$
shape:?????????
}
serving_default_RaPlaceholder*
dtype0*+
_output_shapes
:?????????* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_ORrserving_default_Radense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasconv2d_1/kernelconv2d_1/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias*+
_gradient_op_typePartitionedCall-3093*+
f&R$
"__inference_signature_wrapper_2125*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpConst*-
config_proto

CPU

GPU2*0J 8*
_output_shapes
: *
Tin
2*+
_gradient_op_typePartitionedCall-3127*&
f!R
__inference__traced_save_3126*
Tout
2
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias*
_output_shapes
: *
Tin
2*+
_gradient_op_typePartitionedCall-3176*)
f$R"
 __inference__traced_restore_3175*
Tout
2*-
config_proto

CPU

GPU2*0J 8??
?
?
&__inference_model_1_layer_call_fn_2049
orr
ra"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallorrrastatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-2034*J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_2033*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :	 :
 : : : :# 

_user_specified_nameORr:"

_user_specified_nameRa: : : : : : 
?
?
2__inference_TimeDerivativeModel_layer_call_fn_2940

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*+
_gradient_op_typePartitionedCall-1424*V
fQRO
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1423*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
?l
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_1818

inputs
inputs_1+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOpY
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate/concatConcatV2inputsinputs_1 concatenate/concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????	?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 ^
dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:i
dense/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:`
dense/Tensordot/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0a
dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0_
dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
_output_shapes
: *
T0]
dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0?
dense/Tensordot/transpose	Transposeconcatenate/concat:output:0dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????	?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????q
 dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense/Tensordot/transpose_1	Transpose&dense/Tensordot/ReadVariableOp:value:0)dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:	 p
dense/Tensordot/Reshape_1/shapeConst*
valueB"	       *
dtype0*
_output_shapes
:?
dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1:y:0(dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:	 ?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0"dense/Tensordot/Reshape_1:output:0*'
_output_shapes
:????????? *
T0a
dense/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB: _
dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? b
	dense/EluEludense/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  `
dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:k
dense_1/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:^
dense_1/Tensordot/ShapeShapedense/Elu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0c
!dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0a
dense_1/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Elu:activations:0!dense_1/Tensordot/concat:output:0*/
_output_shapes
:????????? *
T0?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????s
"dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_1/Tensordot/transpose_1	Transpose(dense_1/Tensordot/ReadVariableOp:value:0+dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:  r
!dense_1/Tensordot/Reshape_1/shapeConst*
valueB"        *
dtype0*
_output_shapes
:?
dense_1/Tensordot/Reshape_1Reshape!dense_1/Tensordot/transpose_1:y:0*dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:  ?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0$dense_1/Tensordot/Reshape_1:output:0*'
_output_shapes
:????????? *
T0c
dense_1/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:a
dense_1/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
dense_1/EluEludense_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: `
dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:k
dense_2/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:`
dense_2/Tensordot/ShapeShapedense_1/Elu:activations:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0c
!dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0a
dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense_2/Tensordot/transpose	Transposedense_1/Elu:activations:0!dense_2/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????s
"dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_2/Tensordot/transpose_1	Transpose(dense_2/Tensordot/ReadVariableOp:value:0+dense_2/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: r
!dense_2/Tensordot/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_2/Tensordot/Reshape_1Reshape!dense_2/Tensordot/transpose_1:y:0*dense_2/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: ?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0$dense_2/Tensordot/Reshape_1:output:0*'
_output_shapes
:?????????*
T0c
dense_2/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
IdentityIdentitydense_2/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : 
?
?
!__inference_Ra_layer_call_fn_2481
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*+
_gradient_op_typePartitionedCall-1490*E
f@R>
<__inference_Ra_layer_call_and_return_conditional_losses_1480*
Tout
2*-
config_proto

CPU

GPU2*0J 8*+
_output_shapes
:?????????*
Tin
2d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
p
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_2549
inputs_0
identityR
ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: y
ExpandDims_5
ExpandDimsinputs_0ExpandDims_5/dim:output:0*/
_output_shapes
:?????????*
T0e
IdentityIdentityExpandDims_5:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?C
?
A__inference_model_1_layer_call_and_return_conditional_losses_1994
orr
ra0
,relationmodel_statefulpartitionedcall_args_20
,relationmodel_statefulpartitionedcall_args_30
,relationmodel_statefulpartitionedcall_args_40
,relationmodel_statefulpartitionedcall_args_50
,relationmodel_statefulpartitionedcall_args_60
,relationmodel_statefulpartitionedcall_args_7+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_26
2timederivativemodel_statefulpartitionedcall_args_16
2timederivativemodel_statefulpartitionedcall_args_26
2timederivativemodel_statefulpartitionedcall_args_36
2timederivativemodel_statefulpartitionedcall_args_4
identity??%RelationModel/StatefulPartitionedCall?+TimeDerivativeModel/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
+tf_op_layer_strided_slice_1/PartitionedCallPartitionedCallorr*+
_gradient_op_typePartitionedCall-1450*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_1444*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2?
(tf_op_layer_ExpandDims_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1469*[
fVRT
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_1463*
Tout
2?
Ra_1/PartitionedCallPartitionedCallra*+
_gradient_op_typePartitionedCall-1498*E
f@R>
<__inference_Ra_layer_call_and_return_conditional_losses_1486*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:??????????
"tf_op_layer_Tile_4/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_4/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1515*U
fPRN
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_1509*
Tout
2*-
config_proto

CPU

GPU2*0J 8*+
_output_shapes
:?????????*
Tin
2?
(tf_op_layer_ExpandDims_3/PartitionedCallPartitionedCallRa_1/PartitionedCall:output:0*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1534*[
fVRT
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_1528*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
SelfLoop/PartitionedCallPartitionedCall+tf_op_layer_Tile_4/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1083*K
fFRD
B__inference_SelfLoop_layer_call_and_return_conditional_losses_1077*
Tout
2?
"tf_op_layer_Tile_3/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_3/PartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1554*U
fPRN
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_1548?
(tf_op_layer_ExpandDims_5/PartitionedCallPartitionedCall!SelfLoop/PartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1573*[
fVRT
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_1567*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
concatenate_3/PartitionedCallPartitionedCallorr+tf_op_layer_Tile_3/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1594*P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_1587*
Tout
2?
permute_1/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1101*L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_1095*
Tout
2?
"tf_op_layer_Tile_5/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_5/PartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1614*U
fPRN
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_1608?
%RelationModel/StatefulPartitionedCallStatefulPartitionedCall"permute_1/PartitionedCall:output:0+tf_op_layer_Tile_5/PartitionedCall:output:0,relationmodel_statefulpartitionedcall_args_2,relationmodel_statefulpartitionedcall_args_3,relationmodel_statefulpartitionedcall_args_4,relationmodel_statefulpartitionedcall_args_5,relationmodel_statefulpartitionedcall_args_6,relationmodel_statefulpartitionedcall_args_7*-
config_proto

CPU

GPU2*0J 8*
Tin

2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1835*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1818*
Tout
2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall.RelationModel/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1309*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1303?
%tf_op_layer_Squeeze_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1866*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_1860*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:??????????
concatenate_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0.tf_op_layer_Squeeze_1/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1887*P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_1880*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:??????????
+TimeDerivativeModel/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:02timederivativemodel_statefulpartitionedcall_args_12timederivativemodel_statefulpartitionedcall_args_22timederivativemodel_statefulpartitionedcall_args_32timederivativemodel_statefulpartitionedcall_args_4*+
_gradient_op_typePartitionedCall-1424*V
fQRO
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1423*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*'
_output_shapes
:??????????
!tf_op_layer_mul_1/PartitionedCallPartitionedCall4TimeDerivativeModel/StatefulPartitionedCall:output:0*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1929*T
fORM
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_1923*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
output/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*tf_op_layer_mul_1/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1949*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_1942*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2?
IdentityIdentityoutput/PartitionedCall:output:0&^RelationModel/StatefulPartitionedCall,^TimeDerivativeModel/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::2N
%RelationModel/StatefulPartitionedCall%RelationModel/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2Z
+TimeDerivativeModel/StatefulPartitionedCall+TimeDerivativeModel/StatefulPartitionedCall: :	 :
 : : : :# 

_user_specified_nameORr:"

_user_specified_nameRa: : : : : : 
??
?

A__inference_model_1_layer_call_and_return_conditional_losses_2419
inputs_0
inputs_19
5relationmodel_dense_tensordot_readvariableop_resource7
3relationmodel_dense_biasadd_readvariableop_resource;
7relationmodel_dense_1_tensordot_readvariableop_resource9
5relationmodel_dense_1_biasadd_readvariableop_resource;
7relationmodel_dense_2_tensordot_readvariableop_resource9
5relationmodel_dense_2_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource>
:timederivativemodel_dense_9_matmul_readvariableop_resource?
;timederivativemodel_dense_9_biasadd_readvariableop_resource?
;timederivativemodel_dense_10_matmul_readvariableop_resource@
<timederivativemodel_dense_10_biasadd_readvariableop_resource
identity??*RelationModel/dense/BiasAdd/ReadVariableOp?,RelationModel/dense/Tensordot/ReadVariableOp?,RelationModel/dense_1/BiasAdd/ReadVariableOp?.RelationModel/dense_1/Tensordot/ReadVariableOp?,RelationModel/dense_2/BiasAdd/ReadVariableOp?.RelationModel/dense_2/Tensordot/ReadVariableOp?3TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp?2TimeDerivativeModel/dense_10/MatMul/ReadVariableOp?2TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp?1TimeDerivativeModel/dense_9/MatMul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
1tf_op_layer_strided_slice_1/strided_slice_1/beginConst*
dtype0*
_output_shapes
:*%
valueB"    ????        ?
/tf_op_layer_strided_slice_1/strided_slice_1/endConst*
dtype0*
_output_shapes
:*%
valueB"               ?
3tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*%
valueB"            *
dtype0*
_output_shapes
:?
+tf_op_layer_strided_slice_1/strided_slice_1StridedSliceinputs_0:tf_op_layer_strided_slice_1/strided_slice_1/begin:output:08tf_op_layer_strided_slice_1/strided_slice_1/end:output:0<tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*
Index0*
T0*
shrink_axis_mask
*

begin_mask*
end_mask*'
_output_shapes
:?????????k
)tf_op_layer_ExpandDims_4/ExpandDims_4/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
%tf_op_layer_ExpandDims_4/ExpandDims_4
ExpandDims4tf_op_layer_strided_slice_1/strided_slice_1:output:02tf_op_layer_ExpandDims_4/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:?????????x
#tf_op_layer_Tile_4/Tile_4/multiplesConst*!
valueB"         *
dtype0*
_output_shapes
:?
tf_op_layer_Tile_4/Tile_4Tile.tf_op_layer_ExpandDims_4/ExpandDims_4:output:0,tf_op_layer_Tile_4/Tile_4/multiples:output:0*+
_output_shapes
:?????????*
T0k
)tf_op_layer_ExpandDims_3/ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
%tf_op_layer_ExpandDims_3/ExpandDims_3
ExpandDimsinputs_12tf_op_layer_ExpandDims_3/ExpandDims_3/dim:output:0*/
_output_shapes
:?????????*
T0l
SelfLoop/transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          ?
SelfLoop/transpose	Transpose"tf_op_layer_Tile_4/Tile_4:output:0 SelfLoop/transpose/perm:output:0*
T0*+
_output_shapes
:?????????|
#tf_op_layer_Tile_3/Tile_3/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:?
tf_op_layer_Tile_3/Tile_3Tile.tf_op_layer_ExpandDims_3/ExpandDims_3:output:0,tf_op_layer_Tile_3/Tile_3/multiples:output:0*/
_output_shapes
:?????????*
T0k
)tf_op_layer_ExpandDims_5/ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
%tf_op_layer_ExpandDims_5/ExpandDims_5
ExpandDimsSelfLoop/transpose:y:02tf_op_layer_ExpandDims_5/ExpandDims_5/dim:output:0*/
_output_shapes
:?????????*
T0[
concatenate_3/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate_3/concatConcatV2inputs_0"tf_op_layer_Tile_3/Tile_3:output:0"concatenate_3/concat/axis:output:0*
N*/
_output_shapes
:?????????*
T0q
permute_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:?
permute_1/transpose	Transposeconcatenate_3/concat:output:0!permute_1/transpose/perm:output:0*/
_output_shapes
:?????????*
T0|
#tf_op_layer_Tile_5/Tile_5/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:?
tf_op_layer_Tile_5/Tile_5Tile.tf_op_layer_ExpandDims_5/ExpandDims_5:output:0,tf_op_layer_Tile_5/Tile_5/multiples:output:0*
T0*/
_output_shapes
:?????????g
%RelationModel/concatenate/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :?
 RelationModel/concatenate/concatConcatV2permute_1/transpose:y:0"tf_op_layer_Tile_5/Tile_5:output:0.RelationModel/concatenate/concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????	?
,RelationModel/dense/Tensordot/ReadVariableOpReadVariableOp5relationmodel_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 l
"RelationModel/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:w
"RelationModel/dense/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          |
#RelationModel/dense/Tensordot/ShapeShape)RelationModel/concatenate/concat:output:0*
T0*
_output_shapes
:m
+RelationModel/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
&RelationModel/dense/Tensordot/GatherV2GatherV2,RelationModel/dense/Tensordot/Shape:output:0+RelationModel/dense/Tensordot/free:output:04RelationModel/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-RelationModel/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
(RelationModel/dense/Tensordot/GatherV2_1GatherV2,RelationModel/dense/Tensordot/Shape:output:0+RelationModel/dense/Tensordot/axes:output:06RelationModel/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#RelationModel/dense/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: ?
"RelationModel/dense/Tensordot/ProdProd/RelationModel/dense/Tensordot/GatherV2:output:0,RelationModel/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%RelationModel/dense/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: ?
$RelationModel/dense/Tensordot/Prod_1Prod1RelationModel/dense/Tensordot/GatherV2_1:output:0.RelationModel/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)RelationModel/dense/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
$RelationModel/dense/Tensordot/concatConcatV2+RelationModel/dense/Tensordot/free:output:0+RelationModel/dense/Tensordot/axes:output:02RelationModel/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
#RelationModel/dense/Tensordot/stackPack+RelationModel/dense/Tensordot/Prod:output:0-RelationModel/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
'RelationModel/dense/Tensordot/transpose	Transpose)RelationModel/concatenate/concat:output:0-RelationModel/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????	?
%RelationModel/dense/Tensordot/ReshapeReshape+RelationModel/dense/Tensordot/transpose:y:0,RelationModel/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????
.RelationModel/dense/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       ?
)RelationModel/dense/Tensordot/transpose_1	Transpose4RelationModel/dense/Tensordot/ReadVariableOp:value:07RelationModel/dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:	 ~
-RelationModel/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"	       ?
'RelationModel/dense/Tensordot/Reshape_1Reshape-RelationModel/dense/Tensordot/transpose_1:y:06RelationModel/dense/Tensordot/Reshape_1/shape:output:0*
_output_shapes

:	 *
T0?
$RelationModel/dense/Tensordot/MatMulMatMul.RelationModel/dense/Tensordot/Reshape:output:00RelationModel/dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? o
%RelationModel/dense/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB: m
+RelationModel/dense/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
&RelationModel/dense/Tensordot/concat_1ConcatV2/RelationModel/dense/Tensordot/GatherV2:output:0.RelationModel/dense/Tensordot/Const_2:output:04RelationModel/dense/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0?
RelationModel/dense/TensordotReshape.RelationModel/dense/Tensordot/MatMul:product:0/RelationModel/dense/Tensordot/concat_1:output:0*/
_output_shapes
:????????? *
T0?
*RelationModel/dense/BiasAdd/ReadVariableOpReadVariableOp3relationmodel_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
RelationModel/dense/BiasAddBiasAdd&RelationModel/dense/Tensordot:output:02RelationModel/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
RelationModel/dense/EluElu$RelationModel/dense/BiasAdd:output:0*/
_output_shapes
:????????? *
T0?
.RelationModel/dense_1/Tensordot/ReadVariableOpReadVariableOp7relationmodel_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  n
$RelationModel/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:y
$RelationModel/dense_1/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:z
%RelationModel/dense_1/Tensordot/ShapeShape%RelationModel/dense/Elu:activations:0*
T0*
_output_shapes
:o
-RelationModel/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
(RelationModel/dense_1/Tensordot/GatherV2GatherV2.RelationModel/dense_1/Tensordot/Shape:output:0-RelationModel/dense_1/Tensordot/free:output:06RelationModel/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/RelationModel/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
*RelationModel/dense_1/Tensordot/GatherV2_1GatherV2.RelationModel/dense_1/Tensordot/Shape:output:0-RelationModel/dense_1/Tensordot/axes:output:08RelationModel/dense_1/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0o
%RelationModel/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
$RelationModel/dense_1/Tensordot/ProdProd1RelationModel/dense_1/Tensordot/GatherV2:output:0.RelationModel/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'RelationModel/dense_1/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: ?
&RelationModel/dense_1/Tensordot/Prod_1Prod3RelationModel/dense_1/Tensordot/GatherV2_1:output:00RelationModel/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+RelationModel/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
&RelationModel/dense_1/Tensordot/concatConcatV2-RelationModel/dense_1/Tensordot/free:output:0-RelationModel/dense_1/Tensordot/axes:output:04RelationModel/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
%RelationModel/dense_1/Tensordot/stackPack-RelationModel/dense_1/Tensordot/Prod:output:0/RelationModel/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
)RelationModel/dense_1/Tensordot/transpose	Transpose%RelationModel/dense/Elu:activations:0/RelationModel/dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
'RelationModel/dense_1/Tensordot/ReshapeReshape-RelationModel/dense_1/Tensordot/transpose:y:0.RelationModel/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0RelationModel/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
+RelationModel/dense_1/Tensordot/transpose_1	Transpose6RelationModel/dense_1/Tensordot/ReadVariableOp:value:09RelationModel/dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:  ?
/RelationModel/dense_1/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"        ?
)RelationModel/dense_1/Tensordot/Reshape_1Reshape/RelationModel/dense_1/Tensordot/transpose_1:y:08RelationModel/dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:  ?
&RelationModel/dense_1/Tensordot/MatMulMatMul0RelationModel/dense_1/Tensordot/Reshape:output:02RelationModel/dense_1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? q
'RelationModel/dense_1/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:o
-RelationModel/dense_1/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
(RelationModel/dense_1/Tensordot/concat_1ConcatV21RelationModel/dense_1/Tensordot/GatherV2:output:00RelationModel/dense_1/Tensordot/Const_2:output:06RelationModel/dense_1/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0?
RelationModel/dense_1/TensordotReshape0RelationModel/dense_1/Tensordot/MatMul:product:01RelationModel/dense_1/Tensordot/concat_1:output:0*/
_output_shapes
:????????? *
T0?
,RelationModel/dense_1/BiasAdd/ReadVariableOpReadVariableOp5relationmodel_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
RelationModel/dense_1/BiasAddBiasAdd(RelationModel/dense_1/Tensordot:output:04RelationModel/dense_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? *
T0?
RelationModel/dense_1/EluElu&RelationModel/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
.RelationModel/dense_2/Tensordot/ReadVariableOpReadVariableOp7relationmodel_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: n
$RelationModel/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:y
$RelationModel/dense_2/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:|
%RelationModel/dense_2/Tensordot/ShapeShape'RelationModel/dense_1/Elu:activations:0*
T0*
_output_shapes
:o
-RelationModel/dense_2/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
(RelationModel/dense_2/Tensordot/GatherV2GatherV2.RelationModel/dense_2/Tensordot/Shape:output:0-RelationModel/dense_2/Tensordot/free:output:06RelationModel/dense_2/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0q
/RelationModel/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
*RelationModel/dense_2/Tensordot/GatherV2_1GatherV2.RelationModel/dense_2/Tensordot/Shape:output:0-RelationModel/dense_2/Tensordot/axes:output:08RelationModel/dense_2/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0o
%RelationModel/dense_2/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: ?
$RelationModel/dense_2/Tensordot/ProdProd1RelationModel/dense_2/Tensordot/GatherV2:output:0.RelationModel/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'RelationModel/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
&RelationModel/dense_2/Tensordot/Prod_1Prod3RelationModel/dense_2/Tensordot/GatherV2_1:output:00RelationModel/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+RelationModel/dense_2/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
&RelationModel/dense_2/Tensordot/concatConcatV2-RelationModel/dense_2/Tensordot/free:output:0-RelationModel/dense_2/Tensordot/axes:output:04RelationModel/dense_2/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0?
%RelationModel/dense_2/Tensordot/stackPack-RelationModel/dense_2/Tensordot/Prod:output:0/RelationModel/dense_2/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0?
)RelationModel/dense_2/Tensordot/transpose	Transpose'RelationModel/dense_1/Elu:activations:0/RelationModel/dense_2/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
'RelationModel/dense_2/Tensordot/ReshapeReshape-RelationModel/dense_2/Tensordot/transpose:y:0.RelationModel/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0RelationModel/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
+RelationModel/dense_2/Tensordot/transpose_1	Transpose6RelationModel/dense_2/Tensordot/ReadVariableOp:value:09RelationModel/dense_2/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: ?
/RelationModel/dense_2/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"       ?
)RelationModel/dense_2/Tensordot/Reshape_1Reshape/RelationModel/dense_2/Tensordot/transpose_1:y:08RelationModel/dense_2/Tensordot/Reshape_1/shape:output:0*
_output_shapes

: *
T0?
&RelationModel/dense_2/Tensordot/MatMulMatMul0RelationModel/dense_2/Tensordot/Reshape:output:02RelationModel/dense_2/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????q
'RelationModel/dense_2/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:o
-RelationModel/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
(RelationModel/dense_2/Tensordot/concat_1ConcatV21RelationModel/dense_2/Tensordot/GatherV2:output:00RelationModel/dense_2/Tensordot/Const_2:output:06RelationModel/dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
RelationModel/dense_2/TensordotReshape0RelationModel/dense_2/Tensordot/MatMul:product:01RelationModel/dense_2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:??????????
,RelationModel/dense_2/BiasAdd/ReadVariableOpReadVariableOp5relationmodel_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
RelationModel/dense_2/BiasAddBiasAdd(RelationModel/dense_2/Tensordot:output:04RelationModel/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2d_1/Conv2DConv2D&RelationModel/dense_2/BiasAdd:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*/
_output_shapes
:?????????*
T0?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
tf_op_layer_Squeeze_1/Squeeze_1Squeezeconv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
[
concatenate_4/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :?
concatenate_4/concatConcatV24tf_op_layer_strided_slice_1/strided_slice_1:output:0(tf_op_layer_Squeeze_1/Squeeze_1:output:0"concatenate_4/concat/axis:output:0*
T0*
N*'
_output_shapes
:??????????
1TimeDerivativeModel/dense_9/MatMul/ReadVariableOpReadVariableOp:timederivativemodel_dense_9_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	??
"TimeDerivativeModel/dense_9/MatMulMatMulconcatenate_4/concat:output:09TimeDerivativeModel/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOpReadVariableOp;timederivativemodel_dense_9_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:??
#TimeDerivativeModel/dense_9/BiasAddBiasAdd,TimeDerivativeModel/dense_9/MatMul:product:0:TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
TimeDerivativeModel/dense_9/EluElu,TimeDerivativeModel/dense_9/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
2TimeDerivativeModel/dense_10/MatMul/ReadVariableOpReadVariableOp;timederivativemodel_dense_10_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	??
#TimeDerivativeModel/dense_10/MatMulMatMul-TimeDerivativeModel/dense_9/Elu:activations:0:TimeDerivativeModel/dense_10/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
3TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOpReadVariableOp<timederivativemodel_dense_10_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
$TimeDerivativeModel/dense_10/BiasAddBiasAdd-TimeDerivativeModel/dense_10/MatMul:product:0;TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0^
tf_op_layer_mul_1/mul_1/yConst*
valueB
 *??8*
dtype0*
_output_shapes
: ?
tf_op_layer_mul_1/mul_1Mul-TimeDerivativeModel/dense_10/BiasAdd:output:0"tf_op_layer_mul_1/mul_1/y:output:0*
T0*'
_output_shapes
:??????????

output/addAddV24tf_op_layer_strided_slice_1/strided_slice_1:output:0tf_op_layer_mul_1/mul_1:z:0*
T0*'
_output_shapes
:??????????
IdentityIdentityoutput/add:z:0+^RelationModel/dense/BiasAdd/ReadVariableOp-^RelationModel/dense/Tensordot/ReadVariableOp-^RelationModel/dense_1/BiasAdd/ReadVariableOp/^RelationModel/dense_1/Tensordot/ReadVariableOp-^RelationModel/dense_2/BiasAdd/ReadVariableOp/^RelationModel/dense_2/Tensordot/ReadVariableOp4^TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp3^TimeDerivativeModel/dense_10/MatMul/ReadVariableOp3^TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp2^TimeDerivativeModel/dense_9/MatMul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2`
.RelationModel/dense_1/Tensordot/ReadVariableOp.RelationModel/dense_1/Tensordot/ReadVariableOp2\
,RelationModel/dense_1/BiasAdd/ReadVariableOp,RelationModel/dense_1/BiasAdd/ReadVariableOp2`
.RelationModel/dense_2/Tensordot/ReadVariableOp.RelationModel/dense_2/Tensordot/ReadVariableOp2h
2TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp2TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp2h
2TimeDerivativeModel/dense_10/MatMul/ReadVariableOp2TimeDerivativeModel/dense_10/MatMul/ReadVariableOp2\
,RelationModel/dense/Tensordot/ReadVariableOp,RelationModel/dense/Tensordot/ReadVariableOp2f
1TimeDerivativeModel/dense_9/MatMul/ReadVariableOp1TimeDerivativeModel/dense_9/MatMul/ReadVariableOp2X
*RelationModel/dense/BiasAdd/ReadVariableOp*RelationModel/dense/BiasAdd/ReadVariableOp2\
,RelationModel/dense_2/BiasAdd/ReadVariableOp,RelationModel/dense_2/BiasAdd/ReadVariableOp2j
3TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp3TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp:
 : : : :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : : :	 
?
?
&__inference_model_1_layer_call_fn_2437
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*+
_gradient_op_typePartitionedCall-2034*J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_2033*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : : :	 :
 : : : 
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_2987

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:????????? *
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? N
EluEluBiasAdd:output:0*'
_output_shapes
:????????? *
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*.
_input_shapes
:?????????	::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
&__inference_dense_1_layer_call_fn_3012

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1177*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1171*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
Z
<__inference_Ra_layer_call_and_return_conditional_losses_2476
inputs_0
identityT
IdentityIdentityinputs_0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?

?
,__inference_RelationModel_layer_call_fn_2775
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin

2*+
_gradient_op_typePartitionedCall-1822*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1719?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : 
?
?
!__inference_Ra_layer_call_fn_2486
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*+
_gradient_op_typePartitionedCall-1498*E
f@R>
<__inference_Ra_layer_call_and_return_conditional_losses_1486*
Tout
2*-
config_proto

CPU

GPU2*0J 8*+
_output_shapes
:?????????*
Tin
2d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
X
<__inference_Ra_layer_call_and_return_conditional_losses_1480

inputs
identityR
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
,__inference_RelationModel_layer_call_fn_2865
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*-
config_proto

CPU

GPU2*0J 8*
Tin

2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1283*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1282*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : 
?
?
"__inference_signature_wrapper_2125
orr
ra"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallorrrastatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-2110*(
f#R!
__inference__wrapped_model_1068*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:# 

_user_specified_nameORr:"

_user_specified_nameRa: : : : : : : :	 :
 : : : 
?

?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1303

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
B__inference_dense_10_layer_call_and_return_conditional_losses_1358

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	?i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
R
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_2875
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1866*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_1860*
Tout
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1423

inputs*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2+
'dense_10_statefulpartitionedcall_args_1+
'dense_10_statefulpartitionedcall_args_2
identity?? dense_10/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*
Tin
2*+
_gradient_op_typePartitionedCall-1337*J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_1331*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0'dense_10_statefulpartitionedcall_args_1'dense_10_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1364*K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_1358*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
?
Z
<__inference_Ra_layer_call_and_return_conditional_losses_2472
inputs_0
identityT
IdentityIdentityinputs_0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
X
<__inference_Ra_layer_call_and_return_conditional_losses_1486

inputs
identityR
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
h
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_1509

inputs
identitye
Tile_4/multiplesConst*
dtype0*
_output_shapes
:*!
valueB"         g
Tile_4TileinputsTile_4/multiples:output:0*
T0*+
_output_shapes
:?????????[
IdentityIdentityTile_4:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
s
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_2463
inputs_0
identityn
strided_slice_1/beginConst*%
valueB"    ????        *
dtype0*
_output_shapes
:l
strided_slice_1/endConst*
dtype0*
_output_shapes
:*%
valueB"               p
strided_slice_1/stridesConst*%
valueB"            *
dtype0*
_output_shapes
:?
strided_slice_1StridedSliceinputs_0strided_slice_1/begin:output:0strided_slice_1/end:output:0 strided_slice_1/strides:output:0*
Index0*
T0*
shrink_axis_mask
*

begin_mask*
end_mask*'
_output_shapes
:?????????`
IdentityIdentitystrided_slice_1:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
g
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_1923

inputs
identityL
mul_1/yConst*
valueB
 *??8*
dtype0*
_output_shapes
: X
mul_1Mulinputsmul_1/y:output:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1376
oeb*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2+
'dense_10_statefulpartitionedcall_args_1+
'dense_10_statefulpartitionedcall_args_2
identity?? dense_10/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCalloeb&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:??????????*
Tin
2*+
_gradient_op_typePartitionedCall-1337*J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_1331*
Tout
2?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0'dense_10_statefulpartitionedcall_args_1'dense_10_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1364*K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_1358*
Tout
2?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:# 

_user_specified_nameOEb: : : : 
?
k
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_1860

inputs
identityf
	Squeeze_1Squeezeinputs*
squeeze_dims
*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_1282

inputs
inputs_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????	*
Tin
2*+
_gradient_op_typePartitionedCall-1125*N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1118?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:????????? *+
_gradient_op_typePartitionedCall-1149*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1143*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? *+
_gradient_op_typePartitionedCall-1177*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1171*
Tout
2?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1204*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1198*
Tout
2?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : : : : : :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
D
(__inference_permute_1_layer_call_fn_1104

inputs
identity?
PartitionedCallPartitionedCallinputs*L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_1095*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*J
_output_shapes8
6:4????????????????????????????????????*+
_gradient_op_typePartitionedCall-1101?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_1252

inputs
inputs_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCallinputsinputs_1*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????	*+
_gradient_op_typePartitionedCall-1125*N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1118*
Tout
2?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1149*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1143*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? ?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:????????? *+
_gradient_op_typePartitionedCall-1177*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1171*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1204*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1198*
Tout
2?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : : : : : 
?
?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_2905

inputs*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	?z
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:??
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????_
dense_9/EluEludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	??
dense_10/MatMulMatMuldense_9/Elu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
?
X
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_2468
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*+
_gradient_op_typePartitionedCall-1450*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_1444*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
j
@__inference_output_layer_call_and_return_conditional_losses_1942

inputs
inputs_1
identityP
addAddV2inputsinputs_1*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
U
7__inference_tf_op_layer_ExpandDims_3_layer_call_fn_2508
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*+
_gradient_op_typePartitionedCall-1534*[
fVRT
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_1528*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
O
1__inference_tf_op_layer_Tile_5_layer_call_fn_2565
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*+
_gradient_op_typePartitionedCall-1614*U
fPRN
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_1608*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1401

inputs*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2+
'dense_10_statefulpartitionedcall_args_1+
'dense_10_statefulpartitionedcall_args_2
identity?? dense_10/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1337*J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_1331*
Tout
2*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:??????????*
Tin
2?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0'dense_10_statefulpartitionedcall_args_1'dense_10_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1364*K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_1358*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
?
A__inference_dense_2_layer_call_and_return_conditional_losses_3022

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
n
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_1567

inputs
identityR
ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: w
ExpandDims_5
ExpandDimsinputsExpandDims_5/dim:output:0*
T0*/
_output_shapes
:?????????e
IdentityIdentityExpandDims_5:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?C
?
A__inference_model_1_layer_call_and_return_conditional_losses_2033

inputs
inputs_10
,relationmodel_statefulpartitionedcall_args_20
,relationmodel_statefulpartitionedcall_args_30
,relationmodel_statefulpartitionedcall_args_40
,relationmodel_statefulpartitionedcall_args_50
,relationmodel_statefulpartitionedcall_args_60
,relationmodel_statefulpartitionedcall_args_7+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_26
2timederivativemodel_statefulpartitionedcall_args_16
2timederivativemodel_statefulpartitionedcall_args_26
2timederivativemodel_statefulpartitionedcall_args_36
2timederivativemodel_statefulpartitionedcall_args_4
identity??%RelationModel/StatefulPartitionedCall?+TimeDerivativeModel/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
+tf_op_layer_strided_slice_1/PartitionedCallPartitionedCallinputs*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1450*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_1444*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
(tf_op_layer_ExpandDims_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*[
fVRT
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_1463*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1469?
Ra/PartitionedCallPartitionedCallinputs_1*+
_gradient_op_typePartitionedCall-1490*E
f@R>
<__inference_Ra_layer_call_and_return_conditional_losses_1480*
Tout
2*-
config_proto

CPU

GPU2*0J 8*+
_output_shapes
:?????????*
Tin
2?
"tf_op_layer_Tile_4/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_4/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*+
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1515*U
fPRN
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_1509*
Tout
2?
(tf_op_layer_ExpandDims_3/PartitionedCallPartitionedCallRa/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1534*[
fVRT
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_1528*
Tout
2?
SelfLoop/PartitionedCallPartitionedCall+tf_op_layer_Tile_4/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1083*K
fFRD
B__inference_SelfLoop_layer_call_and_return_conditional_losses_1077*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:??????????
"tf_op_layer_Tile_3/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_3/PartitionedCall:output:0*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1554*U
fPRN
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_1548*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
(tf_op_layer_ExpandDims_5/PartitionedCallPartitionedCall!SelfLoop/PartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1573*[
fVRT
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_1567?
concatenate_3/PartitionedCallPartitionedCallinputs+tf_op_layer_Tile_3/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1594*P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_1587*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
permute_1/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1101*L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_1095*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
"tf_op_layer_Tile_5/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_5/PartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1614*U
fPRN
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_1608*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
%RelationModel/StatefulPartitionedCallStatefulPartitionedCall"permute_1/PartitionedCall:output:0+tf_op_layer_Tile_5/PartitionedCall:output:0,relationmodel_statefulpartitionedcall_args_2,relationmodel_statefulpartitionedcall_args_3,relationmodel_statefulpartitionedcall_args_4,relationmodel_statefulpartitionedcall_args_5,relationmodel_statefulpartitionedcall_args_6,relationmodel_statefulpartitionedcall_args_7*+
_gradient_op_typePartitionedCall-1822*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1719*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin

2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall.RelationModel/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1309*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1303*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
%tf_op_layer_Squeeze_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1866*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_1860*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2?
concatenate_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0.tf_op_layer_Squeeze_1/PartitionedCall:output:0*P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_1880*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1887?
+TimeDerivativeModel/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:02timederivativemodel_statefulpartitionedcall_args_12timederivativemodel_statefulpartitionedcall_args_22timederivativemodel_statefulpartitionedcall_args_32timederivativemodel_statefulpartitionedcall_args_4*+
_gradient_op_typePartitionedCall-1402*V
fQRO
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1401*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*'
_output_shapes
:??????????
!tf_op_layer_mul_1/PartitionedCallPartitionedCall4TimeDerivativeModel/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1929*T
fORM
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_1923*
Tout
2?
output/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*tf_op_layer_mul_1/PartitionedCall:output:0*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1949*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_1942*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentityoutput/PartitionedCall:output:0&^RelationModel/StatefulPartitionedCall,^TimeDerivativeModel/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2Z
+TimeDerivativeModel/StatefulPartitionedCall+TimeDerivativeModel/StatefulPartitionedCall2N
%RelationModel/StatefulPartitionedCall%RelationModel/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : : : : : : :	 :
 : : : 
?
?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1388
oeb*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2+
'dense_10_statefulpartitionedcall_args_1+
'dense_10_statefulpartitionedcall_args_2
identity?? dense_10/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCalloeb&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1337*J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_1331*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*(
_output_shapes
:???????????
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0'dense_10_statefulpartitionedcall_args_1'dense_10_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1364*K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_1358*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall: :# 

_user_specified_nameOEb: : : 
?	
?
A__inference_dense_9_layer_call_and_return_conditional_losses_3040

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	?j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
&__inference_model_1_layer_call_fn_2455
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_2089*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-2090?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : : :	 :
 : 
?
O
1__inference_tf_op_layer_Tile_4_layer_call_fn_2519
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*U
fPRN
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_1509*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1515d
IdentityIdentityPartitionedCall:output:0*+
_output_shapes
:?????????*
T0"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
h
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_1548

inputs
identityi
Tile_3/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:k
Tile_3TileinputsTile_3/multiples:output:0*
T0*/
_output_shapes
:?????????_
IdentityIdentityTile_3:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?C
?
A__inference_model_1_layer_call_and_return_conditional_losses_1957
orr
ra0
,relationmodel_statefulpartitionedcall_args_20
,relationmodel_statefulpartitionedcall_args_30
,relationmodel_statefulpartitionedcall_args_40
,relationmodel_statefulpartitionedcall_args_50
,relationmodel_statefulpartitionedcall_args_60
,relationmodel_statefulpartitionedcall_args_7+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_26
2timederivativemodel_statefulpartitionedcall_args_16
2timederivativemodel_statefulpartitionedcall_args_26
2timederivativemodel_statefulpartitionedcall_args_36
2timederivativemodel_statefulpartitionedcall_args_4
identity??%RelationModel/StatefulPartitionedCall?+TimeDerivativeModel/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
+tf_op_layer_strided_slice_1/PartitionedCallPartitionedCallorr*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1450*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_1444*
Tout
2?
(tf_op_layer_ExpandDims_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*[
fVRT
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_1463*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1469?
Ra_1/PartitionedCallPartitionedCallra*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1490*E
f@R>
<__inference_Ra_layer_call_and_return_conditional_losses_1480*
Tout
2?
"tf_op_layer_Tile_4/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_4/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1515*U
fPRN
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_1509*
Tout
2*-
config_proto

CPU

GPU2*0J 8*+
_output_shapes
:?????????*
Tin
2?
(tf_op_layer_ExpandDims_3/PartitionedCallPartitionedCallRa_1/PartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1534*[
fVRT
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_1528?
SelfLoop/PartitionedCallPartitionedCall+tf_op_layer_Tile_4/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1083*K
fFRD
B__inference_SelfLoop_layer_call_and_return_conditional_losses_1077*
Tout
2?
"tf_op_layer_Tile_3/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_3/PartitionedCall:output:0*U
fPRN
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_1548*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1554?
(tf_op_layer_ExpandDims_5/PartitionedCallPartitionedCall!SelfLoop/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1573*[
fVRT
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_1567*
Tout
2?
concatenate_3/PartitionedCallPartitionedCallorr+tf_op_layer_Tile_3/PartitionedCall:output:0*P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_1587*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1594?
permute_1/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_1095*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1101?
"tf_op_layer_Tile_5/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_5/PartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1614*U
fPRN
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_1608*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
%RelationModel/StatefulPartitionedCallStatefulPartitionedCall"permute_1/PartitionedCall:output:0+tf_op_layer_Tile_5/PartitionedCall:output:0,relationmodel_statefulpartitionedcall_args_2,relationmodel_statefulpartitionedcall_args_3,relationmodel_statefulpartitionedcall_args_4,relationmodel_statefulpartitionedcall_args_5,relationmodel_statefulpartitionedcall_args_6,relationmodel_statefulpartitionedcall_args_7*+
_gradient_op_typePartitionedCall-1822*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1719*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin

2*/
_output_shapes
:??????????
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall.RelationModel/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1309*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1303*
Tout
2?
%tf_op_layer_Squeeze_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_1860*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1866?
concatenate_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0.tf_op_layer_Squeeze_1/PartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1887*P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_1880?
+TimeDerivativeModel/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:02timederivativemodel_statefulpartitionedcall_args_12timederivativemodel_statefulpartitionedcall_args_22timederivativemodel_statefulpartitionedcall_args_32timederivativemodel_statefulpartitionedcall_args_4*+
_gradient_op_typePartitionedCall-1402*V
fQRO
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1401*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*'
_output_shapes
:??????????
!tf_op_layer_mul_1/PartitionedCallPartitionedCall4TimeDerivativeModel/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1929*T
fORM
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_1923*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:??????????
output/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*tf_op_layer_mul_1/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1949*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_1942*
Tout
2?
IdentityIdentityoutput/PartitionedCall:output:0&^RelationModel/StatefulPartitionedCall,^TimeDerivativeModel/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::2N
%RelationModel/StatefulPartitionedCall%RelationModel/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2Z
+TimeDerivativeModel/StatefulPartitionedCall+TimeDerivativeModel/StatefulPartitionedCall: : : : : :	 :
 : : : :# 

_user_specified_nameORr:"

_user_specified_nameRa: : 
?
?
&__inference_model_1_layer_call_fn_2105
orr
ra"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallorrrastatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-2090*J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_2089*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:# 

_user_specified_nameORr:"

_user_specified_nameRa: : : : : : : :	 :
 : : : 
?l
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_2664
inputs_0
inputs_1+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOpY
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????	?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 ^
dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:i
dense/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:`
dense/Tensordot/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0a
dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0_
dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense/Tensordot/transpose	Transposeconcatenate/concat:output:0dense/Tensordot/concat:output:0*/
_output_shapes
:?????????	*
T0?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????q
 dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense/Tensordot/transpose_1	Transpose&dense/Tensordot/ReadVariableOp:value:0)dense/Tensordot/transpose_1/perm:output:0*
_output_shapes

:	 *
T0p
dense/Tensordot/Reshape_1/shapeConst*
valueB"	       *
dtype0*
_output_shapes
:?
dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1:y:0(dense/Tensordot/Reshape_1/shape:output:0*
_output_shapes

:	 *
T0?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0"dense/Tensordot/Reshape_1:output:0*'
_output_shapes
:????????? *
T0a
dense/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:_
dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? b
	dense/EluEludense/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  `
dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:k
dense_1/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          ^
dense_1/Tensordot/ShapeShapedense/Elu:activations:0*
_output_shapes
:*
T0a
dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0c
!dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0a
dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0?
dense_1/Tensordot/transpose	Transposedense/Elu:activations:0!dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*0
_output_shapes
:??????????????????*
T0s
"dense_1/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense_1/Tensordot/transpose_1	Transpose(dense_1/Tensordot/ReadVariableOp:value:0+dense_1/Tensordot/transpose_1/perm:output:0*
_output_shapes

:  *
T0r
!dense_1/Tensordot/Reshape_1/shapeConst*
valueB"        *
dtype0*
_output_shapes
:?
dense_1/Tensordot/Reshape_1Reshape!dense_1/Tensordot/transpose_1:y:0*dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:  ?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0$dense_1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:a
dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
dense_1/EluEludense_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: `
dense_2/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:k
dense_2/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          `
dense_2/Tensordot/ShapeShapedense_1/Elu:activations:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0a
dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense_2/Tensordot/transpose	Transposedense_1/Elu:activations:0!dense_2/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????s
"dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_2/Tensordot/transpose_1	Transpose(dense_2/Tensordot/ReadVariableOp:value:0+dense_2/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: r
!dense_2/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense_2/Tensordot/Reshape_1Reshape!dense_2/Tensordot/transpose_1:y:0*dense_2/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: ?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0$dense_2/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????c
dense_2/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:a
dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
IdentityIdentitydense_2/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : 
?
?
A__inference_dense_2_layer_call_and_return_conditional_losses_1198

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?l
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_2763
inputs_0
inputs_1+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOpY
concatenate/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :?
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????	?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 ^
dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:i
dense/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:`
dense/Tensordot/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0a
dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense/Tensordot/transpose	Transposeconcatenate/concat:output:0dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????	?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????q
 dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense/Tensordot/transpose_1	Transpose&dense/Tensordot/ReadVariableOp:value:0)dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:	 p
dense/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"	       ?
dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1:y:0(dense/Tensordot/Reshape_1/shape:output:0*
_output_shapes

:	 *
T0?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0"dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:_
dense/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*/
_output_shapes
:????????? *
T0?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? *
T0b
	dense/EluEludense/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  `
dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:k
dense_1/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          ^
dense_1/Tensordot/ShapeShapedense/Elu:activations:0*
_output_shapes
:*
T0a
dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0c
!dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0a
dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Elu:activations:0!dense_1/Tensordot/concat:output:0*/
_output_shapes
:????????? *
T0?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????s
"dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_1/Tensordot/transpose_1	Transpose(dense_1/Tensordot/ReadVariableOp:value:0+dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:  r
!dense_1/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"        ?
dense_1/Tensordot/Reshape_1Reshape!dense_1/Tensordot/transpose_1:y:0*dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:  ?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0$dense_1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:a
dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
dense_1/EluEludense_1/BiasAdd:output:0*/
_output_shapes
:????????? *
T0?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: `
dense_2/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:k
dense_2/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          `
dense_2/Tensordot/ShapeShapedense_1/Elu:activations:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: ?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense_2/Tensordot/transpose	Transposedense_1/Elu:activations:0!dense_2/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????s
"dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_2/Tensordot/transpose_1	Transpose(dense_2/Tensordot/ReadVariableOp:value:0+dense_2/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: r
!dense_2/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense_2/Tensordot/Reshape_1Reshape!dense_2/Tensordot/transpose_1:y:0*dense_2/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: ?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0$dense_2/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????c
dense_2/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:a
dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*/
_output_shapes
:?????????*
T0?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0?
IdentityIdentitydense_2/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp: : : : : : :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_3005

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:????????? *
T0N
EluEluBiasAdd:output:0*'
_output_shapes
:????????? *
T0?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*.
_input_shapes
:????????? ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
s
G__inference_concatenate_4_layer_call_and_return_conditional_losses_2882
inputs_0
inputs_1
identityM
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
T0*
N*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?1
?
 __inference__traced_restore_3175
file_prefix$
 assignvariableop_conv2d_1_kernel$
 assignvariableop_1_conv2d_1_bias#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias%
!assignvariableop_4_dense_1_kernel#
assignvariableop_5_dense_1_bias%
!assignvariableop_6_dense_2_kernel#
assignvariableop_7_dense_2_bias%
!assignvariableop_8_dense_9_kernel#
assignvariableop_9_dense_9_bias'
#assignvariableop_10_dense_10_kernel%
!assignvariableop_11_dense_10_bias
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
RestoreV2/shape_and_slicesConst"/device:CPU:0*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0|
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:}
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_9_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_9_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_10_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_10_biasIdentity_11:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_8:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : 
?
?
2__inference_TimeDerivativeModel_layer_call_fn_1409
oeb"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalloebstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*+
_gradient_op_typePartitionedCall-1402*V
fQRO
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1401*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :# 

_user_specified_nameOEb: 
?
?
&__inference_dense_9_layer_call_fn_3047

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_1331*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*(
_output_shapes
:??????????*+
_gradient_op_typePartitionedCall-1337?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?

?
,__inference_RelationModel_layer_call_fn_2787
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin

2*+
_gradient_op_typePartitionedCall-1835*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1818*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : 
?
?
$__inference_dense_layer_call_fn_2994

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? *+
_gradient_op_typePartitionedCall-1149*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1143?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*.
_input_shapes
:?????????	::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
s
G__inference_concatenate_3_layer_call_and_return_conditional_losses_2537
inputs_0
inputs_1
identityM
concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*I
_input_shapes8
6:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?	
?
,__inference_RelationModel_layer_call_fn_2853
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*-
config_proto

CPU

GPU2*0J 8*
Tin

2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1253*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1252*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : 
?
n
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_1463

inputs
identityR
ExpandDims_4/dimConst*
dtype0*
_output_shapes
: *
value	B :s
ExpandDims_4
ExpandDimsinputsExpandDims_4/dim:output:0*+
_output_shapes
:?????????*
T0a
IdentityIdentityExpandDims_4:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
X
,__inference_concatenate_4_layer_call_fn_2888
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1887*P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_1880`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
C
'__inference_SelfLoop_layer_call_fn_1086

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*=
_output_shapes+
):'???????????????????????????*+
_gradient_op_typePartitionedCall-1083*K
fFRD
B__inference_SelfLoop_layer_call_and_return_conditional_losses_1077*
Tout
2*-
config_proto

CPU

GPU2*0J 8v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:& "
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_1068
orr
raA
=model_1_relationmodel_dense_tensordot_readvariableop_resource?
;model_1_relationmodel_dense_biasadd_readvariableop_resourceC
?model_1_relationmodel_dense_1_tensordot_readvariableop_resourceA
=model_1_relationmodel_dense_1_biasadd_readvariableop_resourceC
?model_1_relationmodel_dense_2_tensordot_readvariableop_resourceA
=model_1_relationmodel_dense_2_biasadd_readvariableop_resource3
/model_1_conv2d_1_conv2d_readvariableop_resource4
0model_1_conv2d_1_biasadd_readvariableop_resourceF
Bmodel_1_timederivativemodel_dense_9_matmul_readvariableop_resourceG
Cmodel_1_timederivativemodel_dense_9_biasadd_readvariableop_resourceG
Cmodel_1_timederivativemodel_dense_10_matmul_readvariableop_resourceH
Dmodel_1_timederivativemodel_dense_10_biasadd_readvariableop_resource
identity??2model_1/RelationModel/dense/BiasAdd/ReadVariableOp?4model_1/RelationModel/dense/Tensordot/ReadVariableOp?4model_1/RelationModel/dense_1/BiasAdd/ReadVariableOp?6model_1/RelationModel/dense_1/Tensordot/ReadVariableOp?4model_1/RelationModel/dense_2/BiasAdd/ReadVariableOp?6model_1/RelationModel/dense_2/Tensordot/ReadVariableOp?;model_1/TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp?:model_1/TimeDerivativeModel/dense_10/MatMul/ReadVariableOp?:model_1/TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp?9model_1/TimeDerivativeModel/dense_9/MatMul/ReadVariableOp?'model_1/conv2d_1/BiasAdd/ReadVariableOp?&model_1/conv2d_1/Conv2D/ReadVariableOp?
9model_1/tf_op_layer_strided_slice_1/strided_slice_1/beginConst*%
valueB"    ????        *
dtype0*
_output_shapes
:?
7model_1/tf_op_layer_strided_slice_1/strided_slice_1/endConst*%
valueB"               *
dtype0*
_output_shapes
:?
;model_1/tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*%
valueB"            *
dtype0*
_output_shapes
:?
3model_1/tf_op_layer_strided_slice_1/strided_slice_1StridedSliceorrBmodel_1/tf_op_layer_strided_slice_1/strided_slice_1/begin:output:0@model_1/tf_op_layer_strided_slice_1/strided_slice_1/end:output:0Dmodel_1/tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*

begin_mask*
end_mask*'
_output_shapes
:?????????*
T0*
Index0*
shrink_axis_mask
s
1model_1/tf_op_layer_ExpandDims_4/ExpandDims_4/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
-model_1/tf_op_layer_ExpandDims_4/ExpandDims_4
ExpandDims<model_1/tf_op_layer_strided_slice_1/strided_slice_1:output:0:model_1/tf_op_layer_ExpandDims_4/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:??????????
+model_1/tf_op_layer_Tile_4/Tile_4/multiplesConst*!
valueB"         *
dtype0*
_output_shapes
:?
!model_1/tf_op_layer_Tile_4/Tile_4Tile6model_1/tf_op_layer_ExpandDims_4/ExpandDims_4:output:04model_1/tf_op_layer_Tile_4/Tile_4/multiples:output:0*
T0*+
_output_shapes
:?????????s
1model_1/tf_op_layer_ExpandDims_3/ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
-model_1/tf_op_layer_ExpandDims_3/ExpandDims_3
ExpandDimsra:model_1/tf_op_layer_ExpandDims_3/ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:?????????t
model_1/SelfLoop/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
model_1/SelfLoop/transpose	Transpose*model_1/tf_op_layer_Tile_4/Tile_4:output:0(model_1/SelfLoop/transpose/perm:output:0*+
_output_shapes
:?????????*
T0?
+model_1/tf_op_layer_Tile_3/Tile_3/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:?
!model_1/tf_op_layer_Tile_3/Tile_3Tile6model_1/tf_op_layer_ExpandDims_3/ExpandDims_3:output:04model_1/tf_op_layer_Tile_3/Tile_3/multiples:output:0*/
_output_shapes
:?????????*
T0s
1model_1/tf_op_layer_ExpandDims_5/ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
-model_1/tf_op_layer_ExpandDims_5/ExpandDims_5
ExpandDimsmodel_1/SelfLoop/transpose:y:0:model_1/tf_op_layer_ExpandDims_5/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:?????????c
!model_1/concatenate_3/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
model_1/concatenate_3/concatConcatV2orr*model_1/tf_op_layer_Tile_3/Tile_3:output:0*model_1/concatenate_3/concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????y
 model_1/permute_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:?
model_1/permute_1/transpose	Transpose%model_1/concatenate_3/concat:output:0)model_1/permute_1/transpose/perm:output:0*
T0*/
_output_shapes
:??????????
+model_1/tf_op_layer_Tile_5/Tile_5/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:?
!model_1/tf_op_layer_Tile_5/Tile_5Tile6model_1/tf_op_layer_ExpandDims_5/ExpandDims_5:output:04model_1/tf_op_layer_Tile_5/Tile_5/multiples:output:0*
T0*/
_output_shapes
:?????????o
-model_1/RelationModel/concatenate/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :?
(model_1/RelationModel/concatenate/concatConcatV2model_1/permute_1/transpose:y:0*model_1/tf_op_layer_Tile_5/Tile_5:output:06model_1/RelationModel/concatenate/concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????	?
4model_1/RelationModel/dense/Tensordot/ReadVariableOpReadVariableOp=model_1_relationmodel_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 t
*model_1/RelationModel/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
*model_1/RelationModel/dense/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:?
+model_1/RelationModel/dense/Tensordot/ShapeShape1model_1/RelationModel/concatenate/concat:output:0*
T0*
_output_shapes
:u
3model_1/RelationModel/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
.model_1/RelationModel/dense/Tensordot/GatherV2GatherV24model_1/RelationModel/dense/Tensordot/Shape:output:03model_1/RelationModel/dense/Tensordot/free:output:0<model_1/RelationModel/dense/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0w
5model_1/RelationModel/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
0model_1/RelationModel/dense/Tensordot/GatherV2_1GatherV24model_1/RelationModel/dense/Tensordot/Shape:output:03model_1/RelationModel/dense/Tensordot/axes:output:0>model_1/RelationModel/dense/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0u
+model_1/RelationModel/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
*model_1/RelationModel/dense/Tensordot/ProdProd7model_1/RelationModel/dense/Tensordot/GatherV2:output:04model_1/RelationModel/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: w
-model_1/RelationModel/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
,model_1/RelationModel/dense/Tensordot/Prod_1Prod9model_1/RelationModel/dense/Tensordot/GatherV2_1:output:06model_1/RelationModel/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: s
1model_1/RelationModel/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
,model_1/RelationModel/dense/Tensordot/concatConcatV23model_1/RelationModel/dense/Tensordot/free:output:03model_1/RelationModel/dense/Tensordot/axes:output:0:model_1/RelationModel/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
+model_1/RelationModel/dense/Tensordot/stackPack3model_1/RelationModel/dense/Tensordot/Prod:output:05model_1/RelationModel/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
/model_1/RelationModel/dense/Tensordot/transpose	Transpose1model_1/RelationModel/concatenate/concat:output:05model_1/RelationModel/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????	?
-model_1/RelationModel/dense/Tensordot/ReshapeReshape3model_1/RelationModel/dense/Tensordot/transpose:y:04model_1/RelationModel/dense/Tensordot/stack:output:0*0
_output_shapes
:??????????????????*
T0?
6model_1/RelationModel/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
1model_1/RelationModel/dense/Tensordot/transpose_1	Transpose<model_1/RelationModel/dense/Tensordot/ReadVariableOp:value:0?model_1/RelationModel/dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:	 ?
5model_1/RelationModel/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"	       ?
/model_1/RelationModel/dense/Tensordot/Reshape_1Reshape5model_1/RelationModel/dense/Tensordot/transpose_1:y:0>model_1/RelationModel/dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:	 ?
,model_1/RelationModel/dense/Tensordot/MatMulMatMul6model_1/RelationModel/dense/Tensordot/Reshape:output:08model_1/RelationModel/dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? w
-model_1/RelationModel/dense/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB: u
3model_1/RelationModel/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
.model_1/RelationModel/dense/Tensordot/concat_1ConcatV27model_1/RelationModel/dense/Tensordot/GatherV2:output:06model_1/RelationModel/dense/Tensordot/Const_2:output:0<model_1/RelationModel/dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
%model_1/RelationModel/dense/TensordotReshape6model_1/RelationModel/dense/Tensordot/MatMul:product:07model_1/RelationModel/dense/Tensordot/concat_1:output:0*/
_output_shapes
:????????? *
T0?
2model_1/RelationModel/dense/BiasAdd/ReadVariableOpReadVariableOp;model_1_relationmodel_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
#model_1/RelationModel/dense/BiasAddBiasAdd.model_1/RelationModel/dense/Tensordot:output:0:model_1/RelationModel/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
model_1/RelationModel/dense/EluElu,model_1/RelationModel/dense/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
6model_1/RelationModel/dense_1/Tensordot/ReadVariableOpReadVariableOp?model_1_relationmodel_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  v
,model_1/RelationModel/dense_1/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:?
,model_1/RelationModel/dense_1/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:?
-model_1/RelationModel/dense_1/Tensordot/ShapeShape-model_1/RelationModel/dense/Elu:activations:0*
T0*
_output_shapes
:w
5model_1/RelationModel/dense_1/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
0model_1/RelationModel/dense_1/Tensordot/GatherV2GatherV26model_1/RelationModel/dense_1/Tensordot/Shape:output:05model_1/RelationModel/dense_1/Tensordot/free:output:0>model_1/RelationModel/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_1/RelationModel/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
2model_1/RelationModel/dense_1/Tensordot/GatherV2_1GatherV26model_1/RelationModel/dense_1/Tensordot/Shape:output:05model_1/RelationModel/dense_1/Tensordot/axes:output:0@model_1/RelationModel/dense_1/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0w
-model_1/RelationModel/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
,model_1/RelationModel/dense_1/Tensordot/ProdProd9model_1/RelationModel/dense_1/Tensordot/GatherV2:output:06model_1/RelationModel/dense_1/Tensordot/Const:output:0*
_output_shapes
: *
T0y
/model_1/RelationModel/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
.model_1/RelationModel/dense_1/Tensordot/Prod_1Prod;model_1/RelationModel/dense_1/Tensordot/GatherV2_1:output:08model_1/RelationModel/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3model_1/RelationModel/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
.model_1/RelationModel/dense_1/Tensordot/concatConcatV25model_1/RelationModel/dense_1/Tensordot/free:output:05model_1/RelationModel/dense_1/Tensordot/axes:output:0<model_1/RelationModel/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
-model_1/RelationModel/dense_1/Tensordot/stackPack5model_1/RelationModel/dense_1/Tensordot/Prod:output:07model_1/RelationModel/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
1model_1/RelationModel/dense_1/Tensordot/transpose	Transpose-model_1/RelationModel/dense/Elu:activations:07model_1/RelationModel/dense_1/Tensordot/concat:output:0*/
_output_shapes
:????????? *
T0?
/model_1/RelationModel/dense_1/Tensordot/ReshapeReshape5model_1/RelationModel/dense_1/Tensordot/transpose:y:06model_1/RelationModel/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
8model_1/RelationModel/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
3model_1/RelationModel/dense_1/Tensordot/transpose_1	Transpose>model_1/RelationModel/dense_1/Tensordot/ReadVariableOp:value:0Amodel_1/RelationModel/dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:  ?
7model_1/RelationModel/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"        *
dtype0*
_output_shapes
:?
1model_1/RelationModel/dense_1/Tensordot/Reshape_1Reshape7model_1/RelationModel/dense_1/Tensordot/transpose_1:y:0@model_1/RelationModel/dense_1/Tensordot/Reshape_1/shape:output:0*
_output_shapes

:  *
T0?
.model_1/RelationModel/dense_1/Tensordot/MatMulMatMul8model_1/RelationModel/dense_1/Tensordot/Reshape:output:0:model_1/RelationModel/dense_1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? y
/model_1/RelationModel/dense_1/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:w
5model_1/RelationModel/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
0model_1/RelationModel/dense_1/Tensordot/concat_1ConcatV29model_1/RelationModel/dense_1/Tensordot/GatherV2:output:08model_1/RelationModel/dense_1/Tensordot/Const_2:output:0>model_1/RelationModel/dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
'model_1/RelationModel/dense_1/TensordotReshape8model_1/RelationModel/dense_1/Tensordot/MatMul:product:09model_1/RelationModel/dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
4model_1/RelationModel/dense_1/BiasAdd/ReadVariableOpReadVariableOp=model_1_relationmodel_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
%model_1/RelationModel/dense_1/BiasAddBiasAdd0model_1/RelationModel/dense_1/Tensordot:output:0<model_1/RelationModel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
!model_1/RelationModel/dense_1/EluElu.model_1/RelationModel/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
6model_1/RelationModel/dense_2/Tensordot/ReadVariableOpReadVariableOp?model_1_relationmodel_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: v
,model_1/RelationModel/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:?
,model_1/RelationModel/dense_2/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:?
-model_1/RelationModel/dense_2/Tensordot/ShapeShape/model_1/RelationModel/dense_1/Elu:activations:0*
_output_shapes
:*
T0w
5model_1/RelationModel/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
0model_1/RelationModel/dense_2/Tensordot/GatherV2GatherV26model_1/RelationModel/dense_2/Tensordot/Shape:output:05model_1/RelationModel/dense_2/Tensordot/free:output:0>model_1/RelationModel/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_1/RelationModel/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
2model_1/RelationModel/dense_2/Tensordot/GatherV2_1GatherV26model_1/RelationModel/dense_2/Tensordot/Shape:output:05model_1/RelationModel/dense_2/Tensordot/axes:output:0@model_1/RelationModel/dense_2/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0w
-model_1/RelationModel/dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
,model_1/RelationModel/dense_2/Tensordot/ProdProd9model_1/RelationModel/dense_2/Tensordot/GatherV2:output:06model_1/RelationModel/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/model_1/RelationModel/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
.model_1/RelationModel/dense_2/Tensordot/Prod_1Prod;model_1/RelationModel/dense_2/Tensordot/GatherV2_1:output:08model_1/RelationModel/dense_2/Tensordot/Const_1:output:0*
_output_shapes
: *
T0u
3model_1/RelationModel/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
.model_1/RelationModel/dense_2/Tensordot/concatConcatV25model_1/RelationModel/dense_2/Tensordot/free:output:05model_1/RelationModel/dense_2/Tensordot/axes:output:0<model_1/RelationModel/dense_2/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
-model_1/RelationModel/dense_2/Tensordot/stackPack5model_1/RelationModel/dense_2/Tensordot/Prod:output:07model_1/RelationModel/dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
1model_1/RelationModel/dense_2/Tensordot/transpose	Transpose/model_1/RelationModel/dense_1/Elu:activations:07model_1/RelationModel/dense_2/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
/model_1/RelationModel/dense_2/Tensordot/ReshapeReshape5model_1/RelationModel/dense_2/Tensordot/transpose:y:06model_1/RelationModel/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
8model_1/RelationModel/dense_2/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       ?
3model_1/RelationModel/dense_2/Tensordot/transpose_1	Transpose>model_1/RelationModel/dense_2/Tensordot/ReadVariableOp:value:0Amodel_1/RelationModel/dense_2/Tensordot/transpose_1/perm:output:0*
_output_shapes

: *
T0?
7model_1/RelationModel/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:?
1model_1/RelationModel/dense_2/Tensordot/Reshape_1Reshape7model_1/RelationModel/dense_2/Tensordot/transpose_1:y:0@model_1/RelationModel/dense_2/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: ?
.model_1/RelationModel/dense_2/Tensordot/MatMulMatMul8model_1/RelationModel/dense_2/Tensordot/Reshape:output:0:model_1/RelationModel/dense_2/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????y
/model_1/RelationModel/dense_2/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:w
5model_1/RelationModel/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
0model_1/RelationModel/dense_2/Tensordot/concat_1ConcatV29model_1/RelationModel/dense_2/Tensordot/GatherV2:output:08model_1/RelationModel/dense_2/Tensordot/Const_2:output:0>model_1/RelationModel/dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
'model_1/RelationModel/dense_2/TensordotReshape8model_1/RelationModel/dense_2/Tensordot/MatMul:product:09model_1/RelationModel/dense_2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:??????????
4model_1/RelationModel/dense_2/BiasAdd/ReadVariableOpReadVariableOp=model_1_relationmodel_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
%model_1/RelationModel/dense_2/BiasAddBiasAdd0model_1/RelationModel/dense_2/Tensordot:output:0<model_1/RelationModel/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
&model_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
model_1/conv2d_1/Conv2DConv2D.model_1/RelationModel/dense_2/BiasAdd:output:0.model_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:?????????*
T0*
strides
?
'model_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
model_1/conv2d_1/BiasAddBiasAdd model_1/conv2d_1/Conv2D:output:0/model_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0?
'model_1/tf_op_layer_Squeeze_1/Squeeze_1Squeeze!model_1/conv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
c
!model_1/concatenate_4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
model_1/concatenate_4/concatConcatV2<model_1/tf_op_layer_strided_slice_1/strided_slice_1:output:00model_1/tf_op_layer_Squeeze_1/Squeeze_1:output:0*model_1/concatenate_4/concat/axis:output:0*
N*'
_output_shapes
:?????????*
T0?
9model_1/TimeDerivativeModel/dense_9/MatMul/ReadVariableOpReadVariableOpBmodel_1_timederivativemodel_dense_9_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	??
*model_1/TimeDerivativeModel/dense_9/MatMulMatMul%model_1/concatenate_4/concat:output:0Amodel_1/TimeDerivativeModel/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
:model_1/TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOpReadVariableOpCmodel_1_timederivativemodel_dense_9_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:??
+model_1/TimeDerivativeModel/dense_9/BiasAddBiasAdd4model_1/TimeDerivativeModel/dense_9/MatMul:product:0Bmodel_1/TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'model_1/TimeDerivativeModel/dense_9/EluElu4model_1/TimeDerivativeModel/dense_9/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
:model_1/TimeDerivativeModel/dense_10/MatMul/ReadVariableOpReadVariableOpCmodel_1_timederivativemodel_dense_10_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	??
+model_1/TimeDerivativeModel/dense_10/MatMulMatMul5model_1/TimeDerivativeModel/dense_9/Elu:activations:0Bmodel_1/TimeDerivativeModel/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;model_1/TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOpReadVariableOpDmodel_1_timederivativemodel_dense_10_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
,model_1/TimeDerivativeModel/dense_10/BiasAddBiasAdd5model_1/TimeDerivativeModel/dense_10/MatMul:product:0Cmodel_1/TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
!model_1/tf_op_layer_mul_1/mul_1/yConst*
valueB
 *??8*
dtype0*
_output_shapes
: ?
model_1/tf_op_layer_mul_1/mul_1Mul5model_1/TimeDerivativeModel/dense_10/BiasAdd:output:0*model_1/tf_op_layer_mul_1/mul_1/y:output:0*
T0*'
_output_shapes
:??????????
model_1/output/addAddV2<model_1/tf_op_layer_strided_slice_1/strided_slice_1:output:0#model_1/tf_op_layer_mul_1/mul_1:z:0*
T0*'
_output_shapes
:??????????
IdentityIdentitymodel_1/output/add:z:03^model_1/RelationModel/dense/BiasAdd/ReadVariableOp5^model_1/RelationModel/dense/Tensordot/ReadVariableOp5^model_1/RelationModel/dense_1/BiasAdd/ReadVariableOp7^model_1/RelationModel/dense_1/Tensordot/ReadVariableOp5^model_1/RelationModel/dense_2/BiasAdd/ReadVariableOp7^model_1/RelationModel/dense_2/Tensordot/ReadVariableOp<^model_1/TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp;^model_1/TimeDerivativeModel/dense_10/MatMul/ReadVariableOp;^model_1/TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp:^model_1/TimeDerivativeModel/dense_9/MatMul/ReadVariableOp(^model_1/conv2d_1/BiasAdd/ReadVariableOp'^model_1/conv2d_1/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::2z
;model_1/TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp;model_1/TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp2R
'model_1/conv2d_1/BiasAdd/ReadVariableOp'model_1/conv2d_1/BiasAdd/ReadVariableOp2P
&model_1/conv2d_1/Conv2D/ReadVariableOp&model_1/conv2d_1/Conv2D/ReadVariableOp2p
6model_1/RelationModel/dense_1/Tensordot/ReadVariableOp6model_1/RelationModel/dense_1/Tensordot/ReadVariableOp2l
4model_1/RelationModel/dense_1/BiasAdd/ReadVariableOp4model_1/RelationModel/dense_1/BiasAdd/ReadVariableOp2p
6model_1/RelationModel/dense_2/Tensordot/ReadVariableOp6model_1/RelationModel/dense_2/Tensordot/ReadVariableOp2x
:model_1/TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp:model_1/TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp2x
:model_1/TimeDerivativeModel/dense_10/MatMul/ReadVariableOp:model_1/TimeDerivativeModel/dense_10/MatMul/ReadVariableOp2l
4model_1/RelationModel/dense/Tensordot/ReadVariableOp4model_1/RelationModel/dense/Tensordot/ReadVariableOp2v
9model_1/TimeDerivativeModel/dense_9/MatMul/ReadVariableOp9model_1/TimeDerivativeModel/dense_9/MatMul/ReadVariableOp2h
2model_1/RelationModel/dense/BiasAdd/ReadVariableOp2model_1/RelationModel/dense/BiasAdd/ReadVariableOp2l
4model_1/RelationModel/dense_2/BiasAdd/ReadVariableOp4model_1/RelationModel/dense_2/BiasAdd/ReadVariableOp:# 

_user_specified_nameORr:"

_user_specified_nameRa: : : : : : : :	 :
 : : : 
?
V
*__inference_concatenate_layer_call_fn_2976
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*+
_gradient_op_typePartitionedCall-1125*N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1118*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????	*
Tin
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????	"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?l
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_1719

inputs
inputs_1+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOpY
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate/concatConcatV2inputsinputs_1 concatenate/concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????	?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 ^
dense/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:i
dense/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          `
dense/Tensordot/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0_
dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
_output_shapes
: *
T0a
dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense/Tensordot/transpose	Transposeconcatenate/concat:output:0dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????	?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????q
 dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense/Tensordot/transpose_1	Transpose&dense/Tensordot/ReadVariableOp:value:0)dense/Tensordot/transpose_1/perm:output:0*
_output_shapes

:	 *
T0p
dense/Tensordot/Reshape_1/shapeConst*
valueB"	       *
dtype0*
_output_shapes
:?
dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1:y:0(dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:	 ?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0"dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:_
dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? *
T0b
	dense/EluEludense/BiasAdd:output:0*/
_output_shapes
:????????? *
T0?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  `
dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:k
dense_1/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:^
dense_1/Tensordot/ShapeShapedense/Elu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0c
!dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
_output_shapes
: *
T0_
dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Elu:activations:0!dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????s
"dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_1/Tensordot/transpose_1	Transpose(dense_1/Tensordot/ReadVariableOp:value:0+dense_1/Tensordot/transpose_1/perm:output:0*
_output_shapes

:  *
T0r
!dense_1/Tensordot/Reshape_1/shapeConst*
valueB"        *
dtype0*
_output_shapes
:?
dense_1/Tensordot/Reshape_1Reshape!dense_1/Tensordot/transpose_1:y:0*dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:  ?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0$dense_1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:a
dense_1/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
dense_1/EluEludense_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: `
dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:k
dense_2/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:`
dense_2/Tensordot/ShapeShapedense_1/Elu:activations:0*
_output_shapes
:*
T0a
dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0c
!dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0a
dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
_output_shapes
: *
T0_
dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
dense_2/Tensordot/transpose	Transposedense_1/Elu:activations:0!dense_2/Tensordot/concat:output:0*
T0*/
_output_shapes
:????????? ?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????s
"dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_2/Tensordot/transpose_1	Transpose(dense_2/Tensordot/ReadVariableOp:value:0+dense_2/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: r
!dense_2/Tensordot/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_2/Tensordot/Reshape_1Reshape!dense_2/Tensordot/transpose_1:y:0*dense_2/Tensordot/Reshape_1/shape:output:0*
_output_shapes

: *
T0?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0$dense_2/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????c
dense_2/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
IdentityIdentitydense_2/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : : : : : 
?
q
G__inference_concatenate_3_layer_call_and_return_conditional_losses_1587

inputs
inputs_1
identityM
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: }
concatConcatV2inputsinputs_1concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*I
_input_shapes8
6:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_1233	
orrra
selfloop(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCallorrraselfloop*+
_gradient_op_typePartitionedCall-1125*N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1118*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????	?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2*+
_gradient_op_typePartitionedCall-1149*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1143*
Tout
2?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1177*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1171*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1204*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1198*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : :% !

_user_specified_nameORrRa:($
"
_user_specified_name
selfLoop: : : : 
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_1143

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*.
_input_shapes
:?????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
??
?

A__inference_model_1_layer_call_and_return_conditional_losses_2273
inputs_0
inputs_19
5relationmodel_dense_tensordot_readvariableop_resource7
3relationmodel_dense_biasadd_readvariableop_resource;
7relationmodel_dense_1_tensordot_readvariableop_resource9
5relationmodel_dense_1_biasadd_readvariableop_resource;
7relationmodel_dense_2_tensordot_readvariableop_resource9
5relationmodel_dense_2_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource>
:timederivativemodel_dense_9_matmul_readvariableop_resource?
;timederivativemodel_dense_9_biasadd_readvariableop_resource?
;timederivativemodel_dense_10_matmul_readvariableop_resource@
<timederivativemodel_dense_10_biasadd_readvariableop_resource
identity??*RelationModel/dense/BiasAdd/ReadVariableOp?,RelationModel/dense/Tensordot/ReadVariableOp?,RelationModel/dense_1/BiasAdd/ReadVariableOp?.RelationModel/dense_1/Tensordot/ReadVariableOp?,RelationModel/dense_2/BiasAdd/ReadVariableOp?.RelationModel/dense_2/Tensordot/ReadVariableOp?3TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp?2TimeDerivativeModel/dense_10/MatMul/ReadVariableOp?2TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp?1TimeDerivativeModel/dense_9/MatMul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
1tf_op_layer_strided_slice_1/strided_slice_1/beginConst*
dtype0*
_output_shapes
:*%
valueB"    ????        ?
/tf_op_layer_strided_slice_1/strided_slice_1/endConst*%
valueB"               *
dtype0*
_output_shapes
:?
3tf_op_layer_strided_slice_1/strided_slice_1/stridesConst*%
valueB"            *
dtype0*
_output_shapes
:?
+tf_op_layer_strided_slice_1/strided_slice_1StridedSliceinputs_0:tf_op_layer_strided_slice_1/strided_slice_1/begin:output:08tf_op_layer_strided_slice_1/strided_slice_1/end:output:0<tf_op_layer_strided_slice_1/strided_slice_1/strides:output:0*
end_mask*'
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask
*

begin_maskk
)tf_op_layer_ExpandDims_4/ExpandDims_4/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
%tf_op_layer_ExpandDims_4/ExpandDims_4
ExpandDims4tf_op_layer_strided_slice_1/strided_slice_1:output:02tf_op_layer_ExpandDims_4/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:?????????x
#tf_op_layer_Tile_4/Tile_4/multiplesConst*!
valueB"         *
dtype0*
_output_shapes
:?
tf_op_layer_Tile_4/Tile_4Tile.tf_op_layer_ExpandDims_4/ExpandDims_4:output:0,tf_op_layer_Tile_4/Tile_4/multiples:output:0*
T0*+
_output_shapes
:?????????k
)tf_op_layer_ExpandDims_3/ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
%tf_op_layer_ExpandDims_3/ExpandDims_3
ExpandDimsinputs_12tf_op_layer_ExpandDims_3/ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:?????????l
SelfLoop/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
SelfLoop/transpose	Transpose"tf_op_layer_Tile_4/Tile_4:output:0 SelfLoop/transpose/perm:output:0*
T0*+
_output_shapes
:?????????|
#tf_op_layer_Tile_3/Tile_3/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:?
tf_op_layer_Tile_3/Tile_3Tile.tf_op_layer_ExpandDims_3/ExpandDims_3:output:0,tf_op_layer_Tile_3/Tile_3/multiples:output:0*
T0*/
_output_shapes
:?????????k
)tf_op_layer_ExpandDims_5/ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
%tf_op_layer_ExpandDims_5/ExpandDims_5
ExpandDimsSelfLoop/transpose:y:02tf_op_layer_ExpandDims_5/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:?????????[
concatenate_3/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate_3/concatConcatV2inputs_0"tf_op_layer_Tile_3/Tile_3:output:0"concatenate_3/concat/axis:output:0*
T0*
N*/
_output_shapes
:?????????q
permute_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:?
permute_1/transpose	Transposeconcatenate_3/concat:output:0!permute_1/transpose/perm:output:0*
T0*/
_output_shapes
:?????????|
#tf_op_layer_Tile_5/Tile_5/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:?
tf_op_layer_Tile_5/Tile_5Tile.tf_op_layer_ExpandDims_5/ExpandDims_5:output:0,tf_op_layer_Tile_5/Tile_5/multiples:output:0*
T0*/
_output_shapes
:?????????g
%RelationModel/concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
 RelationModel/concatenate/concatConcatV2permute_1/transpose:y:0"tf_op_layer_Tile_5/Tile_5:output:0.RelationModel/concatenate/concat/axis:output:0*
N*/
_output_shapes
:?????????	*
T0?
,RelationModel/dense/Tensordot/ReadVariableOpReadVariableOp5relationmodel_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 l
"RelationModel/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:w
"RelationModel/dense/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:|
#RelationModel/dense/Tensordot/ShapeShape)RelationModel/concatenate/concat:output:0*
T0*
_output_shapes
:m
+RelationModel/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
&RelationModel/dense/Tensordot/GatherV2GatherV2,RelationModel/dense/Tensordot/Shape:output:0+RelationModel/dense/Tensordot/free:output:04RelationModel/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-RelationModel/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
(RelationModel/dense/Tensordot/GatherV2_1GatherV2,RelationModel/dense/Tensordot/Shape:output:0+RelationModel/dense/Tensordot/axes:output:06RelationModel/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#RelationModel/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
"RelationModel/dense/Tensordot/ProdProd/RelationModel/dense/Tensordot/GatherV2:output:0,RelationModel/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%RelationModel/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
$RelationModel/dense/Tensordot/Prod_1Prod1RelationModel/dense/Tensordot/GatherV2_1:output:0.RelationModel/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)RelationModel/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
$RelationModel/dense/Tensordot/concatConcatV2+RelationModel/dense/Tensordot/free:output:0+RelationModel/dense/Tensordot/axes:output:02RelationModel/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
#RelationModel/dense/Tensordot/stackPack+RelationModel/dense/Tensordot/Prod:output:0-RelationModel/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
'RelationModel/dense/Tensordot/transpose	Transpose)RelationModel/concatenate/concat:output:0-RelationModel/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????	?
%RelationModel/dense/Tensordot/ReshapeReshape+RelationModel/dense/Tensordot/transpose:y:0,RelationModel/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????
.RelationModel/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
)RelationModel/dense/Tensordot/transpose_1	Transpose4RelationModel/dense/Tensordot/ReadVariableOp:value:07RelationModel/dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:	 ~
-RelationModel/dense/Tensordot/Reshape_1/shapeConst*
valueB"	       *
dtype0*
_output_shapes
:?
'RelationModel/dense/Tensordot/Reshape_1Reshape-RelationModel/dense/Tensordot/transpose_1:y:06RelationModel/dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:	 ?
$RelationModel/dense/Tensordot/MatMulMatMul.RelationModel/dense/Tensordot/Reshape:output:00RelationModel/dense/Tensordot/Reshape_1:output:0*'
_output_shapes
:????????? *
T0o
%RelationModel/dense/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:m
+RelationModel/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
&RelationModel/dense/Tensordot/concat_1ConcatV2/RelationModel/dense/Tensordot/GatherV2:output:0.RelationModel/dense/Tensordot/Const_2:output:04RelationModel/dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:?
RelationModel/dense/TensordotReshape.RelationModel/dense/Tensordot/MatMul:product:0/RelationModel/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:????????? ?
*RelationModel/dense/BiasAdd/ReadVariableOpReadVariableOp3relationmodel_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
RelationModel/dense/BiasAddBiasAdd&RelationModel/dense/Tensordot:output:02RelationModel/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
RelationModel/dense/EluElu$RelationModel/dense/BiasAdd:output:0*/
_output_shapes
:????????? *
T0?
.RelationModel/dense_1/Tensordot/ReadVariableOpReadVariableOp7relationmodel_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  n
$RelationModel/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:y
$RelationModel/dense_1/Tensordot/freeConst*!
valueB"          *
dtype0*
_output_shapes
:z
%RelationModel/dense_1/Tensordot/ShapeShape%RelationModel/dense/Elu:activations:0*
_output_shapes
:*
T0o
-RelationModel/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
(RelationModel/dense_1/Tensordot/GatherV2GatherV2.RelationModel/dense_1/Tensordot/Shape:output:0-RelationModel/dense_1/Tensordot/free:output:06RelationModel/dense_1/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0q
/RelationModel/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
*RelationModel/dense_1/Tensordot/GatherV2_1GatherV2.RelationModel/dense_1/Tensordot/Shape:output:0-RelationModel/dense_1/Tensordot/axes:output:08RelationModel/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%RelationModel/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
$RelationModel/dense_1/Tensordot/ProdProd1RelationModel/dense_1/Tensordot/GatherV2:output:0.RelationModel/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'RelationModel/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
&RelationModel/dense_1/Tensordot/Prod_1Prod3RelationModel/dense_1/Tensordot/GatherV2_1:output:00RelationModel/dense_1/Tensordot/Const_1:output:0*
_output_shapes
: *
T0m
+RelationModel/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
&RelationModel/dense_1/Tensordot/concatConcatV2-RelationModel/dense_1/Tensordot/free:output:0-RelationModel/dense_1/Tensordot/axes:output:04RelationModel/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
%RelationModel/dense_1/Tensordot/stackPack-RelationModel/dense_1/Tensordot/Prod:output:0/RelationModel/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:?
)RelationModel/dense_1/Tensordot/transpose	Transpose%RelationModel/dense/Elu:activations:0/RelationModel/dense_1/Tensordot/concat:output:0*/
_output_shapes
:????????? *
T0?
'RelationModel/dense_1/Tensordot/ReshapeReshape-RelationModel/dense_1/Tensordot/transpose:y:0.RelationModel/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0RelationModel/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
+RelationModel/dense_1/Tensordot/transpose_1	Transpose6RelationModel/dense_1/Tensordot/ReadVariableOp:value:09RelationModel/dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

:  ?
/RelationModel/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"        *
dtype0*
_output_shapes
:?
)RelationModel/dense_1/Tensordot/Reshape_1Reshape/RelationModel/dense_1/Tensordot/transpose_1:y:08RelationModel/dense_1/Tensordot/Reshape_1/shape:output:0*
_output_shapes

:  *
T0?
&RelationModel/dense_1/Tensordot/MatMulMatMul0RelationModel/dense_1/Tensordot/Reshape:output:02RelationModel/dense_1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:????????? q
'RelationModel/dense_1/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB: o
-RelationModel/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
(RelationModel/dense_1/Tensordot/concat_1ConcatV21RelationModel/dense_1/Tensordot/GatherV2:output:00RelationModel/dense_1/Tensordot/Const_2:output:06RelationModel/dense_1/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0?
RelationModel/dense_1/TensordotReshape0RelationModel/dense_1/Tensordot/MatMul:product:01RelationModel/dense_1/Tensordot/concat_1:output:0*/
_output_shapes
:????????? *
T0?
,RelationModel/dense_1/BiasAdd/ReadVariableOpReadVariableOp5relationmodel_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
RelationModel/dense_1/BiasAddBiasAdd(RelationModel/dense_1/Tensordot:output:04RelationModel/dense_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? *
T0?
RelationModel/dense_1/EluElu&RelationModel/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
.RelationModel/dense_2/Tensordot/ReadVariableOpReadVariableOp7relationmodel_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: n
$RelationModel/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:y
$RelationModel/dense_2/Tensordot/freeConst*
dtype0*
_output_shapes
:*!
valueB"          |
%RelationModel/dense_2/Tensordot/ShapeShape'RelationModel/dense_1/Elu:activations:0*
T0*
_output_shapes
:o
-RelationModel/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
(RelationModel/dense_2/Tensordot/GatherV2GatherV2.RelationModel/dense_2/Tensordot/Shape:output:0-RelationModel/dense_2/Tensordot/free:output:06RelationModel/dense_2/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0q
/RelationModel/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
*RelationModel/dense_2/Tensordot/GatherV2_1GatherV2.RelationModel/dense_2/Tensordot/Shape:output:0-RelationModel/dense_2/Tensordot/axes:output:08RelationModel/dense_2/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0o
%RelationModel/dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:?
$RelationModel/dense_2/Tensordot/ProdProd1RelationModel/dense_2/Tensordot/GatherV2:output:0.RelationModel/dense_2/Tensordot/Const:output:0*
_output_shapes
: *
T0q
'RelationModel/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:?
&RelationModel/dense_2/Tensordot/Prod_1Prod3RelationModel/dense_2/Tensordot/GatherV2_1:output:00RelationModel/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+RelationModel/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
&RelationModel/dense_2/Tensordot/concatConcatV2-RelationModel/dense_2/Tensordot/free:output:0-RelationModel/dense_2/Tensordot/axes:output:04RelationModel/dense_2/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:?
%RelationModel/dense_2/Tensordot/stackPack-RelationModel/dense_2/Tensordot/Prod:output:0/RelationModel/dense_2/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0?
)RelationModel/dense_2/Tensordot/transpose	Transpose'RelationModel/dense_1/Elu:activations:0/RelationModel/dense_2/Tensordot/concat:output:0*/
_output_shapes
:????????? *
T0?
'RelationModel/dense_2/Tensordot/ReshapeReshape-RelationModel/dense_2/Tensordot/transpose:y:0.RelationModel/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0RelationModel/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:?
+RelationModel/dense_2/Tensordot/transpose_1	Transpose6RelationModel/dense_2/Tensordot/ReadVariableOp:value:09RelationModel/dense_2/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: ?
/RelationModel/dense_2/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"       ?
)RelationModel/dense_2/Tensordot/Reshape_1Reshape/RelationModel/dense_2/Tensordot/transpose_1:y:08RelationModel/dense_2/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: ?
&RelationModel/dense_2/Tensordot/MatMulMatMul0RelationModel/dense_2/Tensordot/Reshape:output:02RelationModel/dense_2/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????q
'RelationModel/dense_2/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:o
-RelationModel/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: ?
(RelationModel/dense_2/Tensordot/concat_1ConcatV21RelationModel/dense_2/Tensordot/GatherV2:output:00RelationModel/dense_2/Tensordot/Const_2:output:06RelationModel/dense_2/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0?
RelationModel/dense_2/TensordotReshape0RelationModel/dense_2/Tensordot/MatMul:product:01RelationModel/dense_2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:??????????
,RelationModel/dense_2/BiasAdd/ReadVariableOpReadVariableOp5relationmodel_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
RelationModel/dense_2/BiasAddBiasAdd(RelationModel/dense_2/Tensordot:output:04RelationModel/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2d_1/Conv2DConv2D&RelationModel/dense_2/BiasAdd:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*/
_output_shapes
:?????????*
T0?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0?
tf_op_layer_Squeeze_1/Squeeze_1Squeezeconv2d_1/BiasAdd:output:0*
squeeze_dims
*
T0*'
_output_shapes
:?????????[
concatenate_4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate_4/concatConcatV24tf_op_layer_strided_slice_1/strided_slice_1:output:0(tf_op_layer_Squeeze_1/Squeeze_1:output:0"concatenate_4/concat/axis:output:0*
N*'
_output_shapes
:?????????*
T0?
1TimeDerivativeModel/dense_9/MatMul/ReadVariableOpReadVariableOp:timederivativemodel_dense_9_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	??
"TimeDerivativeModel/dense_9/MatMulMatMulconcatenate_4/concat:output:09TimeDerivativeModel/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOpReadVariableOp;timederivativemodel_dense_9_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:??
#TimeDerivativeModel/dense_9/BiasAddBiasAdd,TimeDerivativeModel/dense_9/MatMul:product:0:TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
TimeDerivativeModel/dense_9/EluElu,TimeDerivativeModel/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
2TimeDerivativeModel/dense_10/MatMul/ReadVariableOpReadVariableOp;timederivativemodel_dense_10_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	??
#TimeDerivativeModel/dense_10/MatMulMatMul-TimeDerivativeModel/dense_9/Elu:activations:0:TimeDerivativeModel/dense_10/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
3TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOpReadVariableOp<timederivativemodel_dense_10_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
$TimeDerivativeModel/dense_10/BiasAddBiasAdd-TimeDerivativeModel/dense_10/MatMul:product:0;TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
tf_op_layer_mul_1/mul_1/yConst*
valueB
 *??8*
dtype0*
_output_shapes
: ?
tf_op_layer_mul_1/mul_1Mul-TimeDerivativeModel/dense_10/BiasAdd:output:0"tf_op_layer_mul_1/mul_1/y:output:0*
T0*'
_output_shapes
:??????????

output/addAddV24tf_op_layer_strided_slice_1/strided_slice_1:output:0tf_op_layer_mul_1/mul_1:z:0*
T0*'
_output_shapes
:??????????
IdentityIdentityoutput/add:z:0+^RelationModel/dense/BiasAdd/ReadVariableOp-^RelationModel/dense/Tensordot/ReadVariableOp-^RelationModel/dense_1/BiasAdd/ReadVariableOp/^RelationModel/dense_1/Tensordot/ReadVariableOp-^RelationModel/dense_2/BiasAdd/ReadVariableOp/^RelationModel/dense_2/Tensordot/ReadVariableOp4^TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp3^TimeDerivativeModel/dense_10/MatMul/ReadVariableOp3^TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp2^TimeDerivativeModel/dense_9/MatMul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::2`
.RelationModel/dense_1/Tensordot/ReadVariableOp.RelationModel/dense_1/Tensordot/ReadVariableOp2\
,RelationModel/dense_1/BiasAdd/ReadVariableOp,RelationModel/dense_1/BiasAdd/ReadVariableOp2`
.RelationModel/dense_2/Tensordot/ReadVariableOp.RelationModel/dense_2/Tensordot/ReadVariableOp2h
2TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp2TimeDerivativeModel/dense_9/BiasAdd/ReadVariableOp2h
2TimeDerivativeModel/dense_10/MatMul/ReadVariableOp2TimeDerivativeModel/dense_10/MatMul/ReadVariableOp2\
,RelationModel/dense/Tensordot/ReadVariableOp,RelationModel/dense/Tensordot/ReadVariableOp2f
1TimeDerivativeModel/dense_9/MatMul/ReadVariableOp1TimeDerivativeModel/dense_9/MatMul/ReadVariableOp2X
*RelationModel/dense/BiasAdd/ReadVariableOp*RelationModel/dense/BiasAdd/ReadVariableOp2\
,RelationModel/dense_2/BiasAdd/ReadVariableOp,RelationModel/dense_2/BiasAdd/ReadVariableOp2j
3TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp3TimeDerivativeModel/dense_10/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : : :	 :
 : : : 
?
U
7__inference_tf_op_layer_ExpandDims_4_layer_call_fn_2497
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*+
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1469*[
fVRT
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_1463*
Tout
2*-
config_proto

CPU

GPU2*0J 8d
IdentityIdentityPartitionedCall:output:0*+
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
?
2__inference_TimeDerivativeModel_layer_call_fn_2931

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*V
fQRO
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1401*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1402?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
N
0__inference_tf_op_layer_mul_1_layer_call_fn_2951
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*+
_gradient_op_typePartitionedCall-1929*T
fORM
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_1923*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
j
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_2560
inputs_0
identityi
Tile_5/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:m
Tile_5Tileinputs_0Tile_5/multiples:output:0*
T0*/
_output_shapes
:?????????_
IdentityIdentityTile_5:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?C
?
A__inference_model_1_layer_call_and_return_conditional_losses_2089

inputs
inputs_10
,relationmodel_statefulpartitionedcall_args_20
,relationmodel_statefulpartitionedcall_args_30
,relationmodel_statefulpartitionedcall_args_40
,relationmodel_statefulpartitionedcall_args_50
,relationmodel_statefulpartitionedcall_args_60
,relationmodel_statefulpartitionedcall_args_7+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_26
2timederivativemodel_statefulpartitionedcall_args_16
2timederivativemodel_statefulpartitionedcall_args_26
2timederivativemodel_statefulpartitionedcall_args_36
2timederivativemodel_statefulpartitionedcall_args_4
identity??%RelationModel/StatefulPartitionedCall?+TimeDerivativeModel/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
+tf_op_layer_strided_slice_1/PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-1450*^
fYRW
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_1444*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:??????????
(tf_op_layer_ExpandDims_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1469*[
fVRT
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_1463*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
Ra/PartitionedCallPartitionedCallinputs_1*+
_gradient_op_typePartitionedCall-1498*E
f@R>
<__inference_Ra_layer_call_and_return_conditional_losses_1486*
Tout
2*-
config_proto

CPU

GPU2*0J 8*+
_output_shapes
:?????????*
Tin
2?
"tf_op_layer_Tile_4/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_4/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1515*U
fPRN
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_1509*
Tout
2?
(tf_op_layer_ExpandDims_3/PartitionedCallPartitionedCallRa/PartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1534*[
fVRT
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_1528*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
SelfLoop/PartitionedCallPartitionedCall+tf_op_layer_Tile_4/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1083*K
fFRD
B__inference_SelfLoop_layer_call_and_return_conditional_losses_1077*
Tout
2*-
config_proto

CPU

GPU2*0J 8*+
_output_shapes
:?????????*
Tin
2?
"tf_op_layer_Tile_3/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_3/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1554*U
fPRN
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_1548*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
(tf_op_layer_ExpandDims_5/PartitionedCallPartitionedCall!SelfLoop/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1573*[
fVRT
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_1567*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
concatenate_3/PartitionedCallPartitionedCallinputs+tf_op_layer_Tile_3/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1594*P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_1587*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
permute_1/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1101*L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_1095*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2?
"tf_op_layer_Tile_5/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_5/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1614*U
fPRN
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_1608*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
%RelationModel/StatefulPartitionedCallStatefulPartitionedCall"permute_1/PartitionedCall:output:0+tf_op_layer_Tile_5/PartitionedCall:output:0,relationmodel_statefulpartitionedcall_args_2,relationmodel_statefulpartitionedcall_args_3,relationmodel_statefulpartitionedcall_args_4,relationmodel_statefulpartitionedcall_args_5,relationmodel_statefulpartitionedcall_args_6,relationmodel_statefulpartitionedcall_args_7*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin

2*+
_gradient_op_typePartitionedCall-1835*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1818?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall.RelationModel/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1309*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1303?
%tf_op_layer_Squeeze_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1866*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_1860*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:??????????
concatenate_4/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0.tf_op_layer_Squeeze_1/PartitionedCall:output:0*P
fKRI
G__inference_concatenate_4_layer_call_and_return_conditional_losses_1880*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1887?
+TimeDerivativeModel/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:02timederivativemodel_statefulpartitionedcall_args_12timederivativemodel_statefulpartitionedcall_args_22timederivativemodel_statefulpartitionedcall_args_32timederivativemodel_statefulpartitionedcall_args_4*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1424*V
fQRO
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1423?
!tf_op_layer_mul_1/PartitionedCallPartitionedCall4TimeDerivativeModel/StatefulPartitionedCall:output:0*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1929*T
fORM
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_1923*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
output/PartitionedCallPartitionedCall4tf_op_layer_strided_slice_1/PartitionedCall:output:0*tf_op_layer_mul_1/PartitionedCall:output:0*'
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1949*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_1942*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentityoutput/PartitionedCall:output:0&^RelationModel/StatefulPartitionedCall,^TimeDerivativeModel/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*u
_input_shapesd
b:?????????:?????????::::::::::::2N
%RelationModel/StatefulPartitionedCall%RelationModel/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2Z
+TimeDerivativeModel/StatefulPartitionedCall+TimeDerivativeModel/StatefulPartitionedCall: : :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : : : : : : :	 :
 : 
?	
?
,__inference_RelationModel_layer_call_fn_1292	
orrra
selfloop"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallorrraselfloopstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1282*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin

2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1283?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :% !

_user_specified_nameORrRa:($
"
_user_specified_name
selfLoop: : 
?
X
,__inference_concatenate_3_layer_call_fn_2543
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1594*P
fKRI
G__inference_concatenate_3_layer_call_and_return_conditional_losses_1587h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*I
_input_shapes8
6:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
'__inference_dense_10_layer_call_fn_3064

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1364*K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_1358*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
U
7__inference_tf_op_layer_ExpandDims_5_layer_call_fn_2554
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*+
_gradient_op_typePartitionedCall-1573*[
fVRT
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_1567*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_1171

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
A__inference_dense_9_layer_call_and_return_conditional_losses_1331

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	?j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_2841
inputs_0
inputs_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpY
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
T0*
N*'
_output_shapes
:?????????	?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 ?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? Z
	dense/EluEludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  ?
dense_1/MatMulMatMuldense/Elu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ^
dense_1/EluEludense_1/BiasAdd:output:0*'
_output_shapes
:????????? *
T0?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: ?
dense_2/MatMulMatMuldense_1/Elu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_2/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : 
?
l
@__inference_output_layer_call_and_return_conditional_losses_2957
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_2814
inputs_0
inputs_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpY
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
T0*
N*'
_output_shapes
:?????????	?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:	 ?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? Z
	dense/EluEludense/BiasAdd:output:0*'
_output_shapes
:????????? *
T0?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:  ?
dense_1/MatMulMatMuldense/Elu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ^
dense_1/EluEludense_1/BiasAdd:output:0*'
_output_shapes
:????????? *
T0?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

: ?
dense_2/MatMulMatMuldense_1/Elu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_2/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : 
?
O
1__inference_tf_op_layer_Tile_3_layer_call_fn_2530
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*U
fPRN
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_1548*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1554h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
q
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_1444

inputs
identityn
strided_slice_1/beginConst*%
valueB"    ????        *
dtype0*
_output_shapes
:l
strided_slice_1/endConst*%
valueB"               *
dtype0*
_output_shapes
:p
strided_slice_1/stridesConst*%
valueB"            *
dtype0*
_output_shapes
:?
strided_slice_1StridedSliceinputsstrided_slice_1/begin:output:0strided_slice_1/end:output:0 strided_slice_1/strides:output:0*
shrink_axis_mask
*

begin_mask*
end_mask*'
_output_shapes
:?????????*
T0*
Index0`
IdentityIdentitystrided_slice_1:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
i
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_2946
inputs_0
identityL
mul_1/yConst*
valueB
 *??8*
dtype0*
_output_shapes
: Z
mul_1Mulinputs_0mul_1/y:output:0*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
?
'__inference_conv2d_1_layer_call_fn_1314

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*A
_output_shapes/
-:+???????????????????????????*+
_gradient_op_typePartitionedCall-1309*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1303?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
G__inference_RelationModel_layer_call_and_return_conditional_losses_1216	
orrra
selfloop(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCallorrraselfloop*
Tin
2*'
_output_shapes
:?????????	*+
_gradient_op_typePartitionedCall-1125*N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1118*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1149*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1143*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? ?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*'
_output_shapes
:????????? *
Tin
2*+
_gradient_op_typePartitionedCall-1177*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1171*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1204*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1198*
Tout
2?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:% !

_user_specified_nameORrRa:($
"
_user_specified_name
selfLoop: : : : : : 
?
?
B__inference_dense_10_layer_call_and_return_conditional_losses_3057

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	?i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
p
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_2492
inputs_0
identityR
ExpandDims_4/dimConst*
value	B :*
dtype0*
_output_shapes
: u
ExpandDims_4
ExpandDimsinputs_0ExpandDims_4/dim:output:0*+
_output_shapes
:?????????*
T0a
IdentityIdentityExpandDims_4:output:0*+
_output_shapes
:?????????*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
q
G__inference_concatenate_4_layer_call_and_return_conditional_losses_1880

inputs
inputs_1
identityM
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: u
concatConcatV2inputsinputs_1concat/axis:output:0*
T0*
N*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
j
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_2525
inputs_0
identityi
Tile_3/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:m
Tile_3Tileinputs_0Tile_3/multiples:output:0*
T0*/
_output_shapes
:?????????_
IdentityIdentityTile_3:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
o
E__inference_concatenate_layer_call_and_return_conditional_losses_1118

inputs
inputs_1
identityM
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: u
concatConcatV2inputsinputs_1concat/axis:output:0*
T0*
N*'
_output_shapes
:?????????	W
IdentityIdentityconcat:output:0*'
_output_shapes
:?????????	*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
h
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_1608

inputs
identityi
Tile_5/multiplesConst*
dtype0*
_output_shapes
:*%
valueB"            k
Tile_5TileinputsTile_5/multiples:output:0*
T0*/
_output_shapes
:?????????_
IdentityIdentityTile_5:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
Q
%__inference_output_layer_call_fn_2963
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_1942*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1949`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
n
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_1528

inputs
identityR
ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: w
ExpandDims_3
ExpandDimsinputsExpandDims_3/dim:output:0*
T0*/
_output_shapes
:?????????e
IdentityIdentityExpandDims_3:output:0*/
_output_shapes
:?????????*
T0"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_2922

inputs*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	?z
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:??
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????_
dense_9/EluEludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	??
dense_10/MatMulMatMuldense_9/Elu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
?"
?
__inference__traced_save_3126
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_240c8b16697b48189cef6743ab431864/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
SaveV2/shape_and_slicesConst"/device:CPU:0*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapesq
o: :::	 : :  : : ::	?:?:	?:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : :	 :
 : : : :+ '
%
_user_specified_namefile_prefix: : : 
?
j
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_2514
inputs_0
identitye
Tile_4/multiplesConst*
dtype0*
_output_shapes
:*!
valueB"         i
Tile_4Tileinputs_0Tile_4/multiples:output:0*+
_output_shapes
:?????????*
T0[
IdentityIdentityTile_4:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
p
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_2503
inputs_0
identityR
ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: y
ExpandDims_3
ExpandDimsinputs_0ExpandDims_3/dim:output:0*/
_output_shapes
:?????????*
T0e
IdentityIdentityExpandDims_3:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
?
2__inference_TimeDerivativeModel_layer_call_fn_1431
oeb"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalloebstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*+
_gradient_op_typePartitionedCall-1424*V
fQRO
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1423*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:# 

_user_specified_nameOEb: : : : 
?
?
&__inference_dense_2_layer_call_fn_3029

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1204*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1198*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
_
C__inference_permute_1_layer_call_and_return_conditional_losses_1095

inputs
identityg
transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:?
	transpose	Transposeinputstranspose/perm:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????x
IdentityIdentitytranspose:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
m
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_2870
inputs_0
identityh
	Squeeze_1Squeezeinputs_0*
squeeze_dims
*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySqueeze_1:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?	
?
,__inference_RelationModel_layer_call_fn_1262	
orrra
selfloop"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallorrraselfloopstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*+
_gradient_op_typePartitionedCall-1253*P
fKRI
G__inference_RelationModel_layer_call_and_return_conditional_losses_1252*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin

2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :% !

_user_specified_nameORrRa:($
"
_user_specified_name
selfLoop: : : : 
?
^
B__inference_SelfLoop_layer_call_and_return_conditional_losses_1077

inputs
identityc
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????k
IdentityIdentitytranspose:y:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:& "
 
_user_specified_nameinputs
?
q
E__inference_concatenate_layer_call_and_return_conditional_losses_2970
inputs_0
inputs_1
identityM
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
T0*
N*'
_output_shapes
:?????????	W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????	"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
;
ORr4
serving_default_ORr:0?????????
5
Ra/
serving_default_Ra:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
??
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer-14
layer-15
layer_with_weights-2
layer-16
layer-17
layer-18
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_modelǖ{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 4, 3, 7], "dtype": "float32", "sparse": false, "name": "ORr"}, "name": "ORr", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_1", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["ORr_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "5"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "5"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "10"}}}, "constants": {"1": [0, -1, 0, 0], "2": [0, 0, 0, 1], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_1", "inbound_nodes": [[["ORr", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 3, 7], "dtype": "float32", "sparse": false, "name": "Ra"}, "name": "Ra", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_4", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_4", "op": "ExpandDims", "input": ["strided_slice_1", "ExpandDims_4/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_ExpandDims_4", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_3", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_3", "op": "ExpandDims", "input": ["Ra_1", "ExpandDims_3/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims_3", "inbound_nodes": [[["Ra", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Tile_4", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_4", "op": "Tile", "input": ["ExpandDims_4", "Tile_4/multiples"], "attr": {"T": {"type": "DT_FLOAT"}, "Tmultiples": {"type": "DT_INT32"}}}, "constants": {"1": [1, 1, 7]}}, "name": "tf_op_layer_Tile_4", "inbound_nodes": [[["tf_op_layer_ExpandDims_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Tile_3", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_3", "op": "Tile", "input": ["ExpandDims_3", "Tile_3/multiples"], "attr": {"Tmultiples": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 4, 1, 1]}}, "name": "tf_op_layer_Tile_3", "inbound_nodes": [[["tf_op_layer_ExpandDims_3", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "SelfLoop", "trainable": true, "dtype": "float32", "dims": [2, 1]}, "name": "SelfLoop", "inbound_nodes": [[["tf_op_layer_Tile_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": 2}, "name": "concatenate_3", "inbound_nodes": [[["ORr", 0, 0, {}], ["tf_op_layer_Tile_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_5", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_5", "op": "ExpandDims", "input": ["SelfLoop_2/Identity", "ExpandDims_5/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims_5", "inbound_nodes": [[["SelfLoop", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_1", "trainable": true, "dtype": "float32", "dims": [1, 3, 2]}, "name": "permute_1", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Tile_5", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_5", "op": "Tile", "input": ["ExpandDims_5", "Tile_5/multiples"], "attr": {"Tmultiples": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 4, 1, 1]}}, "name": "tf_op_layer_Tile_5", "inbound_nodes": [[["tf_op_layer_ExpandDims_5", 0, 0, {}]]]}, {"class_name": "Model", "config": {"name": "RelationModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "ORrRa"}, "name": "ORrRa", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "selfLoop"}, "name": "selfLoop", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["ORrRa", 0, 0, {}], ["selfLoop", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["ORrRa", 0, 0], ["selfLoop", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "name": "RelationModel", "inbound_nodes": [[["permute_1", 0, 0, {}], ["tf_op_layer_Tile_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [4, 7], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["RelationModel", 1, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["conv2d_1/Identity"], "attr": {"squeeze_dims": {"list": {"i": ["1", "2"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concatenate_4", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}], ["tf_op_layer_Squeeze_1", 0, 0, {}]]]}, {"class_name": "Model", "config": {"name": "TimeDerivativeModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "OEb"}, "name": "OEb", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["OEb", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["OEb", 0, 0]], "output_layers": [["dense_10", 0, 0]]}, "name": "TimeDerivativeModel", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "mul_1", "op": "Mul", "input": ["TimeDerivativeModel_1/Identity", "mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 9.999999747378752e-05}}, "name": "tf_op_layer_mul_1", "inbound_nodes": [[["TimeDerivativeModel", 1, 0, {}]]]}, {"class_name": "Add", "config": {"name": "output", "trainable": true, "dtype": "float32"}, "name": "output", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}], ["tf_op_layer_mul_1", 0, 0, {}]]]}], "input_layers": [["ORr", 0, 0], ["Ra", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [null, null], "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 4, 3, 7], "dtype": "float32", "sparse": false, "name": "ORr"}, "name": "ORr", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_1", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["ORr_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "5"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "5"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "10"}}}, "constants": {"1": [0, -1, 0, 0], "2": [0, 0, 0, 1], "3": [1, 1, 1, 1]}}, "name": "tf_op_layer_strided_slice_1", "inbound_nodes": [[["ORr", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 3, 7], "dtype": "float32", "sparse": false, "name": "Ra"}, "name": "Ra", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_4", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_4", "op": "ExpandDims", "input": ["strided_slice_1", "ExpandDims_4/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_ExpandDims_4", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_3", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_3", "op": "ExpandDims", "input": ["Ra_1", "ExpandDims_3/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims_3", "inbound_nodes": [[["Ra", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Tile_4", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_4", "op": "Tile", "input": ["ExpandDims_4", "Tile_4/multiples"], "attr": {"T": {"type": "DT_FLOAT"}, "Tmultiples": {"type": "DT_INT32"}}}, "constants": {"1": [1, 1, 7]}}, "name": "tf_op_layer_Tile_4", "inbound_nodes": [[["tf_op_layer_ExpandDims_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Tile_3", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_3", "op": "Tile", "input": ["ExpandDims_3", "Tile_3/multiples"], "attr": {"Tmultiples": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 4, 1, 1]}}, "name": "tf_op_layer_Tile_3", "inbound_nodes": [[["tf_op_layer_ExpandDims_3", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "SelfLoop", "trainable": true, "dtype": "float32", "dims": [2, 1]}, "name": "SelfLoop", "inbound_nodes": [[["tf_op_layer_Tile_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": 2}, "name": "concatenate_3", "inbound_nodes": [[["ORr", 0, 0, {}], ["tf_op_layer_Tile_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_5", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_5", "op": "ExpandDims", "input": ["SelfLoop_2/Identity", "ExpandDims_5/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims_5", "inbound_nodes": [[["SelfLoop", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_1", "trainable": true, "dtype": "float32", "dims": [1, 3, 2]}, "name": "permute_1", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Tile_5", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_5", "op": "Tile", "input": ["ExpandDims_5", "Tile_5/multiples"], "attr": {"Tmultiples": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 4, 1, 1]}}, "name": "tf_op_layer_Tile_5", "inbound_nodes": [[["tf_op_layer_ExpandDims_5", 0, 0, {}]]]}, {"class_name": "Model", "config": {"name": "RelationModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "ORrRa"}, "name": "ORrRa", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "selfLoop"}, "name": "selfLoop", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["ORrRa", 0, 0, {}], ["selfLoop", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["ORrRa", 0, 0], ["selfLoop", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "name": "RelationModel", "inbound_nodes": [[["permute_1", 0, 0, {}], ["tf_op_layer_Tile_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [4, 7], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["RelationModel", 1, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["conv2d_1/Identity"], "attr": {"squeeze_dims": {"list": {"i": ["1", "2"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concatenate_4", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}], ["tf_op_layer_Squeeze_1", 0, 0, {}]]]}, {"class_name": "Model", "config": {"name": "TimeDerivativeModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "OEb"}, "name": "OEb", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["OEb", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["OEb", 0, 0]], "output_layers": [["dense_10", 0, 0]]}, "name": "TimeDerivativeModel", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "mul_1", "op": "Mul", "input": ["TimeDerivativeModel_1/Identity", "mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 9.999999747378752e-05}}, "name": "tf_op_layer_mul_1", "inbound_nodes": [[["TimeDerivativeModel", 1, 0, {}]]]}, {"class_name": "Add", "config": {"name": "output", "trainable": true, "dtype": "float32"}, "name": "output", "inbound_nodes": [[["tf_op_layer_strided_slice_1", 0, 0, {}], ["tf_op_layer_mul_1", 0, 0, {}]]]}], "input_layers": [["ORr", 0, 0], ["Ra", 0, 0]], "output_layers": [["output", 0, 0]]}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "ORr", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 4, 3, 7], "config": {"batch_input_shape": [null, 4, 3, 7], "dtype": "float32", "sparse": false, "name": "ORr"}}
?
	constants
regularization_losses
	variables
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "strided_slice_1", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_1", "op": "StridedSlice", "input": ["ORr_1", "strided_slice_1/begin", "strided_slice_1/end", "strided_slice_1/strides"], "attr": {"ellipsis_mask": {"i": "0"}, "begin_mask": {"i": "5"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "5"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "10"}}}, "constants": {"1": [0, -1, 0, 0], "2": [0, 0, 0, 1], "3": [1, 1, 1, 1]}}}
?
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "Ra", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 3, 7], "config": {"batch_input_shape": [null, 3, 7], "dtype": "float32", "sparse": false, "name": "Ra"}}
?
&	constants
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ExpandDims_4", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_4", "op": "ExpandDims", "input": ["strided_slice_1", "ExpandDims_4/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}}
?
+	constants
,regularization_losses
-	variables
.trainable_variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ExpandDims_3", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_3", "op": "ExpandDims", "input": ["Ra_1", "ExpandDims_3/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
?
0	constants
1regularization_losses
2	variables
3trainable_variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Tile_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Tile_4", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_4", "op": "Tile", "input": ["ExpandDims_4", "Tile_4/multiples"], "attr": {"T": {"type": "DT_FLOAT"}, "Tmultiples": {"type": "DT_INT32"}}}, "constants": {"1": [1, 1, 7]}}}
?
5	constants
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Tile_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Tile_3", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_3", "op": "Tile", "input": ["ExpandDims_3", "Tile_3/multiples"], "attr": {"Tmultiples": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 4, 1, 1]}}}
?
:regularization_losses
;	variables
<trainable_variables
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Permute", "name": "SelfLoop", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "SelfLoop", "trainable": true, "dtype": "float32", "dims": [2, 1]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": 2}}
?
B	constants
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ExpandDims_5", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_5", "op": "ExpandDims", "input": ["SelfLoop_2/Identity", "ExpandDims_5/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 1}}}
?
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Permute", "name": "permute_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "permute_1", "trainable": true, "dtype": "float32", "dims": [1, 3, 2]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
K	constants
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Tile_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Tile_5", "trainable": true, "dtype": "float32", "node_def": {"name": "Tile_5", "op": "Tile", "input": ["ExpandDims_5", "Tile_5/multiples"], "attr": {"Tmultiples": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 4, 1, 1]}}}
?&
Player-0
Qlayer-1
Rlayer-2
Slayer_with_weights-0
Slayer-3
Tlayer_with_weights-1
Tlayer-4
Ulayer_with_weights-2
Ulayer-5
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?#
_tf_keras_model?#{"class_name": "Model", "name": "RelationModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RelationModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "ORrRa"}, "name": "ORrRa", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "selfLoop"}, "name": "selfLoop", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["ORrRa", 0, 0, {}], ["selfLoop", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["ORrRa", 0, 0], ["selfLoop", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "input_spec": [null, null], "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "RelationModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "ORrRa"}, "name": "ORrRa", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "selfLoop"}, "name": "selfLoop", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["ORrRa", 0, 0, {}], ["selfLoop", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["ORrRa", 0, 0], ["selfLoop", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
?

Zkernel
[bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [4, 7], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
?
`	constants
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Squeeze_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["conv2d_1/Identity"], "attr": {"squeeze_dims": {"list": {"i": ["1", "2"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": 1}}
?
ilayer-0
jlayer_with_weights-0
jlayer-1
klayer_with_weights-1
klayer-2
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"class_name": "Model", "name": "TimeDerivativeModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "TimeDerivativeModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "OEb"}, "name": "OEb", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["OEb", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["OEb", 0, 0]], "output_layers": [["dense_10", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "TimeDerivativeModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "OEb"}, "name": "OEb", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["OEb", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["OEb", 0, 0]], "output_layers": [["dense_10", 0, 0]]}}}
?
p	constants
qregularization_losses
r	variables
strainable_variables
t	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_mul_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "mul_1", "op": "Mul", "input": ["TimeDerivativeModel_1/Identity", "mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 9.999999747378752e-05}}}
?
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32"}}
 "
trackable_list_wrapper
y
y0
z1
{2
|3
}4
~5
Z6
[7
8
?9
?10
?11"
trackable_list_wrapper
y
y0
z1
{2
|3
}4
~5
Z6
[7
8
?9
?10
?11"
trackable_list_wrapper
?
 ?layer_regularization_losses
regularization_losses
?metrics
?layers
	variables
trainable_variables
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
 ?layer_regularization_losses
?metrics
?layers
	variables
trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
 ?layer_regularization_losses
?metrics
?layers
	variables
 trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"regularization_losses
 ?layer_regularization_losses
?metrics
?layers
#	variables
$trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'regularization_losses
 ?layer_regularization_losses
?metrics
?layers
(	variables
)trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,regularization_losses
 ?layer_regularization_losses
?metrics
?layers
-	variables
.trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1regularization_losses
 ?layer_regularization_losses
?metrics
?layers
2	variables
3trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
6regularization_losses
 ?layer_regularization_losses
?metrics
?layers
7	variables
8trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
:regularization_losses
 ?layer_regularization_losses
?metrics
?layers
;	variables
<trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
>regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
@trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cregularization_losses
 ?layer_regularization_losses
?metrics
?layers
D	variables
Etrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gregularization_losses
 ?layer_regularization_losses
?metrics
?layers
H	variables
Itrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lregularization_losses
 ?layer_regularization_losses
?metrics
?layers
M	variables
Ntrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "ORrRa", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 6], "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "ORrRa"}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "selfLoop", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 3], "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "selfLoop"}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}}
?

ykernel
zbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}}
?

{kernel
|bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
?

}kernel
~bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
 "
trackable_list_wrapper
J
y0
z1
{2
|3
}4
~5"
trackable_list_wrapper
J
y0
z1
{2
|3
}4
~5"
trackable_list_wrapper
?
 ?layer_regularization_losses
Vregularization_losses
?metrics
?layers
W	variables
Xtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
?
\regularization_losses
 ?layer_regularization_losses
?metrics
?layers
]	variables
^trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
aregularization_losses
 ?layer_regularization_losses
?metrics
?layers
b	variables
ctrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
eregularization_losses
 ?layer_regularization_losses
?metrics
?layers
f	variables
gtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "OEb", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 6], "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "OEb"}}
?

kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
 "
trackable_list_wrapper
?
0
?1
?2
?3"
trackable_list_wrapper
?
0
?1
?2
?3"
trackable_list_wrapper
?
 ?layer_regularization_losses
lregularization_losses
?metrics
?layers
m	variables
ntrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
qregularization_losses
 ?layer_regularization_losses
?metrics
?layers
r	variables
strainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
uregularization_losses
 ?layer_regularization_losses
?metrics
?layers
v	variables
wtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 2dense/kernel
: 2
dense/bias
 :  2dense_1/kernel
: 2dense_1/bias
 : 2dense_2/kernel
:2dense_2/bias
!:	?2dense_9/kernel
:?2dense_9/bias
": 	?2dense_10/kernel
:2dense_10/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
P0
Q1
R2
S3
T4
U5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?	variables
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
i0
j1
k2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
A__inference_model_1_layer_call_and_return_conditional_losses_1994
A__inference_model_1_layer_call_and_return_conditional_losses_2419
A__inference_model_1_layer_call_and_return_conditional_losses_2273
A__inference_model_1_layer_call_and_return_conditional_losses_1957?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_1068?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *Q?N
L?I
%?"
ORr?????????
 ?
Ra?????????
?2?
&__inference_model_1_layer_call_fn_2437
&__inference_model_1_layer_call_fn_2049
&__inference_model_1_layer_call_fn_2455
&__inference_model_1_layer_call_fn_2105?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_2463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_2468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
<__inference_Ra_layer_call_and_return_conditional_losses_2472
<__inference_Ra_layer_call_and_return_conditional_losses_2476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
!__inference_Ra_layer_call_fn_2481
!__inference_Ra_layer_call_fn_2486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_2492?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_tf_op_layer_ExpandDims_4_layer_call_fn_2497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_2503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_tf_op_layer_ExpandDims_3_layer_call_fn_2508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_2514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_tf_op_layer_Tile_4_layer_call_fn_2519?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_2525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_tf_op_layer_Tile_3_layer_call_fn_2530?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_SelfLoop_layer_call_and_return_conditional_losses_1077?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
'__inference_SelfLoop_layer_call_fn_1086?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
G__inference_concatenate_3_layer_call_and_return_conditional_losses_2537?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_3_layer_call_fn_2543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_2549?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_tf_op_layer_ExpandDims_5_layer_call_fn_2554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_permute_1_layer_call_and_return_conditional_losses_1095?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_permute_1_layer_call_fn_1104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_2560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_tf_op_layer_Tile_5_layer_call_fn_2565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_RelationModel_layer_call_and_return_conditional_losses_2664
G__inference_RelationModel_layer_call_and_return_conditional_losses_2814
G__inference_RelationModel_layer_call_and_return_conditional_losses_2841
G__inference_RelationModel_layer_call_and_return_conditional_losses_1233
G__inference_RelationModel_layer_call_and_return_conditional_losses_1216
G__inference_RelationModel_layer_call_and_return_conditional_losses_2763?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_RelationModel_layer_call_fn_1262
,__inference_RelationModel_layer_call_fn_2775
,__inference_RelationModel_layer_call_fn_2787
,__inference_RelationModel_layer_call_fn_1292
,__inference_RelationModel_layer_call_fn_2865
,__inference_RelationModel_layer_call_fn_2853?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
'__inference_conv2d_1_layer_call_fn_1314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_2870?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_2875?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_4_layer_call_and_return_conditional_losses_2882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_4_layer_call_fn_2888?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_2922
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1376
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_2905
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1388?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_TimeDerivativeModel_layer_call_fn_2931
2__inference_TimeDerivativeModel_layer_call_fn_2940
2__inference_TimeDerivativeModel_layer_call_fn_1409
2__inference_TimeDerivativeModel_layer_call_fn_1431?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_2946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_tf_op_layer_mul_1_layer_call_fn_2951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_output_layer_call_and_return_conditional_losses_2957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_output_layer_call_fn_2963?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
/B-
"__inference_signature_wrapper_2125ORrRa
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
E__inference_concatenate_layer_call_and_return_conditional_losses_2970?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_concatenate_layer_call_fn_2976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_2987?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_2994?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_3005?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_3012?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_3022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_3029?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
A__inference_dense_9_layer_call_and_return_conditional_losses_3040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_9_layer_call_fn_3047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_10_layer_call_and_return_conditional_losses_3057?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_10_layer_call_fn_3064?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
,__inference_RelationModel_layer_call_fn_2775?yz{|}~r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p

 
? " ???????????
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1303?Z[I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
A__inference_dense_2_layer_call_and_return_conditional_losses_3022\}~/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? ?
7__inference_tf_op_layer_ExpandDims_4_layer_call_fn_2497V6?3
,?)
'?$
"?
inputs/0?????????
? "???????????
,__inference_RelationModel_layer_call_fn_2787?yz{|}~r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p 

 
? " ???????????
B__inference_dense_10_layer_call_and_return_conditional_losses_3057_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_2273?yz{|}~Z[???n?k
d?a
W?T
*?'
inputs/0?????????
&?#
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_concatenate_layer_call_fn_2976vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????	?
,__inference_RelationModel_layer_call_fn_1292?yz{|}~_?\
U?R
H?E
?
ORrRa?????????
"?
selfLoop?????????
p 

 
? "???????????
,__inference_RelationModel_layer_call_fn_2853?yz{|}~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_2987\yz/?,
%?"
 ?
inputs?????????	
? "%?"
?
0????????? 
? ?
0__inference_tf_op_layer_mul_1_layer_call_fn_2951R6?3
,?)
'?$
"?
inputs/0?????????
? "???????????
!__inference_Ra_layer_call_fn_2481rJ?G
0?-
+?(
&?#
inputs/0?????????
?

trainingp"$?!
?
0??????????
B__inference_SelfLoop_layer_call_and_return_conditional_losses_1077?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
"__inference_signature_wrapper_2125?yz{|}~Z[???c?`
? 
Y?V
&
Ra ?
Ra?????????
,
ORr%?"
ORr?????????"/?,
*
output ?
output??????????
!__inference_Ra_layer_call_fn_2486rJ?G
0?-
+?(
&?#
inputs/0?????????
?

trainingp "$?!
?
0??????????
L__inference_tf_op_layer_Tile_5_layer_call_and_return_conditional_losses_2560o>?;
4?1
/?,
*?'
inputs/0?????????
? "-?*
#? 
0?????????
? ?
A__inference_dense_1_layer_call_and_return_conditional_losses_3005\{|/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ?
,__inference_RelationModel_layer_call_fn_2865?yz{|}~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
L__inference_tf_op_layer_Tile_4_layer_call_and_return_conditional_losses_2514g:?7
0?-
+?(
&?#
inputs/0?????????
? ")?&
?
0?????????
? ?
&__inference_model_1_layer_call_fn_2049?yz{|}~Z[???c?`
Y?V
L?I
%?"
ORr?????????
 ?
Ra?????????
p

 
? "???????????
&__inference_model_1_layer_call_fn_2105?yz{|}~Z[???c?`
Y?V
L?I
%?"
ORr?????????
 ?
Ra?????????
p 

 
? "??????????}
'__inference_dense_10_layer_call_fn_3064R??0?-
&?#
!?
inputs??????????
? "???????????
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_2905i???7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
'__inference_conv2d_1_layer_call_fn_1314?Z[I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
E__inference_concatenate_layer_call_and_return_conditional_losses_2970?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????	
? ?
%__inference_output_layer_call_fn_2963vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
1__inference_tf_op_layer_Tile_3_layer_call_fn_2530b>?;
4?1
/?,
*?'
inputs/0?????????
? " ???????????
R__inference_tf_op_layer_ExpandDims_5_layer_call_and_return_conditional_losses_2549k:?7
0?-
+?(
&?#
inputs/0?????????
? "-?*
#? 
0?????????
? ?
G__inference_RelationModel_layer_call_and_return_conditional_losses_2664?yz{|}~r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p

 
? "-?*
#? 
0?????????
? ?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_2922i???7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_2875Z>?;
4?1
/?,
*?'
inputs/0?????????
? "???????????
A__inference_model_1_layer_call_and_return_conditional_losses_2419?yz{|}~Z[???n?k
d?a
W?T
*?'
inputs/0?????????
&?#
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_RelationModel_layer_call_and_return_conditional_losses_1216?yz{|}~_?\
U?R
H?E
?
ORrRa?????????
"?
selfLoop?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_concatenate_4_layer_call_and_return_conditional_losses_2882?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1376f???4?1
*?'
?
OEb?????????
p

 
? "%?"
?
0?????????
? ?
1__inference_tf_op_layer_Tile_5_layer_call_fn_2565b>?;
4?1
/?,
*?'
inputs/0?????????
? " ???????????
G__inference_RelationModel_layer_call_and_return_conditional_losses_1233?yz{|}~_?\
U?R
H?E
?
ORrRa?????????
"?
selfLoop?????????
p 

 
? "%?"
?
0?????????
? ?
R__inference_tf_op_layer_ExpandDims_4_layer_call_and_return_conditional_losses_2492c6?3
,?)
'?$
"?
inputs/0?????????
? ")?&
?
0?????????
? ?
L__inference_tf_op_layer_Tile_3_layer_call_and_return_conditional_losses_2525o>?;
4?1
/?,
*?'
inputs/0?????????
? "-?*
#? 
0?????????
? ?
M__inference_TimeDerivativeModel_layer_call_and_return_conditional_losses_1388f???4?1
*?'
?
OEb?????????
p 

 
? "%?"
?
0?????????
? ?
R__inference_tf_op_layer_ExpandDims_3_layer_call_and_return_conditional_losses_2503k:?7
0?-
+?(
&?#
inputs/0?????????
? "-?*
#? 
0?????????
? ?
7__inference_tf_op_layer_ExpandDims_3_layer_call_fn_2508^:?7
0?-
+?(
&?#
inputs/0?????????
? " ???????????
A__inference_model_1_layer_call_and_return_conditional_losses_1957?yz{|}~Z[???c?`
Y?V
L?I
%?"
ORr?????????
 ?
Ra?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_RelationModel_layer_call_and_return_conditional_losses_2763?yz{|}~r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p 

 
? "-?*
#? 
0?????????
? ?
G__inference_RelationModel_layer_call_and_return_conditional_losses_2814?yz{|}~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? y
&__inference_dense_2_layer_call_fn_3029O}~/?,
%?"
 ?
inputs????????? 
? "???????????
7__inference_tf_op_layer_ExpandDims_5_layer_call_fn_2554^:?7
0?-
+?(
&?#
inputs/0?????????
? " ???????????
,__inference_concatenate_3_layer_call_fn_2543?j?g
`?]
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
? " ???????????
G__inference_RelationModel_layer_call_and_return_conditional_losses_2841?yz{|}~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_1994?yz{|}~Z[???c?`
Y?V
L?I
%?"
ORr?????????
 ?
Ra?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_output_layer_call_and_return_conditional_losses_2957?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
__inference__wrapped_model_1068?yz{|}~Z[???[?X
Q?N
L?I
%?"
ORr?????????
 ?
Ra?????????
? "/?,
*
output ?
output??????????
(__inference_permute_1_layer_call_fn_1104?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
2__inference_TimeDerivativeModel_layer_call_fn_1409Y???4?1
*?'
?
OEb?????????
p

 
? "???????????
2__inference_TimeDerivativeModel_layer_call_fn_2931\???7?4
-?*
 ?
inputs?????????
p

 
? "???????????
2__inference_TimeDerivativeModel_layer_call_fn_2940\???7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
2__inference_TimeDerivativeModel_layer_call_fn_1431Y???4?1
*?'
?
OEb?????????
p 

 
? "???????????
U__inference_tf_op_layer_strided_slice_1_layer_call_and_return_conditional_losses_2463g>?;
4?1
/?,
*?'
inputs/0?????????
? "%?"
?
0?????????
? ?
1__inference_tf_op_layer_Tile_4_layer_call_fn_2519Z:?7
0?-
+?(
&?#
inputs/0?????????
? "??????????w
$__inference_dense_layer_call_fn_2994Oyz/?,
%?"
 ?
inputs?????????	
? "?????????? ?
G__inference_concatenate_3_layer_call_and_return_conditional_losses_2537?j?g
`?]
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
? "-?*
#? 
0?????????
? ?
,__inference_concatenate_4_layer_call_fn_2888vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
'__inference_SelfLoop_layer_call_fn_1086wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
K__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_2946_6?3
,?)
'?$
"?
inputs/0?????????
? "%?"
?
0?????????
? ?
C__inference_permute_1_layer_call_and_return_conditional_losses_1095?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
A__inference_dense_9_layer_call_and_return_conditional_losses_3040^?/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ?
<__inference_Ra_layer_call_and_return_conditional_losses_2472~J?G
0?-
+?(
&?#
inputs/0?????????
?

trainingp"0?-
&?#
!?
0/0?????????
? ?
<__inference_Ra_layer_call_and_return_conditional_losses_2476~J?G
0?-
+?(
&?#
inputs/0?????????
?

trainingp "0?-
&?#
!?
0/0?????????
? {
&__inference_dense_9_layer_call_fn_3047Q?/?,
%?"
 ?
inputs?????????
? "???????????y
&__inference_dense_1_layer_call_fn_3012O{|/?,
%?"
 ?
inputs????????? 
? "?????????? ?
&__inference_model_1_layer_call_fn_2437?yz{|}~Z[???n?k
d?a
W?T
*?'
inputs/0?????????
&?#
inputs/1?????????
p

 
? "???????????
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_2870g>?;
4?1
/?,
*?'
inputs/0?????????
? "%?"
?
0?????????
? ?
:__inference_tf_op_layer_strided_slice_1_layer_call_fn_2468Z>?;
4?1
/?,
*?'
inputs/0?????????
? "???????????
&__inference_model_1_layer_call_fn_2455?yz{|}~Z[???n?k
d?a
W?T
*?'
inputs/0?????????
&?#
inputs/1?????????
p 

 
? "???????????
,__inference_RelationModel_layer_call_fn_1262?yz{|}~_?\
U?R
H?E
?
ORrRa?????????
"?
selfLoop?????????
p

 
? "??????????