# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:36:01 2018

@author: zhyno
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
#%matplotlib qt5  --Instructions for displaying Matplotlib image frames separately in Spyder (can also be set directly in “Tools”)
# -------------------Data loading----------------------
tf.reset_default_graph() #reset tensorflow graph
fname =open('Train dataset.csv',encoding='UTF-8')
df = pd.read_csv(fname)

#depth = np.array(df['Depth(m)'])
tmp=np.array(df['Temp(degC)'])
#cond=np.array(df['Cond(mS/cm)'])
salinity=np.array(df['Salinity(PSU)'])
#chl=np.array(df['Chl-Flu.(ppb)'])
#turb=np.array(df['Turb(NTU)'])
ph=np.array(df['pH'])
do=np.array(df['DO(mg/l)'])
#tf.reset_default_graph()   #Reset tensorflow graph
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"   #Assign which gpu or cpu to be used
#tf.device('/gpu:2')   #Another way to assign gpu or cpu
# -------------------Moving average function for pretreatment----------------------
def moving_average(l, N):
    sum = 0
    result = list(0 for x in l)

    for i in range(0, N):
        sum = sum + l[i]
        result[i] = sum / (i + 1)

    for i in range(N, len(l)):
        sum = sum - l[i - N] + l[i]
        result[i] = sum / N

    return result

new_tmp = moving_average(tmp.tolist(), 3)
#new_cond = moving_average(cond.tolist(), 4)
new_salinity = moving_average(salinity.tolist(), 4)
new_do = moving_average(do.tolist(), 4)
#new_depth = depth.tolist()
#new_chl = chl.tolist()
new_ph = ph.tolist()

data=[new_ph,new_tmp,new_do]  #Combine arrays into one matrix

normalize_data=(data-np.mean(data))/np.std(data)   #Data standardization
#normalize_data=normalize_data[:,np.newaxis]

time_step=20      #Time step
rnn_unit=5        #Hidden layer units
batch_size=60     #Batch size
input_size=3      #Input size
output_size=1     #Output size
lr=0.00005        #Learning rate
loss_a=[]         #Undefined loos function

def dataset(datas):
    train_x= []
    for i in range(len(datas)-time_step-1):
        x=datas[i:i+time_step]
        train_x.append(x.tolist())
    return train_x

train_x1=dataset(normalize_data[0])
train_x2=dataset(normalize_data[1])
train_x3=dataset(normalize_data[2])
train_Y=dataset(normalize_data[2])

train_XT=[train_x1,train_x2,train_x3]
#The input data can be weighted according to the correlation coefficient.
train_X=[]
for j in range(0,len(train_XT[0])):
    tem=[]
    for i in range(0,len(train_XT)):
        tem.append(train_XT[i][j])
    train_X.append(tem)

#——————————————————Define the neural network variables——————————————————
X=tf.placeholder(tf.float32)    #Tensor entered into the network in each batch
Y=tf.placeholder(tf.float32)   #Labels corresponding to tensor for each batch
Z=tf.add(X,Y)
with tf.Session() as sess:
    sess.run(Z,feed_dict={X:[None,input_size,time_step],Y:[None,time_step,None]})




#Input layer、the weights of output layer、biases
weights={                                                 #Define the weights
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])), #Random real number: input_size line, rnn_unit row
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={                                                 #Define the biases
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }


#——————————————————Define the neural network variables——————————————————

def lstm(batch):      #Formal parameter：Number of batches entered network
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  
#Tenor needs to be converted into two-dimensional to conduct computation, and the calculated results are used as input of the hidden layer.
    #print(input)
    input_rnn=tf.matmul(input,w_in)+b_in
    #print(input_rnn)
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  
#Tenor needs to be converted into three-dimensional to conduct computation, and the calculated results are used as input of the lstm cell.
    #print(input_rnn)

    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)

    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
    #Output_rnn is used to record the result of each output node of lstm, and final_states is the result of the last cell. 
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #The input of output layer

    w_out=weights['out']
    b_out=biases['out']

    pred=tf.matmul(output,w_out)+b_out
    #print("pred",pred)

    return pred,final_states

timeperstep=[]

#——————————————————Model training——————————————————
timeall_start=time.time()
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)

    #Loss function
    loss=tf.reduce_mean(abs(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))  #MAE
    #loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1]))) #RMSE
    #loss=tf.reduce_mean(abs((tf.reshape(pred,[-1])-tf.reshape(Y, [-1]))/tf.reshape(Y, [-1])))  #MAPE

    
    train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #Set the training times
        step=0
        print("lstm train start:")
        for i in range(300):
            
            start=0
            end=start+batch_size
            time_start = time.time()
            while(end<len(train_X)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_X[start:end],Y:train_Y[start:end]})
                loss_a.append(loss_)
                start+=batch_size
                end=start+batch_size
                #save parameters every 10 steps
            if step%10==0:
                print(i,loss_)
                print("save the lstm model into",saver.save(sess,'model/maestock.model'))
            step+=1
            time_end = time.time()
            timeperstep.append(time_end - time_start)
            #pred.eval(session=tf.Session())
            #print(pred)
timeall_stop=time.time()
alltime=timeall_stop-timeall_start   ##Timer
with tf.variable_scope('train'):
 starttime = time.clock() 
 train_lstm()
 endtime = time.clock()
 print('Running time:%s Seconds'%(endtime-starttime))

plt.plot( loss_a, color='green', label='loss')
plt.legend(loc=0)
dataframe=pd.DataFrame({'Deviation':loss_a})
dataframe.to_csv("../lstmloss-5-300-mae.csv",index=False,sep=',')
timeframe=pd.DataFrame({'timeperstep':timeperstep,'timeall':alltime})
timeframe.to_csv("../lstmtime-5-300-mae.csv",index=False,sep=',')
plt.show()



#fname1 =open('Test dataset.csv',encoding='UTF-8')
#df1 = pd.read_csv(fname1)
#timeperstep1=depth = np.array(df1['DO(mg/l)'])
#timeperstep1=moving_average(timeperstep1.tolist(), 3)
#timeperstep1=(timeperstep1-np.mean(timeperstep1))/np.std(timeperstep1)
#————————————————Prediction————————————————————
def prediction():
    pred,_=lstm(1)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #Recover parameters
        saver.restore(sess, 'model/maestock.model')

        #Take the last line of train dataset as test sample. #shape=[1,time_step,input_size]
        prev_seq=train_X[0:batch_size]

        predict=[]
        #Set the amount of predicted data 
        for i in range(3000):
            next_seq=sess.run(pred,feed_dict={X:prev_seq})
            #print(next_seq)
            predict.append(next_seq[-1][0])
#Each time we get the prediction results of last time step, we add them up with the previous data to form a new test sample.
            #prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
            prev_seq=train_X[1+i:batch_size+i+1]
        #plt.figure()
        #plt.plot( normalize_data[-1], color='black')
        #plt.plot( predict, color='r')
        #plt.plot(timeperstep1, color='b')
        dataframe = pd.DataFrame({'predict': predict})
        dataframe.to_csv("../lstmpre-5-300-mae.csv", index=False, sep=',')
        #print(predict)
        plt.show()
with tf.variable_scope('train',reuse=True):
 
 prediction()

#plt.figure()
#plt.plot(timeperstep1, color='b')
#plt.show()
#sub_axix = filter(lambda x:x%200 == 0, depth)

#plt.title('Result Analysis')

