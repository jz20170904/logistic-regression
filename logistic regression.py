
#逻辑逻辑回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def todata():
    path='data.txt'#数据地址
    data=pd.read_csv(path,header=None,names=['DX','DY','Node_0_1'])
    return data
    # print(data.head())

#显示图形
def show_pc():
    data=todata()
    positive=data[data['Node_0_1'].isin([1])]
    negative=data[data['Node_0_1'].isin([0])]
    fig,ax=plt.subplots(figsize=(15,10))#绘制散点图
    ax.scatter(positive['DX'],positive['DY'],s=50,c='b',marker='o',label='Node_1')
    ax.scatter(negative['DX'],negative['DY'],s=50,c='r',marker='x',label='Node_0')
    ax.legend()
    ax.set_xlabel('DX Score')
    ax.set_ylabel('DY Score')
    plt.show()

#定义假设函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#绘制假设函数
def show_z():
    nums=np.arange(-15,15,step=1)
    fig,ax=plt.subplots(figsize=(15,10))
    ax.plot(nums,sigmoid(nums),'r')
    # plt.show()



#对数据做一些处理
def taking_data():
    data=todata()
    data.insert(0,'ones',1)
    cols=data.shape[1]
    X1=data.iloc[:,0:cols-1]
    y1=data.iloc[:,cols-1:cols]

    X=np.array(X1.values)
    y=np.array(y1.values)
    theta=np.zeros(3)
    return X,y,theta

#代价函数
def cost(theta,X,y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    return np.sum(first-second)/len(X)

#梯度下降
def gradients(theta,X,y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)

    parameters=int(theta.ravel().shape[1])
    grads_n=np.zeros(parameters)

    error=sigmoid(X*theta.T)-y

    for i in range(parameters):
        term=np.multiply(error,X[:,i])
        grads_n[i]=np.sum(term)/len(X)
    return grads_n




#寻找最优参数
def result():
    X,y,theta=taking_data()
    cost(theta,X,y)
    gradients(theta,X,y)
    result=opt.fmin_tnc(func=cost,x0=theta,fprime=gradients,args=(X,y))
    return result


#计算分类器精度
def predict(theta,X):
    probability=sigmoid(X*theta.T)
    return [1 if x >=0.5 else 0 for x in probability]

if __name__=='__main__':
    show_pc()
    # show_z()
    X,y,theta=taking_data()
    result=result()
    theta_min=np.matrix(result[0])
    predictions=predict(theta_min,X)
    correct=[1 if ((a==1 and b==1) or (a==0 and b==0)) else 0 for (a,b) in zip(predictions,y)]
    accuray=(sum(map(int,correct)) % len(correct))
    print ('accuracy = {0}%'.format(accuray))

