import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mean_01 = np.array([1,0.5])
cov_01 = np.array([[1,0.1],[0.1,1.2]])

mean_02 = np.array([4,5])
cov_02 = np.array([[1.21,0.1],[0.1,1.3]])


# Normal Distribution
dist_01 = np.random.multivariate_normal(mean_01,cov_01,500)
dist_02 = np.random.multivariate_normal(mean_02,cov_02,500)

data = np.zeros((1000,3))
print(data.shape)
split = int(0.8*data.shape[0])
data[:500,:2] = dist_01
data[500:,:2] = dist_02
data[500:,-1] = 1.0
X_train = data[:split,:-1]
X_test = data[split:,:-1]

Y_train = data[:split,-1]
Y_test  = data[split:,-1]
W = 2*np.random.random((X_train.shape[1],))

def hypothesis(x,w,b):
    '''accepts input vector x, input weight vector w and bias b'''   
    h = np.dot(x,w) + b
    return sigmoid(h)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-1.0*z))

def error(y_true,x,w,b):
    m = x.shape[0]
    err = 0.0
    for i in range(m):
        hx = hypothesis(x[i],w,b) 
        err += y_true[i]*np.log2(hx) + (1-y_true[i])*np.log2(1-hx)
    return -err/m
def get_grads(y_true,x,w,b):
    
    grad_w = np.zeros(w.shape)
    grad_b = 0.0
    m = x.shape[0]
    for i in range(m):
        hx = hypothesis(x[i],w,b)
        
        grad_w += (y_true[i] - hx)*x[i]
        grad_b +=  (y_true[i]-hx)
    grad_w /= m
    grad_b /= m
    return [grad_w,grad_b]

# One Iteration of Gradient Descent
def grad_descent(x,y_true,w,b,learning_rate=0.1):
    [grad_w,grad_b] = get_grads(y_true,x,w,b)
    w = w + learning_rate*grad_w
    b = b + learning_rate*grad_b
    return err,w,b
def predict(x,w,b):
    
    confidence = hypothesis(x,w,b)
    if confidence<0.5:
        return 0
    else:
        return 1
    
def get_acc(x_tst,y_tst,w,b):
    
    y_pred = []
    
    for i in range(y_tst.shape[0]):
        p = predict(x_tst[i],w,b)
        y_pred.append(p)
        
    y_pred = np.array(y_pred)
    
    return  float((y_pred==y_tst).sum())/y_tst.shape[0]

W = 2*np.random.random((X_train.shape[1],))
b = 5*np.random.random()
def find_gradient(theta,X,Y):
    grad_theta=[0,0]
    grad_const=0
    for i in range(len(X)):
        grad_theta+=(-hypothesis(X[i],theta,grad_const)+Y[i])*X[i]
        grad_const+=(-hypothesis(X[i],theta,grad_const)+Y[i])
    grad_theta=grad_theta/(len(X))
    grad_const=grad_const/len(X)
    return grad_theta,grad_const    
def gradient_descent(X,Y,learning_rate,n_iterations=100):
    theta=np.random.random((2,))
    b=0
    for i in range(n_iterations):
        grad_theta,grad_const=find_gradient(theta,X,Y)
        theta=theta+learning_rate*grad_theta
        b=b+learning_rate*grad_const
    return theta,b
for i in range(1000):
    l,W,b = grad_descent(X_train,Y_train,W,b,learning_rate=0.1)
    #acc.append(get_acc(X_test,Y_test,W,b))
    #loss.append(l)
plt.figure(0)
theta,b=gradient_descent(X_train,Y_train,0.1,1000)
plt.scatter(dist_01[:,0],dist_01[:,1],label='Class 0')
plt.scatter(dist_02[:,0],dist_02[:,1],color='r',marker='^',label='Class 1')
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.xlabel('x1')
plt.ylabel('x2')

x = np.linspace(-4,8,10)
y = -(W[0]*x + b)/W[1]
plt.plot(x,y,color='k')

plt.legend()
plt.show()    