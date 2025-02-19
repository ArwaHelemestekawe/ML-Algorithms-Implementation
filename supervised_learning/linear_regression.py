import numpy as np
import matplotlib.pyplot as plt
def gety(b0,b1 ,x):
    n = len(x)
    y_hat = np.zeros(n)  
    for i in range (n):
      y_hat[i]= b0+(b1*x[i])
    return y_hat

def sum_squar(y,y_hat):
   error =0
   n= len(y)
   for i in range (n):
      error +=(y[i]-y_hat[i])**2
   return error

def get_b1(x,y):
   x_bar =np.mean(x)
   y_bar = np.mean(y)
   frist_sum =0
   second_sum =0
   n=len(x)
   for i in range (n):
      frist_sum +=x[i]*(y[i]- y_bar)
   for i in range (n):
      second_sum +=x[i]*(x[i]-x_bar)
   return frist_sum /second_sum

def get_b0(x,y,b1):
   x_bar= np.mean(x)
   y_bar= np.mean(y)
   b0= y_bar-(b1*x_bar)
   return b0
       

 
