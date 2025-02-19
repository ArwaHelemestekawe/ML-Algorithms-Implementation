from itertools import combinations_with_replacement
import numpy as np
class PolynomialF:
    def __init__(self,degree=2):
        self.degree = degree

    def transform(self,x):
        x= np.array(x) 
         # this line to make sure that the x is array  to apply multiplication in row and coloumns (combinations )
        n_sampels,n_features=x.shape
        polyfeatures=np.ones((n_sampels,1))
        # this is the bias term  ,  this line allows me to put it in matrix not in vector
        # 5 houses every house has 3 feature
        # x is the  data set with ex 2features
        for d in range(1,self.degree+1):
            for i in combinations_with_replacement(range(n_features),d):
                            # for i in combinations_with_replacement([0,1],2):
                            # print(i)
                            # the out put will be (0,0)(0,1)(1,1)
                 new_features = np.prod(x[:,i],axis=1,keepdims=True)
                 polyfeatures=np.c_[polyfeatures,new_features]
        return polyfeatures

class   min_max_scaler():
    def __init__(self):
        self.min_val = None
        self.max_val=None
    def fit(self,x):
        self.min_val=x.min(axis=0)
        self.max_val=x.max(axis=0)
        #minimum and maximum are calculated for each feature (column) independently
    def transform_to_min_max(self,x):
        if self.min_val is None or self.max_val is None:
            raise ValueError("fit frist")
        x_normalized = (x-self.min_val)/(self.max_val-self.min_val)
        return x_normalized
    
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)
    '''
    scaler = min_max_scaler()  # Create a new scaler
    scaler.fit(x_train)  # Fit the scaler to the training data
    x_train_scaled = scaler.transform(x_train)  # Scale the training data
    x_test_scaled = scaler.transform(x_test)  # Scale the test data using the same scaler

    '''



class   standard_scaler():
    def __init__(self):
        self.mean= None
        self.std_val=None
    def fit(self,x):
        self.mean=np.mean(x,axis=0)
        self.std_val=np.std(axis=0)
    def transform(self,x):
        if self.mean is None or self.std_val is None:
            raise ValueError("fit frist")
        x_standrized = (x-self.mean)/(self.std_val)
        return x_standrized
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)


