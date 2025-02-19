import numpy as np 
import os 
os.system("cls")
from sklearn.tree import DecisionTreeRegressor

class Gradient_boost:
    def __init__(self,number_of_estimators=100,learning_rate=0.1,max_depth=2):
        self.number_of_estimators = number_of_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.max_depth = max_depth
        self.y=None

    def fit(self,x,y):
        y_initial_mean=np.mean(y)
        #هنا انا جبت قيمة مفردة بس انا عاوز يبقا فيكتور الصفوف بتاعته بتساوي عدد السامبلز والاعمدة بتساوي واحد فهخليه ياخد نفس شكل  ال y الاصلية 
        y_initial_vector=np.ones_like(y)*y_initial_mean

        for i in range(self.number_of_estimators):
            Error=y-y_initial_vector
            model=DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(x,Error)
            predicted_error=model.predict(x)
            y_initial_vector+=(self.learning_rate*predicted_error)
        #  fit uses model.predict to iteratively improve the model during training,
        #  while predict uses the trained models to make new predictions.
            self.models.append(model)
    
    def predict(self,x):
        y_hat=np.mean(self.y)
        for model in self.models:
            y_hat+=self.learning_rate*model.predict(x)
        
        return y_hat
    




'''
x = np.array([[1], [2], [3], [4]])  # 4 samples, 1 feature
y = np.array([2, 4, 5, 8])  # Target values

# Assume we have these parameters
number_of_estimators = 2
learning_rate = 0.1
max_depth = 2
```

Let's trace the `fit` function first:

1. `y_initial_mean = np.mean(y)`
   * Calculates mean of y: (2 + 4 + 5 + 8) / 4 = 4.75

2. `y_initial_vector = np.ones_like(y) * y_initial_mean`
   * Creates vector: [4.75, 4.75, 4.75, 4.75]

3. First iteration (i = 0):
   * `Error = y - y_initial_vector`
   * Error = [2 - 4.75, 4 - 4.75, 5 - 4.75, 8 - 4.75]
   * Error = [-2.75, -0.75, 0.25, 3.25]
   
   *** Creates and fits tree model to this error***
   *** Assume predicted_error = [-2.5, -0.5, 0.5, 2.5]***
   
   * `y_initial_vector += (learning_rate * predicted_error)`
   * Update = [0.1 * -2.5, 0.1 * -0.5, 0.1 * 0.5, 0.1 * 2.5]
   * y_initial_vector becomes [4.5, 4.7, 4.8, 5.0]

4. Second iteration (i = 1):
   * Error = [2 - 4.5, 4 - 4.7, 5 - 4.8, 8 - 5.0]
   * Error = [-2.5, -0.7, 0.2, 3.0]
   
   * Fits another tree model
   * Assume predicted_error = [-2.0, -0.5, 0.3, 2.2]
   
   * Updates y_initial_vector again
   * Final y_initial_vector ≈ [4.3, 4.65, 4.83, 5.22]

Now let's trace the `predict` function with a new input, say x_new = [[2.5]]:

1. `y_hat = np.mean(self.y)`
   * Starting with 4.75

2. First model prediction:
   * Assume model1.predict([[2.5]]) = 0.0
   * y_hat += 0.1 * 0.0
   * y_hat = 4.75

3. Second model prediction:
   * Assume model2.predict([[2.5]]) = 0.1
   * y_hat += 0.1 * 0.1
   * Final y_hat = 4.76

The key points in this tracing:
1. The fit function builds multiple trees, each trying to correct the errors of the previous predictions
2. Each tree's contribution is scaled by the learning_rate (0.1 in this case)
3. The predict function uses all trained trees to make the final prediction, combining their outputs

'''

