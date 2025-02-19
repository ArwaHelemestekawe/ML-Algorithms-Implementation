import numpy as np
class node:
    # نوعين  nodes هو انا عندي كام نوع من ال 
    # root that contains condition === decision root 
    # leaf >> classification node
    def __init__(self,feature=None,condition_value=None,right=None,left=None,classification_result=None):
        self.condition_value=condition_value
        self.right=right
        self.left=left
        self.classification_result=classification_result
        self.feature=feature

class decision_tree:
    def __init__(self,max_depth=3) :
        self.max_depth=max_depth
        self.tree=None


    def split(self,condition_value,feature,x):
        left_indices=np.where(x[: ,feature]<=condition_value)
        right_indices=np.where(x[: ,feature]>=condition_value)
        return left_indices,right_indices
        # where function returns the indices of data points that follow the condition


    def mean_squar_error(self,y):
        return np.mean(y-np.mean(y)**2)
    





    def var_reduction(self,y,left_indices,right_indices):
        parent_impurity=self.mean_squar_error(y)
        right_impurity=self.mean_squar_error(y[right_indices])
        left_impurity=self.mean_squar_error(y[left_indices])
        wieght_left=len(left_indices)/len(y)
        weight_right=len(right_indices)/len(y)
        var_reduction=parent_impurity-((wieght_left*left_impurity)+(weight_right*right_impurity))
        return var_reduction
    



    def fit(self,x,y,depth=0):
        if depth==self.max_depth or np.all(y[0]==y):
                return node(classification_result=np.mean(y))
        #pure node
        num_sampels,num_features=x.shape
        best_var_reduction=0
        best_split=None
        # مفروض انه بيجرب عند قسمة معينة مثلا من كل فيتشر
        # best split that will have best information gain 
        for i in range(num_features):
            feature_value=x[:, i]
            unique_values=np.unique(feature_value)
            for value in unique_values:
                # لو في سامبل قيمتهاعند فيتشر معينة متكررة يعني 
             left_indices,right_indices=self.split(value,i,x)
             if len(left_indices)==0 or len(right_indices)==0:
                 continue
             var_reduct=self.var_reduction(y,left_indices,right_indices)
             if var_reduct>best_var_reduction:
                best_var_reduction=var_reduct
                best_split=(value,i,left_indices,right_indices)
             """ كدة انا حفظت كل المعلومات عن احسن سبليت عملته كان عند انهي فيتشر وانهي فاليو للفيتشر دي واليمين والشمال فيهم ايه """
        if best_var_reduction==0:
            return node(classification_result=np.mean(y))
        value,selcted_feature,left_indices,right_indices=best_split
        # لحد هنا هو كدة اتقسم عي عمق واحد بس عايزين بقا نكرر لحد ما نحقق البيز كيس
        left_sub_tree=self.fit(x[left_indices],y[left_indices],depth+1)
        right_sub_tree=self.fit(x[right_indices],y[right_indices],depth+1)
        self.tree=node(condition_value=value,feature=selcted_feature,left=left_sub_tree,right=right_sub_tree)
        return self.tree
    


    


    def predict_data_point(self,point,node):
        # node is the current node in the tree we're examining
        # امتي هنعمل بريديكت لما اوصل لليف نود
        if node is None:
            return node
        if node.classification_result is not None:
            return node.classification_result

        if point[node.feature]<=node.condition_value:
            return self.predict_data_point(point,node.left)

        # x[node.feature] gets the value of the feature we're testing
        # For example, if node.feature = 0, we look at first column
        # If that value is less than or equal to our condition_value
        # We go down the left path


        else:
            return self.predict_data_point(point, node.right)
    



    def predict(self, X):
     result = [self.predict_data_point(x, self.tree) for x in X]
     return np.array(result)
