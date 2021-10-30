import pandas as pd
import numpy as np


from math import log2
import itertools 
from sklearn.model_selection import train_test_split

class e_Support_Vector_Machine:
    """ This is a Support Vector Machine algorithm I've kindly called the emotional Support Vector Machine. 
    This can be used to for binary classification into 'happy' and 'sad' classes depending on the context of the target variable
    (in the case of the assignment, a fire is a 'sad' outcome) - emijis in the relevent plots will support this

    There are also variables named after some of my friends dotted throughout this where I feel appropriate eg.:
    
    'julian_stress' - will be used inplace of C to tune the size of the margin between the support vector and decision boundary. 
        Low stress would give us a wide margin, giving the system high confidence in its predictions

    I am using the sci-kit learn project template as the structure for this class.
    
    

    Parameters
    ----------
    happy_class : str, default='no fire'
        A parameter used during plotting to 

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, happy_class='no fire'):
        self.happy_class = happy_class


    def fit(self, X, y, C=0):
        """
        Training an SVM is an optimisation problem where the objective is to maximise the distance between a hyperplane and the closest datapoint for any weight and bias (W, b).
        The equation for a seperating hyperplane is (W).(X) + b = 0 where W is a weight vector, X is the input vector and b is a bias term.
                    
        The distance between the hyperplanes is 2/||W|| - so the goal is to minimise the magnitude of W and therefore maximise the 
        distnce between the hyperplanes with the condition no points lie between the two hyperplanes.
        This minimising variable can be re-written as (||W||^2)*(1/2) for convenience, it is now a quadratic with one global minimum.
        
        So writing the eq. for hyperplanes for the two classes gives:
             (W).(Xi) + b = 1 for positive class (yi=1)
             (W).(Xi) + b = -1 for negative class (yi=-1)
        This can be generalised for both classes as yi is known and can be multiplied by the above equations to give the form:
            yi(Xi.W+b)-1=0

        We want to minimise ||W|| and maximise b by iterating through possible values for W and keeping the W and b that satisfy the above equation 
        and picking the W and b that minimise ||W||. This will be done using Stochastic Gradient Descent and the bias term will be evaluated in the W vector.

        The cost funciton is described by the equation:
        J= (||W||^2)/2  +  (C/N)*SUMALL(maxvalue(0, 1-yi*W*xi)) 

        The gradient of this cost funciton is :
        DJ = 1/N SUMALL(w) if the max(0, 1-yi*W*xi) = 0 and satisfies the general form of the hyperplane equation or
        DJ = 1/N SUMALL(w-C*yi*xi) where is does not satify the eqn.




        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y


        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


    def f1_score(self, y_test, y_train):
        pass



def information_gain(df, target, columns):
    """
    Calculate the information gain for all the columns to be presented at the feature selection screen. 
    Mean value will be used to bucket the values.
    """
    df_output = pd.DataFrame()
    
    target_vals = list(set(df[target]))
    val1 = target_vals[0]
    val2 = target_vals[1]
    
    df_entropy = -(len(df[df[target]==val1])/len(df))*log2(len(df[df[target]==val1])/len(df)) - (len(df[df[target]==val2])/len(df))*log2(len(df[df[target]==val2])/len(df))
    
    
    for col in columns:        
        mean_val = np.mean(df[col])
        high_val= df[df[col]>=mean_val]
        low_val = df[df[col]<mean_val]
        try:
            # Some columns like rain have no fires above the mean value so the below equation breaks down - this is a very significant feature to include
            if len(set(high_val[target])) ==2 :
                high_exp1 = -(len(high_val[high_val[target]==val1]) / len(high_val))*log2(len(high_val[high_val[target]==val1])/len(high_val)) 
                high_exp2 = -(len(high_val[high_val[target]==val2]) / len(high_val))*log2(len(high_val[high_val[target]==val2])/len(high_val))
                high_ent =  high_exp1 + high_exp2
            else:
                high_ent=0

            if len(set(low_val[target])) ==2 :
                low_exp1 = -(len(low_val[low_val[target]==val1]) / len(low_val))*log2(len(low_val[low_val[target]==val1])/len(low_val)) 
                low_exp2 = -(len(low_val[low_val[target]==val2]) / len(low_val))*log2(len(low_val[low_val[target]==val2])/len(low_val))
                low_ent =  low_exp1 + low_exp2
            else:
                low_ent=0

            info_gain = df_entropy - (len(high_val)/len(df))*high_ent - (len(low_val)/len(df))*low_ent
            df_output = df_output.append([[col,np.round(mean_val,2),np.round(info_gain, 2)]])
        except:
            pass
    df_output = df_output.rename(columns={0:'Column', 1:"Mean Value", 2:"Information Gain"})
    return(df_output)
        
        

def feature_selection(df):
    """
    Allows user input to pick the dependant and independant variables. 
    Once the dependant variable is chosen the information gain for the independant variables is calculated to help the user pick out useful features. 
    To calculate information gain variables were binned according to the feature mean value 
        - this is not useful in the case of categorical data but the user should know that, this is just to assist the feature selection process
    
    """
    print(df.dtypes)
    #target='yes'
    target = input('Pick the target variable')
    
    df[target] = [x.strip() for x in df[target]]
    df[target] = df[target].replace({'no':-1, 'yes':1})
    df_cols = df.drop(target ,axis=1)
   
    ig = information_gain(df, target, df_cols.columns)
    info_cols = pd.DataFrame(df_cols.dtypes)
    info_cols.reset_index(inplace=True)
    info_cols = info_cols.rename(columns={'index':'Column', 0:'Data Type'})
    info_cols= info_cols.merge(ig, on='Column').sort_values("Information Gain" ,ascending=False)
    print("Information gain calculated for bins either side of mean values for each feature")
    print(info_cols)
    #cols = "rainfall, humidity, buildup_index, drought_code"
    cols = input("Please enter the desired columns for anaylsis: ")
    cols = [x.strip() for x in cols.split(',')]
    return target, cols


def normalise(df, column):
    """
    Range normalisation - Function to normalise the data in the datasets column if needed.
    """
    return (df[column]-min(df[column]))/(max(df[column]) - min(df[column]))


def cost_function(W, X, y, C):
    """
    The cost funciton is described by the equation below and will be evaluated to determine 
    if the model has achieved an acceptably low cost function before the number of iterations has been reached.
    J= (||W||^2)/2  +  (C/N)*SUMALL(maxvalue(0, 1-yi*W*xi)) 
    """
    for i in range(len(X)):
        # Evaluate for the left side of the '+'. Dot product of a vector on itself returns the magnitude
        lhs = (1/2) * np.dot(W,W)#**0.5
        
        # Evaluate for right hand side of the '+'
        hyper_plane_distance = np.max([0,1-(y[i]*np.dot(X[i],W))])
        N = len(X)
        rhs = (C/N)*np.sum(hyper_plane_distance)
        return (lhs+rhs)


def gradient_desc(W, X, y, C):
    '''
    Calculate the gradient at a point for given values of W, Xi, yi and return the value for the SVM to evaluate the next values for W.
    '''
    
    grad = np.zeros(len(W))
    # Calculate distance to the hyperplane for W, Xi, yi 
    distance = np.max([0, 1 - y * np.dot(W,X)])
    
    # If the max value of the above is 0 then the point is a support vector and the weights are insightful else decrease the weights by C(yi*Xi)
    if distance == 0:
        grad = W
    else:
        grad = W - (C * y * X)
    return grad


def svm(X, y, learning_rate, n_iter, tolerance, C):
    # Initialise the weight vector with length the same as the number of features being analysed and give each coefficient a value of 1.
    W = np.ones(X.shape[1])  
    
    # For each iteration, calculate the weights.
    for step in range(n_iter):
        # For each X value evaluate the gradient at that point with the given weights and subtract the gradient*learning rate from the weights to refine the W vector
        for index, val in enumerate(X):
            W = W-learning_rate*gradient_desc(W, val,  y[index], C)
    return(W)



if __name__ == '__main__':
    df = pd.read_csv("../Data/wildfires.txt", delimiter='\t')

    target, cols = feature_selection(df)
    X_train, X_test, y_train, y_test = train_test_split(df[cols], df[target], test_size=0.3, random_state=10)
    dataset = [X_train, X_test, y_train, y_test]
    for i in dataset:
        i.reset_index(inplace=True, drop=True)

    # The b value from the hyperplane equation will be evaluated at the same time as W so it makes sense to add a feature to the dataset of value 1 to act as the b ceofficient.
    X_train = np.hstack([X_train, np.ones(X_train.shape[0]).reshape(-1,1)])
    model = svm(X_, y_, 1e-4, 100, 1, 0.5)

    
