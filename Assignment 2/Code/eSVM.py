import pandas as pd
import numpy as np
import pdb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from math import log2
from sklearn.svm import SVC
import itertools 
from sklearn.model_selection import train_test_split
import random
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

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
    def __init__(self, learning_rate, n_iter, tolerance, C):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.C = C
        #self.happy=happy


    def fit(self, X, y):
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
        X, y = check_X_y(X, y, accept_sparse=True)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        

        # Initialise the weight vector with length the same as the number of features being analysed and give each coefficient a value of 1.
        W = np.ones(X.shape[1])  

        # For each iteration, calculate the weights.
        for step in range(self.n_iter):
            # For each X value evaluate the gradient at that point with the given weights and subtract the gradient*learning rate from the weights to refine the W vector
            for index, value in enumerate(X):
                W = W-self.learning_rate*gradient(W, value,  y[index], self.C)
        self.W = W
        return(self.W)

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
        #check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        output = []
        for i in X:
            if np.dot(self.W.T,i)>0:
                output.append(1)
            else:
                output.append(-1)
        return(output)
    
   
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
    print("\n")
    print(df.dtypes)
    #target='yes'
    target = input('Pick the target variable: ')
    
    df[target] = [x.strip() for x in df[target]]
    df[target] = df[target].replace({'no':-1, 'yes':1})
    df_cols = df.drop(target ,axis=1)
   
    ig = information_gain(df, target, df_cols.columns)
    info_cols = pd.DataFrame(df_cols.dtypes)
    info_cols.reset_index(inplace=True)
    info_cols = info_cols.rename(columns={'index':'Column', 0:'Data Type'})
    info_cols= info_cols.merge(ig, on='Column').sort_values("Information Gain" ,ascending=False)
    print("\n")
    print("Information gain calculated for bins either side of mean values for each feature")
    print(info_cols)
    #cols = "rainfall, humidity, buildup_index, drought_code"
    cols = input("Please enter the desired columns for anaylsis (use a comma seperate the features): ")
    cols = [x.strip() for x in cols.split(',')]
    return target, cols

def normalise(df, column):
    """
    Function to normalise the data in the datasets columns - this has a negative impact on the models performance but included because was covered in lectures and to show work done.
    """
    return 2*(df[column]-min(df[column]))/(max(df[column]) - min(df[column]))-1
    

def cost_function(W, X, y, C):
    """
    The cost funciton is described by the equation below and will be evaluated to determine if the model has achieved an acceptably low cost function before the number of iterations has been reached.
    J= (||W||^2)/2  +  (C/N)*SUMALL(maxvalue(0, 1-yi*W*xi)) 
    """
    for i in range(len(X)):
        # Evaluate for the left side of the '+'. Dot product of a vector on itself returns the magnitude
        lhs = (1/2) * np.dot(W.T,W)#**0.5
        
        # Evaluate for right hand side of the '+'
        hyper_plane_distance = np.max([0,1-(y[i]*np.dot(X[i],W.T))])
        N = len(X)
        rhs = (C/N)*np.sum(hyper_plane_distance)
        return (lhs+rhs)




def gradient(W, X, y, C):
    '''
    Calculate the hyperplane distance at a point for given values of W, Xi, yi and return the value for the SVM to evaluate the next values for W.
    '''
    
    grad = np.zeros(len(W))
    # Calculate distance to the hyperplane for W, Xi, yi 
    distance = np.max([0, 1 - y * np.dot(W.T,X)])
    
    # If the max value of the above is 0 then the point is a support vector and the weights are insightful else decrease the weights by C(yi*Xi)
    if distance == 0:
        grad = W
    else:
        grad = W - (C * y * X)
    return grad




def cross_val(clf, X, y, n_folds):
    
    """
    SK Learns cross_val_score was not working with my implimentation of the SVM so the below code shuffles and splits the dataset into 9/10 and 1/10 for training and validation. 
    The first j elements are taken for validation and the remainder are training. Once the first j items have been used for validaiton they are concatenated onto the end of the training set and the next j elements are taken from the top of the training set.
    """
    X,y = shuffle(X,y)
    output_scores=[]
    print("\n")
    for i in range(n_folds):
        index_slicer = len(X)//n_folds
        X_val, y_val = X[ :index_slicer ], y[ : index_slicer]
        X_train, y_train = X[index_slicer: ], y[index_slicer: ]
        
        # To iterate through the folds of the cross validation, append the first j elements to the end of the array and then slice them off the start.
        # By always treating the first j elements as the validation set and the remainder as the training set, I can do n-fold CV without adapting my e_Support_Vector_machine class to accecpt the sklearn implimentation.

        X, y =np.concatenate((X_train,X_val)),np.concatenate((y_train,y_val))
        #X, y = X[index_slicer: ], y[index_slicer :] 
        
        #pdb.set_trace()
        
        clf.fit(X_train, y_train)
        clf_predicts = clf.predict(X_val)
        f1 = f1_score(y_val, clf_predicts)
        print(f'Iteration: {i}.  F1 score: {f1}')
        output_scores.append(f1)
    print(f'\nMean F1 is: {np.mean(output_scores)}')


def tp_fp(actual, predicions):
    tp=0
    tf=0
    for i in range(len(actual)):
        if actual[i]==predicions[i]==1:
            tp+=1
        elif (actual[i]==1) & (predicions[i]==0):
            fp+=1
    return tp,fp



if __name__ == '__main__':
    df = pd.read_csv("../Data/wildfires.txt", delimiter='\t')

    target, cols = feature_selection(df)
    X_train, X_test, y_train, y_test = train_test_split(df[cols], df[target], test_size=0.3, random_state=10)

    dataset = [X_train, X_test, y_train, y_test]
    for i in dataset:
        i.reset_index(inplace=True, drop=True)

    # The b value from the hyperplane equation will be evaluated at the same time as W so it makes sense to add a feature to the dataset of value 1 to act as the b ceofficient.
    #X_train = np.hstack([X_train, np.ones(X_train.shape[0]).reshape(-1,1)])
    #X_test = np.hstack([X_test, np.ones(X_test.shape[0]).reshape(-1,1)])


    # Instantiate my SVM object and use gridsearch to select the optimum hyperparameters
    
    my_svm = e_Support_Vector_Machine(learning_rate= 1e-3, n_iter = 1000, tolerance = 0.001, C=0.05)

    # 10 fold cross validation
    print("\nStarting 10 fold cross validation using my SVM:")
    cross_val(my_svm, X_train, y_train, 10)

    # Fit the model to the training data
    my_svm.fit(X_train, y_train)

    # Create predicitons with the model
    my_y_predictions = my_svm.predict(X_test)
    my_svm_score = f1_score(y_test, my_y_predictions)

    
    # Compare to Sci-kit learns implimentation
    svm = SVC()
    print("\nStarting 10 fold cross validation using Sci-kit learn SVM:")
    cross_val(svm, X_train, y_train, 10)
    skl_svm = svm.fit(X_train, y_train)

    skl_y_predicitons = skl_svm.predict(X_test)
    skl_svm_score = f1_score(y_test, skl_y_predicitons)

    print(f"\nMy SVM F1 score: {my_svm_score}. Sci-kit Learn SVM F1 score {skl_svm_score}")


    output = str(list(zip(y_test,my_y_predictions)))
    with open('Output.txt', 'w') as f:
        f.writelines(output)
        
    f.close()
    print("\nOutput written to Output.txt. Format is [(y_test_0, y_prediciton_0),...,(y_test_n, y_prediciton_n)]")

    print("\nAssignment Complete")