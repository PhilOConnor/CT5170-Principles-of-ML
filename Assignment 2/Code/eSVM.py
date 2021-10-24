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
    
    'julian_stress' - a non-dimensional value from 0-1 for points between the Support Vectors 
    and the decision boundary to imply possible 'stress' or uncertainty in the decision.

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


    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

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
    
    print(df.dtypes)
    
    target = input('Pick the target variable')
    
    df[target] = [x.strip() for x in df[target]]
    df_cols = df.drop(target ,axis=1)
   
    ig = information_gain(df, target, df_cols.columns)
    info_cols = pd.DataFrame(df_cols.dtypes)
    info_cols.reset_index(inplace=True)
    info_cols = info_cols.rename(columns={'index':'Column', 0:'Data Type'})
    info_cols= info_cols.merge(ig, on='Column').sort_values("Information Gain" ,ascending=False)
    print("Information gain calculated for bins either side of mean values for each feature")
    print(info_cols)
    cols = input("Please enter the desired columns for anaylsis: ")
    cols = [x.strip() for x in cols.split(',')]
    return target, cols


if __name__ == '__main__':
    df = pd.read_csv("../Data/wildfires.txt", delimiter='\t')

    target, cols = feature_selection(df)
    X_train, X_test, y_train, y_test = train_test_split(df[cols], df[target], test_size=0.3, random_state=10)
