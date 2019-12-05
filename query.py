"""Module containing the class that queries the csv file"""

import numpy as np
import pandas as pd

class PrivateQuery():
    """Private Query is a class that executes differentially private queries

    Parameters
    ----------
    path_to_dataset: str
        Path to dataset
    epsilon : double
        Level of Differential Privacy
    **kwargs : dict
        Keyword arguments to be passed to create csv file
    """

    def __init__(self, path_to_dataset, epsilon, names):
        self.dataset = pd.read_csv(path_to_dataset, names=names, index_col=False)
        self.epsilon = epsilon
        self.cont_col, self.cat.col = self.preprocess()
    
    def preprocess(self, max_card=20, dep_var=None):
        """
        Helper function that returns column names of cont and cat variables from given df.
        Copied from https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py
        """
        cont_names, cat_names = [], []
        for label in self.dataset:
            if label == dep_var: continue
            if self.dataset[label].dtype == int and self.dataset[label].unique().shape[0] > max_card or self.dataset[label].dtype == float: cont_names.append(label)
            else: cat_names.append(label)
        return cont_names, cat_names


    def count(self):
        """Returns a differentially private count of rows in the database
        
        Parameters
        ----------
        col_name : str
            The name of the column whose values is to be counted
        """
        return len(self.dataset) + self.get_noise()

    def get_noise(self):
        """Returns Laplacian noise based on the created epsilon value
        """
        return np.random.laplace(scale=1/self.epsilon)

    def categorical_count(self, col_name, categories):
        """Returns a differentially private count of categorical values 
        
        Parameters
        ----------
        col_name : str
            Name of column
        categories : list
            List of string containing the name of the categories
        """
        # Step 1 - Count
        col_dict = dict.fromkeys(categories, 0)

        for i in range(len(self.dataset)):
            col_dict[self.dataset.iloc[i][col_name]] += 1

        # Step 2 - Add noise to the distribution
        for keys in col_dict:
            col_dict[keys] += self.get_noise()

        return col_dict

    def average(self, col_name):
        """Returns differentially private average of continuous values
        
        Parameters
        ----------
        col_name : str
            Name of column for whose average is to be calculated
        """
        # Create 
        sum = 0
        for i in range(len(self.dataset)):
            sum += self.dataset.iloc[i][col_name]
        
        avg = sum/len(self.dataset)

        return (avg + self.get_noise())
