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

    def __init__(self, path_to_dataset, epsilon, names, use_laplace=True, demo_mode=False):
        self.dataset = pd.read_csv(path_to_dataset, names=names, index_col=False)
        self.epsilon = epsilon
        self.cont_col, self.cat_col = self.preprocess()
        self.use_laplace = use_laplace
        self.demo_mode = demo_mode
    
    def preprocess(self, max_card=20, dep_var=None):
        """
        Helper function that returns column names of cont and cat variables from given df.
        Copied from https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py
        """
        cont_names, cat_names = [], []
        for label in self.dataset:
            if label == dep_var: continue
            if (self.dataset[label].dtype == int and self.dataset[label].unique().shape[0] > max_card) or \
               (self.dataset[label].dtype == float):
                cont_names.append(label)
            else:
                cat_names.append(label)
        return cont_names, cat_names


    def count(self):
        """Returns a differentially private count of rows in the database
        
        Parameters
        ----------
        col_name : str
            The name of the column whose values is to be counted
        """
        if not self.demo_mode: return len(self.dataset) + self.get_noise()
        else: return (len(self.dataset), len(self.dataset) + self.get_noise)

    def get_noise(self):
        """Returns Laplacian noise based on the created epsilon value"""
        if self.use_laplace:
            noise = np.random.laplace(scale=1/self.epsilon)
        else:
            noise = np.random.normal(scale=1/self.epsilon)
        
        return noise

    def categorical_count(self, col_name):
        """Returns a differentially private count of categorical values 
        
        Parameters
        ----------
        col_name : str
            Name of column
        """
        # Step 1 - Count number of categories in each category
        col_dict = {}

        for i in range(len(self.dataset)):
            try:
                col_dict[self.dataset.iloc[i][col_name]] += 1
            except KeyError:
                col_dict[self.dataset.iloc[i][col_name]] = 1

        # Step 2 - Add noise to the distribution
        if self.demo_mode:
            col_dict_w_noise = col_dict.copy()
            for keys in col_dict_w_noise:
                col_dict_w_noise[keys] += self.get_noise()
            return (col_dict, col_dict_w_noise)
        else:
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

        if self.demo_mode:
            return (avg, avg + self.get_noise())
        else:
            return avg + self.get_noise()
