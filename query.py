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
        self.preprocess()
    
    def preprocess(self):
        """Preprocesses the dataset and calculates some values
        """
        return


    def count(self, col_name):
        """Returns a differentially private count of the values in the row
        
        Parameters
        ----------
        col_name : str
            The name of the column whose values is to be counted
        """
    
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
