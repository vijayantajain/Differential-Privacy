import pandas as pd
from query import PrivateQuery

# Parameters to pass
dataset_path = 'data/adult.csv'
epsilon = 0.5
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
         'marital-status', 'occupation', 'relationship', 'race', 'sex',
         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

# Create `PrivateQuery` object
print("------------------------------------")
print("Creating PrivateQuery Object")
query = PrivateQuery(dataset_path, epsilon, names)
print(query.dataset.head())
print("------------------------------------")

# Test the categorical count method for the education column. 
# Note this is for a column with categorical values
col_name = 'education'
education_cat_values = [' Bachelors', ' Some-college', ' 11th', ' HS-grad', ' Prof-school', ' Assoc-acdm', ' Assoc-voc', ' 9th', ' 7th-8th', ' 12th', ' Masters', ' 1st-4th', ' 10th', ' Doctorate', ' 5th-6th', ' Preschool']
print("Counting the education levels in the dataset")
result = query.categorical_count(col_name, education_cat_values)
for keys in result.keys():
    print("{}: {}".format(keys, result[keys]))
print("------------------------------------")

# Test the average method for age column.
# Note this is for a column with continuous values
col_name = 'age'
print("Calculating the average age")
result = query.average(col_name=col_name)
print("The average age is : {}".format(result))
print("------------------------------------")
