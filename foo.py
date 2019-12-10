import pandas as pd
from query import PrivateQuery

# Parameters to pass
dataset_path = 'data/adult.csv'
epsilon = 0.5
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
         'marital-status', 'occupation', 'relationship', 'race', 'sex',
         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
use_laplace = True

# Create `PrivateQuery` object
print("------------------------------------")
print("Creating PrivateQuery Object")
query = PrivateQuery(dataset_path, epsilon, names, use_laplace=False, demo_mode=True)
print(query.dataset.head())
print("------------------------------------")

# Test the categorical count method for the education column.
# Note this is for a column with categorical values
col_name = 'education'
print("Counting the education levels in the dataset")
result, dif_pri_result = query.categorical_count(col_name)
for res_key, dif_pri_key in zip(result.keys(), dif_pri_result.keys()):
    print("{}: {} | {}: {}\n".format(res_key, result[res_key], dif_pri_key, dif_pri_result[dif_pri_key]))
print("------------------------------------")

# Test the average method for age column.
# Note this is for a column with continuous values
col_name = 'age'
print("Calculating the average age")
result, dif_pri_result = query.average(col_name=col_name)
print("The actual average age is : {}".format(result))
print("The differentially private average is : {}".format(dif_pri_result))
print("------------------------------------")
