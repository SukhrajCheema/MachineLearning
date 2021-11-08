from main import load_housing_data
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

data = load_housing_data()

data["income cat"] = pd.cut(data["median_income"],
                            bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                            labels=[1, 2, 3, 4, 5])

"""
converts median income into a categorical variable to 
ensure test set is representative of the various categories of income.
Rather than completely random.
"""


data["income cat"].hist()
# will not be shown because we removed these data points below.


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(data, data["income cat"]):
    # generates indices to split data into training and test sets. Stratification based on the second parameter.
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
    # we access the groups of data via their computed indexes, and assign them to the 'strat' variables, using loc().


"""
we remove the income_cat attribute so data is back to original.
"""

for data_point in (strat_train_set, strat_test_set):
    data_point.drop("income cat", axis=1, inplace=True)