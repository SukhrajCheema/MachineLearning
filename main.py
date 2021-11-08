import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import StratifiedShuffleSplit

HOUSING_PATH = r"C:\Users\Sukh\Documents\coding\Python\MachineLearning\housing.csv"


# to avoid unicode error, include r to produce raw string


def load_housing_data(housing_path=HOUSING_PATH):
    """Returns a pandas DataFrame object"""
    return pd.read_csv(HOUSING_PATH)


data_main = load_housing_data()

print(data_main.info())
print(data_main.head())

# we observe total_bedrooms attribute has 20433 non-null values.
# Hence 207 california districts are missing this feature.

data_main.hist(bins=50, figsize=(20, 15))

"""
median income is pre-processed.
attributes have vastly different scales.
transformations required to compensate for the skews in data.
"""
