from main import data
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np


data["income cat"] = pd.cut(data["median_income"],
                            bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                            labels=[1, 2, 3, 4, 5])

data["income cat"].hist()