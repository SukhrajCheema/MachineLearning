from create_test_set import strat_test_set, strat_train_set
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude')