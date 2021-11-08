from create_test_set import strat_test_set, strat_train_set, data
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

housing = strat_train_set.copy()
housing.plot(kind='scatter',
             x='longitude',
             y='latitude',
             alpha=0.4,
             s=strat_train_set['population'] / 100,  # radius of circle represents size of pop.
             label='population',
             figsize=(10, 7),
             c='median_house_value',  # colour represents median house value.
             cmap=plt.get_cmap('jet'),
             colorbar=True
             )


corr_matrix = housing.corr()
var = corr_matrix['median_house_value'].sort_values(ascending=False)
print(var)

"""
We check correlation between some promising attributes.
scatter_matrix() function plots every numerical attribute with every other numerical attribute. 
"""

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

"""
The most promising attribute to predict median house value is median income.
We observe a strong correlation.
Price cap at $500,000 visible via horizontal line.
Data quirks via less obvious horizontal lines around 450000, 350000 and 280000. 
May remove to prevent algorithm reproducing these. 
"""