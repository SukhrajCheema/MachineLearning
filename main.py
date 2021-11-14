import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from combine_attributes import CombinedAttributes

HOUSING_PATH = r"C:\Users\Sukh\Documents\coding\Python\MachineLearning\housing.csv"


# to avoid unicode error, include r to produce raw string


def load_housing_data(housing_path=HOUSING_PATH):
    """Returns a pandas DataFrame object"""
    return pd.read_csv(HOUSING_PATH)


housing = load_housing_data()
print(housing.info())
print(housing.head())

# we observe total_bedrooms attribute has 20433 non-null values.
# Hence 207 california districts are missing this feature.

housing.hist(bins=50, figsize=(20, 15))

"""
median income is pre-processed.
attributes have vastly different scales.
transformations required to compensate for the skews in data.
"""

########################################################################################################################

"""
We are told median_income is an important attribute. Hence we plot the different categories of median_income on a 
histogram. 
"""
housing["income cat"] = pd.cut(housing["median_income"],
                               bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

"""
converts median income into a categorical variable to 
ensure test set is representative of the various categories of income.
Rather than completely random.
"""

housing["income cat"].hist()
# To show graph, write plt.show() here.


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(housing, housing["income cat"]):
    # generates indices to split data into training and test sets. Stratification based on the second parameter.
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    # we access the groups of data via their computed indexes, and assign them to the 'strat' variables, using loc().

"""
we remove the income_cat attribute below so data is back to original.
"""
for data_point in (strat_train_set, strat_test_set):
    data_point.drop("income cat", axis=1, inplace=True)

print(strat_train_set.info())
print(strat_test_set.info())

########################################################################################################################

housing_train = strat_train_set.copy()
housing_train.plot(kind='scatter',
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

corr_matrix = housing_train.corr()
var = corr_matrix['median_house_value'].sort_values(ascending=False)
print(var)

"""
We check correlation between some promising attributes.
scatter_matrix() function plots every numerical attribute with every other numerical attribute. 
"""

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing_train[attributes], figsize=(12, 8))

"""
The most promising attribute to predict median house value is median income.
We observe a strong correlation.
Price cap at $500,000 visible via horizontal line.
Data quirks via less obvious horizontal lines around 450000, 350000 and 280000. 
May remove to prevent algorithm reproducing these. 
"""

########################################################################################################################


housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

imputer = SimpleImputer(strategy='median')
housing_numerical = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_numerical)

"""
The imputer computed the median of each attribute and stored in statistics_ variable.
Only total_bedrooms has missing data however we cannot be sure that other attributes will have complete data. 
"""

X = imputer.transform(housing_numerical)  # imputes the missing values into X.
housing_train_imputed = pd.DataFrame(X, columns=housing_numerical.columns, index=housing_numerical.index)
housing_categorical = strat_train_set[['ocean_proximity']]  # do not forget double square bracket again.

print(housing_categorical.head(10))
# We observe this is a categorical attribute. We convert from text to numbers using one-hot encoding.

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_categorical)

attr_adder = CombinedAttributes(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

"""
Pipeline constructor takes a list of name/estimator pairs defining a sequence of steps.
All but the last estimator must be transformers
"""

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributes()),
    ('std_scalar', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_numerical)

"""
Single transformer which deals with numerical and categorical attributes.
"""
num_attribs = list(housing_numerical)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)

#######################################################################################################################

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions_1 = tree_reg.predict(housing_prepared)

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions_2 = forest_reg.predict(housing_prepared)

"""
Use part of training set for training, and part of it for model validation. 
To evaluate the decision tree model, we utilise train_test_split() function.
Instead; we split training set into 10 unique subsets known as folds. Then evaluates Decision Tree model
on a fold, training using the other 9 folds. 
"""


def display_scores(scores):
    print(f"Scores: {scores}")
    print(f"Mean: {scores.mean()}")
    print(f"Standard_deviation: {scores.std()}")


tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
print("Finished.")
# This one takes a little longer to compute. About 2 minutes.
# Most promising model.


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

"""
grid_search.best_params_
{'max_features': 8, 'n_estimators': 30
As these are the highest available parameters, would be a good idea to experiment with higher values,
score may continue to improve. 
RandomizedSearchCV better, evaluates randomly, iterates 1000 times etc. 
"""

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

#######################################################################################################################

"""
We inspect the best models.
Below, we display importance scores next to their corresponding attribute names.
"""

feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

extra_attributes = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
categorical_encoder = full_pipeline.named_transformers_['cat']
categorical_one_hot_attributes = list(categorical_encoder.categories_[0])
attributes = num_attribs + extra_attributes + categorical_one_hot_attributes
print(sorted(zip(feature_importances, attributes), reverse=True))

"""
With this information, we should drop less useful features : all categorical except inland.
"""


#######################################################################################################################

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
