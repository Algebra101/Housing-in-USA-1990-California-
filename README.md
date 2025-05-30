```markdown
California Housing Price Prediction(1990)

This project is a Machine Learning pipeline designed to predict housing prices in California using various regression models. The pipeline involves data acquisition, preprocessing, visualization, transformation, model training, hyperparameter tuning, and model evaluation.

 📌 Features
- Data fetching from an external source
- Stratified sampling for training and testing sets
- Data visualization and correlation analysis
- Handling missing values and categorical encoding
- Feature engineering using a custom transformer
- Standardization and scaling
- Model training and evaluation using different algorithms
- Cross-validation and hyperparameter tuning
- Feature importance analysis
- Test set evaluation with confidence intervals

📂 Dependencies
Ensure you have the following Python libraries installed:

```python
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
```

## 🚀 Model Training & Evaluation
### 1️⃣ Linear Regression
```python
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

### 2️⃣ Decision Tree Regressor
```python
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
```

### 3️⃣ Random Forest Regressor
```python
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
```

## 📏 Model Performance
#### Root Mean Squared Error (RMSE)
```python
housing_predictions = lin_reg.predict(housing_prepared)
lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
print(f"Linear Regression RMSE: {lin_rmse}")
```

#### Cross Validation for Better Model Evaluation
```python
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```

## 🔧 Hyperparameter Tuning for Random Forest
```python
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
```

## 🔍 Best Model Analysis
```python
best_estimator = grid_search.best_estimator_
feature_importances = best_estimator.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
```

## 🎯 Evaluating the Best Model on the Test Set
```python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(f"Final RMSE on Test Set: {final_rmse}")
```

📊 Confidence Interval for Test RMSE
```python
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
```

## 🎉 Conclusion
This project successfully builds a Machine Learning pipeline for predicting California housing prices. Through hyperparameter tuning and model evaluation, we improve predictive accuracy and assess feature importance.

## ✍ Author
Created by **Algebra**.

---




