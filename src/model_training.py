from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def split_data(df):
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_linear(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_ridge(X_train, y_train):
    ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])

    param_grid = {'model__alpha': [0.1, 1, 10, 100]}

    grid = GridSearchCV(ridge, param_grid, cv=5)
    grid.fit(X_train, y_train)

    return grid


def train_lasso(X_train, y_train):
    lasso = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1]}

    grid = GridSearchCV(lasso, param_grid, cv=5)
    grid.fit(X_train, y_train)

    return grid
