from sklearn.model_selection import RandomizedSearchCV, train_test_split, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # Import ColumnTransformer

# Split data (assuming last column is target)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features (columns with string values)
categorical_features = X_train.select_dtypes(include=['object']).columns

# Create a ColumnTransformer to handle numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)]) # Apply OneHotEncoder to categorical features

# Create a pipeline with preprocessing and the classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier())])
# Hyperparameter tuning (adjust param_grid for pipeline)
param_dist = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# ... (rest of your code using 'pipeline' instead of 'clf') ...
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
clf = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=ss, verbose=2, random_state=42, n_jobs=-1) # Use pipeline
clf.fit(X_train, y_train) # Fit the pipeline
print("Best Parameters:", clf.best_params_)

