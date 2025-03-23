from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import necessary preprocessing tools
from sklearn.compose import ColumnTransformer # Import ColumnTransformer

# Extract best parameters for the RandomForestClassifier
# by removing the 'classifier__' prefix from the keys
rf_params = {k.replace('classifier__', ''): v for k, v in clf.best_params_.items() if 'classifier__' in k}


models = {
    "RandomForest": RandomForestClassifier(**rf_params),  # Pass extracted parameters
    "SVM": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "NaiveBayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=1000)
}
categorical_features = X_train.select_dtypes(include=['object']).columns

# Create a ColumnTransformer to handle numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)]) # Apply OneHotEncoder to categorical features


# Fit the preprocessor on the training data and transform both train and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Train and evaluate
for name, model in models.items():
    # Fit the model on preprocessed training data
    model.fit(X_train_processed, y_train)

    # Predict on preprocessed test data
    y_pred = model.predict(X_test_processed)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

