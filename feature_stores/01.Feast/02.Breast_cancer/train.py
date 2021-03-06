# Importing dependencies
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump


# Configure store path
store_path = "/home/pliu/git/FeatureEngineering/feature_stores/01.Feast/02.Breast_cancer/breast_cancer"

# Getting our FeatureStore
store = FeatureStore(repo_path=store_path)

# Retrieving the saved dataset and converting it to a DataFrame
training_df = store.get_saved_dataset(name="breast_cancer_dataset").to_df()
print(training_df.head())

# Separating the features and labels
labels = training_df['target']
features = training_df.drop(
    labels=['target', 'event_timestamp', "patient_id"], 
    axis=1)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    labels, 
                                                    stratify=labels)

# Creating and training LogisticRegression
reg = LogisticRegression()
# The sorted(X_train) in the call to reg.fit provides the features to our model in alphabetical order. This is
# necessary because when loading features from feature views, Feast may not preserve the order from the source data.
reg.fit(X=X_train[sorted(X_train)], y=y_train)

# Saving the model
dump(value=reg, filename="./tmp/model.joblib")