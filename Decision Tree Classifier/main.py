#!pip install scikit-learn matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(url, names=column_names)

# 1) Print Data to answer 1) Include a basic description of the data (what are the features and labels)
#print(iris_data.to_string())
#print(iris_data)

# Separate features and classifiers
X = iris_data.drop('class', axis=1)
y = iris_data['class']

# 2) Split the data into training, validation, and testing sets 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a range of hyperparameter values to test (max depth and  min_samples_split)
max_depth_values = list(range(1, 11))
min_samples_split_values = list(range(2, 11))

# Create lists to store performance metrics for the chosen hyper parameter values
validation_accuracy_max_depth = []
validation_accuracy_split = []

# 3) Fit a decision tree on the training dataset.
# 4) Tune at least 2 hyperparameters in the decision tree model
# Train and evaluate the decision tree with different hyperparameter values
for max_depth in max_depth_values:
  # Create a new decision tree classifier with the new hyperparameter value and the same random state as the sets
  dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42) # new DT 
  dt_classifier.fit(X_train, y_train) # fit the data

  y_val_pred = dt_classifier.predict(X_validation)  # outputs with the validation set as inputs
  val_accuracy = accuracy_score(y_validation, y_val_pred) # accuracy of this instance of DT on validation set
  validation_accuracy_max_depth.append(val_accuracy) # add to the accuracy list

# Plot hyperparameter values vs. performance metrics
plt.plot(max_depth_values, validation_accuracy_max_depth, marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Validation Accuracy')
plt.title('Decision Tree Performance with Varying Max Depth')
plt.grid(True)
plt.show()

# Now lets tune min_samples_split
for min_samples_split in min_samples_split_values:
  # Create a new decision tree classifier with the new hyperparameter value and the same random state as the sets
  dt_classifier = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=42) # new DT
  dt_classifier.fit(X_train, y_train) # fit the data

  y_val_pred = dt_classifier.predict(X_validation) # outputs with the validation set as inputs
  val_accuracy = accuracy_score(y_validation, y_val_pred) # accuracy of this instance of DT on validation set
  validation_accuracy_split.append(val_accuracy) # add to the accuracy list

# Plot min_samples_split vs. performance metrics
plt.figure()  # Create a new figure for this plot
plt.plot(min_samples_split_values, validation_accuracy_split, marker='o')
plt.xlabel('Min Samples Split')
plt.ylabel('Validation Accuracy')
plt.title('Decision Tree Performance with Varying Min Samples Split')
plt.grid(True)
plt.show()

best_max_depth = 2  # Replace with the tuned value
best_min_samples_split = 2  # Replace with the tuned value

# Create a new DecisionTreeClassifier with the tuned hyperparameters
final_dt_classifier = DecisionTreeClassifier(max_depth=best_max_depth, 
                                              min_samples_split=best_min_samples_split,
                                              random_state=42)  

# Fit the model on the entire training data (X_train + X_validation)
final_dt_classifier.fit(pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation]))

# Make predictions on the test data
y_pred = final_dt_classifier.predict(X_test)

# Generate and print the classification report
report = classification_report(y_test, y_pred)
print("\n" + report)
