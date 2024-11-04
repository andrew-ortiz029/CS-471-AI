import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math


training_data = pd.read_csv("./training.csv", header=None, usecols=[19,23], names=['Time','Current'])
test_data = pd.read_csv("./test.csv", header=None, usecols=[0, 4], names=['Time','Current'])

training_data = training_data[training_data['Time'] <= 5.4]
test_data = test_data[test_data['Time'] <= 2.4]

df = training_data
fault_start = 5.1
fault_end = 5.4
# Separate the data points
fault_data = df[(df['Time'] >= fault_start) & (df['Time'] <= fault_end)]
normal_data = df[(df['Time'] < fault_start) | (df['Time'] > fault_end)]

plt.figure(figsize=(40, 6))

# Plotting for column D
plt.scatter(normal_data['Time'], normal_data['Current'], c='blue', label='Normal Operation (D)', alpha=0.5)
plt.scatter(fault_data['Time'], fault_data['Current'], c='red', label='Fault (D)', alpha=0.5)


plt.title('Scatter Plot of Current Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

df = test_data
fault_start = 2.1
fault_end = 2.4
# Separate the data points
fault_data = df[(df['Time'] >= fault_start) & (df['Time'] <= fault_end)]
normal_data = df[(df['Time'] < fault_start) | (df['Time'] > fault_end)]

plt.figure(figsize=(40, 6))

# Plotting for column D
plt.scatter(normal_data['Time'], normal_data['Current'], c='blue', label='Normal Operation (D)', alpha=0.5)
plt.scatter(fault_data['Time'], fault_data['Current'], c='red', label='Fault (D)', alpha=0.5)


plt.title('Scatter Plot of Current Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

# Define segmenting and labeling function

def segment_labeling(data, window, overlap, time1, time2):

  # Define the number of data points per segment = window size

  #index determines the start of a window
  #in each step of segmenting loop
  index = 0

  #windolap incorporates overlaping percentage
  windolap = math.floor (window * overlap)

  # Create an empty DataFrame for storing the labels
  labels_df = pd.DataFrame(columns=['label'])

  time_series = []

  while (index + window) < len(data):
      # Extract a segment of data
      segment = data.iloc[index : (index+window)]

      # Labeling based on a given time (the oscillation time is given)
      if any((time1 <= t <= time2) for t in segment['Time']):
        label = 'oscillation'
      else:
        label = 'normal'

      time_series.append(segment['Current'])

      # Append the label to the labels DataFrame
      labels_df = pd.concat([labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)

      #Shifting the index forward by stride = window - windolap
      index += window - windolap

  # return lables_df as a DataFrame
  return time_series, labels_df

window = 200
overlap = 0.75

train_X, train_y = segment_labeling(training_data, window, overlap, 5.1, 5.4)
test_X, test_y = segment_labeling(test_data, window, overlap, 2.1, 2.4)

train_y.value_counts()

test_y.value_counts()

X_train = np.array(train_X)
X_test = np.array(test_X)

print(X_train.shape)
print(X_test.shape)

# Step 1: Fit a machine learning algorithm of your choice to the training data
# I will be using a support vector machine (SVM) for this example
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Create an instance of the SVM model
svm = SVC(random_state=42)

# Ensure y is a 1D array
train_y = train_y.values.ravel()
test_y = test_y.values.ravel()

# Split the test data into validation and test sets
X_val, X_test_final, y_val, y_test_final = train_test_split(X_test, test_y, test_size=0.5, random_state=42)

# Fit the model to the training data
svm.fit(X_train, train_y)

# Define the hyperparameters to tune
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}

best_score = 0
best_params = {}

# Store results for plotting
scores = []

# Iterate through hyperparameter combinations
for C in param_grid['C']:
    for gamma in param_grid['gamma']:
        # Create and train the model
        model = SVC(C=C, gamma=gamma, random_state=42, class_weight='balanced')
        model.fit(X_train, train_y)

        # Evaluate on the validation set
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        # Store the score
        scores.append((C, gamma, score))

        # Update best score and params if current is better
        if score > best_score:
            best_score = score
            best_params = {'C': C, 'gamma': gamma}

print(f"Best hyperparameters: {best_params}")
print(f"Best validation accuracy: {best_score}")

# Reshape scores for plotting to show the 3D plane
scores = np.array(scores)
C_values = scores[:, 0]
gamma_values = scores[:, 1]
accuracy_scores = scores[:, 2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(C_values, gamma_values, accuracy_scores)

# Highlight the best hyperparameter combination
ax.scatter(best_params['C'], best_params['gamma'], best_score, color='red', marker='o', s=100)

ax.set_xlabel('C')
ax.set_ylabel('gamma')
ax.set_zlabel('Accuracy')
ax.set_title('Hyperparameter Tuning Results')

plt.show()

# Train the final model with the best hyperparameters on the combined training and validation data
final_model = SVC(**best_params, random_state=42, class_weight='balanced')
final_model.fit(np.concatenate((X_train, X_val)), np.concatenate((train_y, y_val)))

# Make predictions on the test data
predictions = final_model.predict(X_test)

# Step 4: Evaluate the model
from sklearn.metrics import classification_report

# Print the classification report
print(classification_report(test_y, predictions))