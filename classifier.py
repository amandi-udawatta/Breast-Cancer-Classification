import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns


# Fix randomness across libraries
tf.random.set_seed(2)
np.random.seed(2)
random.seed(2)

################################ Model creation #################################

# Load the breast cancer dataset and store in a dataframe
data_frame = pd.read_csv('data.csv')

#add target columns to data frame
data_frame['label'] = data_frame['diagnosis'].map({'M': 1, 'B': 0})

# handle missing values
data_frame = data_frame.dropna(axis=1)

# drop the columns that are not required
data_frame = data_frame.drop(columns=['id', 'diagnosis'], axis=1)

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
#when u want to drop a column = axis=1, if row, axis=0

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#test size will be 20% of the data, random state will decide how to split it, just a random number

# standardize the feature data
scalar = StandardScaler()
X_train_std = scalar.fit_transform(X_train)
X_test_std = scalar.transform(X_test)

#setting up layers of NN
model = keras.Sequential([  keras.layers.Flatten(input_shape=(30,)),
                            keras.layers.Dense(20, activation='relu'),
                            keras.layers.Dense(2, activation='sigmoid')])
#input layer flatten, hidden and output dense
#dense means all neurons in the previous layer are connected to all neurons in the current layer
#flatten means we flat out input matrix to single dimensional array
#we give each feature a neuron at input layer. therefore #neurons = #features

#compile the neural network
model.compile(  optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#training the NN
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
#epoch is the number of times the model will see the same data

# Predictions
Y_pred = model.predict(X_test_std)
Y_pred_labels = [np.argmax(i) for i in Y_pred]
#argmax returns the index of the max value in the array

#################### Evaluate the model on the test set ##########################

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# model accuracy
axs[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
axs[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[0, 0].set_title('Model Accuracy')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend(loc='lower right')

# confusion Matrix
cm = confusion_matrix(Y_test, Y_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0, 1], xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
axs[0, 1].set_title("Confusion Matrix")
axs[0, 1].set_xlabel("Predicted")
axs[0, 1].set_ylabel("Actual")

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(Y_test, Y_pred[:, 1])
area_under_the_curve = auc(fpr, tpr)
#auc closer to 1 is a strong classifier, if its closer to 0.5, weak classifier

axs[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % area_under_the_curve)
axs[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1, 0].set_xlim([0.0, 1.0])
axs[1, 0].set_ylim([0.0, 1.05])
axs[1, 0].set_title('Receiver Operating Characteristic (ROC)')
axs[1, 0].set_xlabel('False Positive Rate')
axs[1, 0].set_ylabel('True Positive Rate')
axs[1, 0].legend(loc="lower right")

# Correct and Incorrect Predictions
tp = cm[1, 1]  # True Positives
tn = cm[0, 0]  # True Negatives
fp = cm[0, 1]  # False Positives
fn = cm[1, 0]  # False Negatives

correct = tp + tn
incorrect = fp + fn

axs[1, 1].bar(['Correct Predictions', 'Incorrect Predictions'], [correct, incorrect], color=['green', 'red'])
axs[1, 1].set_title('Correct vs Incorrect Predictions')
axs[1, 1].set_ylabel('Count')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # Add more space between rows to avoid overlap
plt.show()

# Display the AUC score
print(f'ROC Area Under the Curve Score: {area_under_the_curve:.2f}')

#####################build predictive model###########################################

#give input here
input_data = (8.618,11.79,54.34,224.5,0.09752,0.05272,0.02061,0.007799,0.1683,0.07187,0.1559,0.5796,1.046,8.322,0.01011,0.01055,0.01981,0.005742,0.0209,0.002788,9.507,15.4,59.9,274.9,0.1733,0.1239,0.1168,0.04419,0.322,0.09026
)
#change input_data to a numpy array
input_data = np.asarray(input_data)
#reshape input_data as we are predicting for one data point
input_data = input_data.reshape(1, -1)
#standardize input_data
input_data = scalar.transform(input_data)

#predict the input_data
prediction = model.predict(input_data)
prediction = [np.argmax(prediction)]

if (prediction[0]==0):
    print('The Tumour is Benign')
else:
    print('The Tumour is Malignant')