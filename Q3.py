import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.regularizers import L1L2
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import gradient_descent_v2
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score

# Plot the prediction result of binary classification
def plot_binary_result(yPred):

  plt.figure(figsize=(8,8))
  plt.plot(yPred, color = 'aquamarine', label = 'Predicted decision')
  plt.title('Decision Prediction')
  plt.legend()
  st.pyplot(plt)
  plt.clf() 

# Plot the prediction result of multiclass classification
def plot_multiclass_result(yPred):

  plt.figure(figsize=(8,8))
  plt.scatter(*zip(*yPred), label = 'Predicted score')
  plt.title('Score Prediction')
  plt.legend()
  st.pyplot(plt)
  plt.clf() 

st.title('Loan Application Modeling')
st.write('')
df = pd.read_csv('Bank_CreditScoring.csv')
st.write('Descriptive Statistics of dataset: ')
# Display descriptive statistics of dataset
st.write(df.describe())

st.write('Characteristics of dataset: ')
# Plot out a chart that shows the score, loan amount and the decision made on the data 
data_char1 = sns.catplot(x="Score", y="Loan_Amount", hue="Decision", kind="swarm", data=df,
palette=sns.color_palette(['hotpink', 'deepskyblue']), height=7, aspect=6/6)
st.pyplot(data_char1)
plt.clf()

# Plot out the relationship between total sum of loan and the decision made. It is classified into different employment types 
data_char2 = sns.catplot(data=df, kind="bar", x="Employment_Type", y="Total_Sum_of_Loan", 
palette=sns.color_palette(['hotpink', 'deepskyblue']), hue="Decision", height=7, aspect=8/8)
st.pyplot(data_char2)
plt.clf()

st.header('1. Binary Classification')
st.write('Classification of the loan decision (Accept/ Reject) from the dataset.')
st.write('Independent variable: all columns except for Decision')
st.write('Dependent variable: Decision')

# In binary classification, independent variables are all columns except for the Decision
# Dependent variable is Decision, which is either Accept or Reject (binary)
X = df.drop(columns=['Decision']).values
y = df['Decision'].values
y = y.reshape(y.shape[0],-1)

# Splitting the dataset into training and test set
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, train_size = 0.8, random_state = 0)

# Ordinal Encoder is used as categorical encoding for input variables
# Convert categorical columns to numerical so that the model can understand
ct = ColumnTransformer([('XTrain', OrdinalEncoder(), [1, 4, 12, 14])], remainder ='passthrough')
XTrain = ct.fit_transform(XTrain)
XTest = ct.transform(XTest)

# Label Encoder is used as categorical encoding for target variable
# Convert categorical column to numerical so that the model can understand
le = LabelEncoder()
yTrain = le.fit_transform(yTrain)
yTest = le.transform(yTest)

# Normalize dataset by rescaling or transforming each feature into a range of 0-1
scaler = MinMaxScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform(XTest)

# Convert input variables into float32 because Keras model accepts numpy array of this type
# int or float can be incompatible
XTrain = np.asarray(XTrain).astype('float32')
XTest = np.asarray(XTest).astype('float32')

# Create Sequential model
model = Sequential()

n_cols = XTrain.shape[1]
# Add multiple layers
# For 1st Dense layer, L1 and L2 regularization is applied to avoid overfitting
# Dropout layer are used to randomly sets input units to 0 with rate of 0.12
# To set the initial weight of Keras layers, HeNormal is chosen
model.add(Dense(150, activation='relu', input_shape=(n_cols,), kernel_initializer='he_normal', kernel_regularizer=L1L2(0.007, 0.007)))
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.12))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.12))
model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.12))
model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.12))
# Since the target variable is of binary type, sigmoid  is chosen as the activation function
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Set early stopping monitor so the model stops training when it won't improve anymore for 7 epoch
early_stopping_monitor = EarlyStopping(patience=7)
# Fit the model into traning data
# 0.3 of the training data will be used as validation data
history = model.fit(XTrain, yTrain, epochs=100, callbacks=[early_stopping_monitor], shuffle=True, validation_split=0.3, verbose=2, batch_size=64)

# Plot model training and testing accuracy values
plt.figure(figsize=(8,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
st.pyplot(plt)
plt.clf() 

# Plot model training and testing loss values
plt.figure(figsize=(8,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
st.pyplot(plt)
plt.clf() 

# Plot the training and validation accuracy curves
plt.figure(figsize=(8,8))
plt.plot(history.history['accuracy'],'r')
plt.plot(history.history['val_accuracy'],'b')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epochs ')
plt.ylabel('Accuracy')
plt.ylim((0, 1))
plt.title('Accuracy Curves')
plt.grid(True)
st.pyplot(plt)
plt.clf() 

# Evaluate model accuracy on the test set
st.write('Evaluation: ')
test_results = model.evaluate(XTest, yTest, verbose=1)
st.write('Loss: ', test_results[0]) 
st.write('Accuracy: ', test_results[1])

# Predict and plot
yPred = model.predict(XTest)
plot_binary_result(yPred)

st.header('2. Multiclass Classification')
st.write('Classification of the Score, provided with the Loan Amount from the dataset.')
st.write('Independent variable: Loan Amount')
st.write('Dependent variable: Score')

# In multiclass classification, independent variable is Loan Amount and dependent variable is Score 
X = df[['Loan_Amount']].values
y = df['Score'].values
y = y.reshape(y.shape[0],-1)

# Label Encoder is used as categorical encoding for target variable
# Encode class value of score as integer
le = LabelEncoder()
y = le.fit_transform(y)
# Converted into a matrix which has binary values and has columns equal to the number of categories in the data
y_one_hot = np_utils.to_categorical(y)

# Splitting the dataset into training and test set
XTrain, XTest, yTrain, yTest = train_test_split(X, y_one_hot, test_size = 0.2, train_size = 0.8, random_state = 0)

# Standardise data by rescaling the distribution of values so that the mean of observed values is 0 
# and the standard deviation is 1.
scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform(XTest)

# Convert input variables into float32 because Keras model accepts numpy array of this type
# int or float can be incompatible
XTrain = np.asarray(XTrain).astype('float32')
XTest = np.asarray(XTest).astype('float32')

# Create model
model = Sequential()

n_cols = XTrain.shape[1]
# Add multiple Dense layers
# Dropout layers are used to randomly sets input units to 0 with rate of 0.15
# To set the initial weight of Keras layers, "normal" is chosen
model.add(Dense(150, activation='relu', input_shape=(n_cols,), kernel_initializer='normal'))
model.add(Dense(100, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.15))
model.add(Dense(100, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.15))
model.add(Dense(50, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.15))
model.add(Dense(20, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.15))
# Since the target is Score with more than 2 classes, softmax is used 
model.add(Dense(4, activation='softmax'))

# Compile model
model.compile(optimizer=gradient_descent_v2.SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# Set early stopping monitor so the model stops training when it won't improve anymore for 7 epoch
early_stopping_monitor = EarlyStopping(patience=7)
# Fit the model into traning data
# 0.3 of the training data will be used as validation data
history = model.fit(XTrain, yTrain, epochs=100, callbacks=[early_stopping_monitor], shuffle=True, validation_split=0.3, verbose=2, batch_size=64)

# Plot training and testing accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
st.pyplot(plt)
plt.clf() 

# Plot training and testing loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
st.pyplot(plt)
plt.clf() 

# Plot the training and validation accuracy curves
plt.plot(history.history['accuracy'],'r')
plt.plot(history.history['val_accuracy'],'b')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epochs ')
plt.ylabel('Accuracy')
plt.ylim((0, 1))
plt.title('Accuracy Curves')
plt.grid(True)
st.pyplot(plt)
plt.clf() 

# Evaluate model accuracy on the test set
st.write('Evaluation: ')
test_results = model.evaluate(XTest, yTest, verbose=1)
st.write('Loss: ', test_results[0]) 
st.write('Accuracy: ', test_results[1])

# Predict and plot
yPred = model.predict(XTest)
plot_multiclass_result(yPred)

st.header('3. K-Means Clustering')
choice = st.selectbox('Choose the k value for K-Means:', ('k = 2', 'k = 3', 'k = 4'))

# MinMaxScaler scales and translates each feature individually such that it is in the given range on the training set
# Applied on Loan Amount and Monthly Salary
scaler = MinMaxScaler()
scaler.fit(df[['Loan_Amount']])
df['Loan_Amount'] = scaler.transform(df[['Loan_Amount']])
scaler.fit(df[['Monthly_Salary']])
df['Monthly_Salary'] = scaler.transform(df[['Monthly_Salary']])

# Plot the scatter data of Monthly Salary and Loan Amount before clustering
plt.scatter(df['Monthly_Salary'], df['Loan_Amount'])
plt.title('Loan Amount and Monthly Salary Before Clustering')
st.pyplot(plt)
plt.clf() 

st.write('Cluster Analysis on the Loan Amount and Monthly Salary.')

# Perform K-Means and plot the resulting clusters according to user's choice (k value)
# V-measure is also done on each clusters formed
# The score from V-measure is a measure between 0–1 that quantifies the goodness of the clustering partition
if choice == 'k = 2':
  kmeans = KMeans(n_clusters=2, random_state=0)
  y_predicted = kmeans.fit_predict(df[['Monthly_Salary', 'Loan_Amount']])
  df['cluster'] = y_predicted
  df1 = df[df.cluster==0]
  df2 = df[df.cluster==1]
  plt.scatter(df1.Monthly_Salary,df1['Loan_Amount'],color='green', label='Cluster 1')
  plt.scatter(df2.Monthly_Salary,df2['Loan_Amount'],color='peru', label = 'Cluster 2')
  plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],color='black', marker='*',label='Centroid')
  plt.title('K-Means Clustering (k=2)')
  plt.legend()
  st.pyplot(plt)
  plt.clf() 
  table = pd.DataFrame({'Cluster': ["Cluster 1", "Cluster 2"], 
  'V-measure value': [v_measure_score(df1['Monthly_Salary'], df1['Loan_Amount']), 
  v_measure_score(df2['Monthly_Salary'], df2['Loan_Amount'])]})
  table = table.astype(str)
  st.write(table)

elif choice == 'k = 3':
  kmeans = KMeans(n_clusters=3, random_state=0)
  y_predicted = kmeans.fit_predict(df[['Monthly_Salary', 'Loan_Amount']])
  df['cluster'] = y_predicted
  df1 = df[df.cluster==0]
  df2 = df[df.cluster==1]
  df3 = df[df.cluster==2]
  plt.scatter(df1.Monthly_Salary,df1['Loan_Amount'],color='green', label='Cluster 1')
  plt.scatter(df2.Monthly_Salary,df2['Loan_Amount'],color='peru', label = 'Cluster 2')
  plt.scatter(df3.Monthly_Salary,df3['Loan_Amount'],color='deepskyblue', label='Cluster 3')
  plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],color='black', marker='*',label='Centroid')
  plt.title('K-Means Clustering (k=3)')
  plt.legend()
  st.pyplot(plt)
  plt.clf()
  table = pd.DataFrame({'Cluster': ["Cluster 1", "Cluster 2", "Cluster 3"], 
  'V-measure value': [v_measure_score(df1['Monthly_Salary'], df1['Loan_Amount']), 
  v_measure_score(df2['Monthly_Salary'], df2['Loan_Amount']), v_measure_score(df3['Monthly_Salary'], df3['Loan_Amount'])]})
  table = table.astype(str)
  st.write(table)

elif choice == 'k = 4':
  kmeans = KMeans(n_clusters=4, random_state=0)
  y_predicted = kmeans.fit_predict(df[['Monthly_Salary', 'Loan_Amount']])
  df['cluster'] = y_predicted
  df1 = df[df.cluster==0]
  df2 = df[df.cluster==1]
  df3 = df[df.cluster==2]
  df4 = df[df.cluster==3]
  plt.scatter(df1.Monthly_Salary,df1['Loan_Amount'],color='green', label='Cluster 1')
  plt.scatter(df2.Monthly_Salary,df2['Loan_Amount'],color='peru', label = 'Cluster 2')
  plt.scatter(df3.Monthly_Salary,df3['Loan_Amount'],color='deepskyblue', label='Cluster 3')
  plt.scatter(df4.Monthly_Salary,df4['Loan_Amount'],color='lightcoral', label='Cluster 4')
  plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],color='black', marker='*',label='Centroid')
  plt.title('K-Means Clustering (k=4)')
  plt.legend()
  st.pyplot(plt)
  plt.clf()
  table = pd.DataFrame({'Cluster': ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"], 
  'V-measure value': [v_measure_score(df1['Monthly_Salary'], df1['Loan_Amount']), 
  v_measure_score(df2['Monthly_Salary'], df2['Loan_Amount']), v_measure_score(df3['Monthly_Salary'], df3['Loan_Amount']),
  v_measure_score(df4['Monthly_Salary'], df4['Loan_Amount'])]})
  table = table.astype(str)
  st.write(table)

# Alternatively, Elbow Method can be used to decide the best number of cluster
# Select the elbow point for number of clusters
data = pd.DataFrame({'x':df.Monthly_Salary, 'y':df.Loan_Amount})

sse = {}
for k in range(1,6): 
  kmeans = KMeans(n_clusters=k, max_iter=100).fit(data)
  data['clusters'] = kmeans.labels_
  sse[k] = kmeans.inertia_

# Plot out the Elbow Method
# The ideal number of clusters is 4
plt.figure()
plt.plot(list(sse.keys()),list(sse.values()))
plt.xlabel("Number of Cluster")
plt.ylabel("SSE / WCSS")
plt.title("Elbow Method")
st.pyplot(plt)
plt.clf()
