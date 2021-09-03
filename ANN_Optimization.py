from silence_tensorflow import silence_tensorflow; silence_tensorflow()
import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
from getDB import getDB
from keras.wrappers.scikit_learn import KerasClassifier
from Augmentation import Augmentation

def DL_Model(activation1='relu',activation2='relu', neurons1=100, neurons2=100, dropout1=0.1 , dropout2=0.1):
    model = Sequential([
        Dense(neurons1, activation=activation1, input_shape=(63,)),
        Dropout(dropout1),
        Dense(neurons2, activation=activation2),
        Dropout(dropout2),
        Dense(gestures_num, activation='softmax')])
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print()

# Configure database parameters
gestures_num = 10
participants_num = 14

# Get database, divide to partitions, augment and normalize samples
samples, labels = getDB(gestures_num, participants_num)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(samples, labels, test_size=0.3) # Device database to train (70%), test (15%) and valid (15%) partitions
X_val_and_test = preprocessing.MinMaxScaler().fit_transform(X_val_and_test)
Augmentation(X_train, Y_train, [-60, -30, 30, 60], [0.1]) # augmentation
X_train = preprocessing.MinMaxScaler().fit_transform(X_train)

X_train = numpy.array(X_train); Y_train = numpy.array(Y_train)
X_val_and_test = numpy.array(X_val_and_test); Y_val_and_test = numpy.array(Y_val_and_test)

clf = KerasClassifier(build_fn=DL_Model, verbose=0)

# Defining grid parameters
activation1 = ['tanh']
activation2 = ['tanh']
neurons1 = [150]
neurons2 = [150]
dropout1 = [0.2]
dropout2 = [0.2]
batch_size = [30]
epochs = [50]
param_grid = dict(activation1=activation1, activation2=activation2, neurons1=neurons1, neurons2=neurons2, dropout1=dropout1, dropout2=dropout2, batch_size=batch_size, epochs=epochs)

model = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1)
grid_result = model.fit(X_train, Y_train, validation_data=(X_val_and_test, Y_val_and_test))

# summarize results
means = model.cv_results_['mean_test_score']
params = model.cv_results_['params']
for mean, param in zip(means, params):
    print("acc %f with: %r" % (mean, param))
print("Best: %f using %s" % (model.best_score_, model.best_params_))