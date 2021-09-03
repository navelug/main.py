from silence_tensorflow import silence_tensorflow; silence_tensorflow()
import numpy
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.preprocessing import minmax_scale
from keras.layers import Dense, Dropout
from keras import regularizers
import matplotlib.pyplot as plt
import confusionMatrixPlotter
from getDB import getDB
from Augmentation import Augmentation

print()

# Configure database parameters
gestures_num = 10
participants_num = 14

# Get database, normalize, divide to partitions and augment training data
samples, labels = getDB(gestures_num, participants_num)
samples = preprocessing.MinMaxScaler().fit_transform(samples).tolist()
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(samples, labels, test_size=0.3) # Device database to train (70%), test (15%) and valid (15%) partitions
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
Augmentation(X_train, Y_train, [-30, -15, 15, 30], [0.05, 0.1]) # augmentation

X_train = numpy.array(X_train); Y_train = numpy.array(Y_train)
X_test = numpy.array(X_test); Y_test = numpy.array(Y_test)
X_val = numpy.array(X_val); Y_val = numpy.array(Y_val)
print("\u0332".join('Train data:'), X_train.shape, 'Samples,', len(Y_train), ' Labels')
print("\u0332".join('Test data:'), X_test.shape, 'Samples,', len(Y_test), ' Labels')
print("\u0332".join('Validation data:'), X_val.shape, 'Samples,', len(Y_val), ' Labels'); print()

# Build, compile and train the model
model = Sequential([
    Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(0.001), input_shape=(63,)),
    Dropout(0.3),
    Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dense(gestures_num, activation='softmax')])
print("\u0332".join('Model Architecture')); model.summary(); print()
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']); print("\u0332".join('Model Training'));
hist = model.fit(X_train, Y_train, batch_size=40, epochs=150, validation_data=(X_val, Y_val)); print()

# Plot Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot Accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# Plot Confusion Matrix
Y_predicted = numpy.argmax(model.predict(X_test), axis=1)
confusion = confusion_matrix(Y_test, Y_predicted)
confusionMatrixPlotter.pretty_plot_confusion_matrix(confusion)
model.evaluate(X_test, Y_test)
model.save('model_NN.h5')

data = minmax_scale(numpy.load('tal_6.npy'))
print("\u0332".join('Prediction :'), int(numpy.argmax(model.predict(data.reshape(1, 63)), axis=1)) + 1)