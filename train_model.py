from model_utils import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np

MAX_FRAMES = 600

# Load the data
data = np.load('./data/data_arrays.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Model definition
model = create_model()

# Model compilation
model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

# Model training
model.fit(X_train, y_train,
          batch_size=64,
          epochs=10,
          verbose=1,
          validation_data=(X_test, y_test))

# Save model weights only
model.save_weights('./models/my_model_weights.keras')


# Model evaluation
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Model prediction
predictions = model.predict(X_test[:5])
print('Predicted values:', np.argmax(predictions, axis=1))
print('Actual values:', y_test[:5])

