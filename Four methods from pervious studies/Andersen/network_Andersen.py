import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


# CNN-LSTM Model
def cnn_lstm(learning_rate=0.0013, momentum_coef=0.99, regularization_term=0.000017):
    regularizer = tf.keras.regularizers.L2(
        l2=regularization_term
        )
    
    model = Sequential()
    model.add(Input(shape=(30,1)))
    model.add(Conv1D(60, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizer))
    model.add(Conv1D(80, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizer))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, 
        momentum=momentum_coef,
        nesterov=True, 
        name='SGD'
        )
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='Acc'), 
        tf.keras.metrics.TruePositives(name='TP'), 
        tf.keras.metrics.TrueNegatives(name='TN'), 
        tf.keras.metrics.FalsePositives(name='FP'), 
        tf.keras.metrics.FalseNegatives(name='FN')
        ]
    model.compile(
        optimizer=optimizer, 
        loss=loss,
        metrics=metrics
        )
    return model