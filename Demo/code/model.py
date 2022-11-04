
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def add_conv_weight(layer, filter_length, num_filters, subsample_length=1, normal=False):
    if not normal:
        layer = KL.Conv1D(
            filters=num_filters,
            kernel_size=filter_length,
            strides=subsample_length,
            padding='same',
            kernel_initializer='he_normal',
            activation='relu')(layer)
        return layer
    else:
        layer = KL.Conv1D(
            filters=num_filters,
            kernel_size=filter_length,
            strides=subsample_length,
            padding='same',
            kernel_initializer='he_normal')(layer)
        layer = KL.LayerNormalization()(layer)
        layer = KL.Activation('relu')(layer)
        return layer


def build_model_lstm(sample_shape=(90, 1)):
    gen_input = KL.Input(shape=sample_shape, name='z_input')
    mid = add_conv_weight(gen_input, 5, 32)
    mid = add_conv_weight(mid, 5, 32, normal=True)
    mid = KL.Dropout(0.2)(mid)
    mid = add_conv_weight(mid, 5, 32, normal=True)
    mid = KL.Dropout(0.2)(mid)

    d1 = KL.Bidirectional(KL.LSTM(units=16, return_sequences=True))(mid)

    mid = add_conv_weight(mid, 5, 64, normal=True)
    mid = KL.Dropout(0.2)(mid)
    mid = add_conv_weight(mid, 5, 64, normal=True)
    mid = KL.Dropout(0.2)(mid)

    d2 = KL.Bidirectional(KL.LSTM(units=32, return_sequences=True))(mid)

    mid = add_conv_weight(mid, 5, 128, normal=True)
    mid = KL.Dropout(0.2)(mid)
    mid = add_conv_weight(mid, 5, 128, normal=True)
    mid = KL.Dropout(0.2)(mid)

    d3 = KL.Bidirectional(KL.LSTM(units=64, return_sequences=True))(mid)

    mid = add_conv_weight(mid, 5, 256, normal=True)
    mid = KL.Dropout(0.2)(mid)
    mid = add_conv_weight(mid, 5, 256, normal=True)
    mid = KL.Dropout(0.2)(mid)

    d4 = mid

    mid = add_conv_weight(d4, 5, 512, normal=True)
    mid = KL.Dropout(0.2)(mid)

    p1 = add_conv_weight(mid, 5, 256)

    mid = KL.concatenate([d4, p1])
    mid = add_conv_weight(mid, 5, 256, normal=True)
    mid = KL.Dropout(0.2)(mid)

    p2 = add_conv_weight(mid, 5, 128)

    mid = KL.concatenate([d3, p2])
    mid = add_conv_weight(mid, 5, 128, normal=True)
    mid = KL.Dropout(0.2)(mid)

    p3 = add_conv_weight(mid, 5, 64)

    mid = KL.concatenate([d2, p3])
    mid = add_conv_weight(mid, 5, 64, normal=True)
    mid = KL.Dropout(0.2)(mid)

    p4 = add_conv_weight(mid, 5, 32)

    mid = KL.concatenate([d1, p4])
    mid = add_conv_weight(mid, 5, 32, normal=True)
    mid = KL.Dropout(0.2)(mid)

    mid = add_conv_weight(mid, 5, 32)
    mid = KL.Bidirectional(KL.LSTM(units=32, return_sequences=True))(mid)
    output = KL.Conv1D(1, 5, padding='same', kernel_initializer='he_normal',
                       activation='sigmoid')(mid)

    generator = KM.Model(inputs=[gen_input], outputs=[output], name='gen_model')
    opt_generator = Adam()
    generator.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=opt_generator,
                      metrics=['accuracy'])
    return generator


if __name__ == '__main__':
    model = build_model_lstm((None, 1))
    print(model.summary())
