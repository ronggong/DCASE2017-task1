from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, Merge, ELU, GRU, Permute, Masking, BatchNormalization
from keras.regularizers import l2


def modelVGG(input_shape, filter_density, include_top=True):
    model = Sequential()

    freq_axis = 2
    channel_axis = 3
    # model.add(BatchNormalization(axis=freq_axis, input_shape=input_shape, name='bn0'))
    # conv 0
    model.add(Conv2D(int(32 * filter_density), (3,3), padding="valid",
                     input_shape=input_shape, data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn1'))
    model.add(ELU())

    # model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", data_format="channels_last"))

    # conv 1
    model.add(Conv2D(int(32 * filter_density), (3,3), padding="valid",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn2'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", data_format="channels_last"))
    model.add(Dropout(0.3))

    # conv 2
    model.add(Conv2D(int(64 * filter_density), (3,3), padding="valid",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn3'))

    model.add(ELU())

    # model.add(MaxPooling2D(pool_size=((3,3)), padding="valid", data_format="channels_last"))

    # conv 3
    model.add(Conv2D(int(64 * filter_density), (3,3), padding="valid",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn4'))

    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", data_format="channels_last"))
    model.add(Dropout(0.3))

    # conv 4
    model.add(Conv2D(int(128 * filter_density), (3,3), padding="valid",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn5'))

    model.add(ELU())

    # model.add(MaxPooling2D(pool_size=((3,3)), padding="valid", data_format="channels_last"))

    # conv 5
    model.add(Conv2D(int(128 * filter_density), (3,3), padding="valid",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn6'))

    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", data_format="channels_last"))
    model.add(Dropout(0.3))

    # print(model.output_shape)

    model.add(Flatten())

    # model.add(Dense(output_dim=128, kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    # model.add(ELU())
    model.add(Dropout(0.5))

    if include_top:
        model.add(Dense(15, activation='softmax', name='output'))

    optimizer = Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


    return model

def modelVGGJKU(input_shape, filter_density, include_top=True):
    model = Sequential()

    freq_axis = 2
    channel_axis = 3

    # conv 0
    model.add(Conv2D(int(32 * filter_density), (5,5), padding="same",
                     input_shape=input_shape, data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn1'))
    model.add(ELU())

    # conv 1
    model.add(Conv2D(int(32 * filter_density), (3,3), padding="same",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn2'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", data_format="channels_last"))
    model.add(Dropout(0.3))

    # conv 2
    model.add(Conv2D(int(64 * filter_density), (3,3), padding="same",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn3'))

    model.add(ELU())

    # conv 3
    model.add(Conv2D(int(64 * filter_density), (3,3), padding="same",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn4'))

    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", data_format="channels_last"))
    model.add(Dropout(0.3))

    # conv 4
    model.add(Conv2D(int(128 * filter_density), (3,3), padding="same",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn5'))

    model.add(ELU())


    # conv 5
    model.add(Conv2D(int(128 * filter_density), (3,3), padding="same",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn6'))

    model.add(ELU())

    # conv 6
    model.add(Conv2D(int(128 * filter_density), (3,3), padding="same",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn7'))

    model.add(ELU())

    # conv 7
    model.add(Conv2D(int(128 * filter_density), (3,3), padding="same",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn8'))

    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", data_format="channels_last"))
    model.add(Dropout(0.3))

    # conv 8
    model.add(Conv2D(int(512 * filter_density), (3,3), padding="valid",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn9'))
    model.add(Dropout(0.5))


    # conv 9
    model.add(Conv2D(int(512 * filter_density), (1,1), padding="valid",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn10'))
    model.add(Dropout(0.5))

    # conv 10
    model.add(Conv2D(int(15 * filter_density), (1, 1), padding="valid",
                     data_format="channels_last",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model.add(BatchNormalization(axis=channel_axis, name='bn11'))

    model.add(GlobalAveragePooling2D(data_format="channels_last"))

    print(model.output_shape)

    # model.add(Flatten())

    # model.add(Dense(output_dim=128, kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    # model.add(ELU())
    # model.add(Dropout(0.5))

    if include_top:
        model.add(Dense(15, activation='softmax', name='output'))

    optimizer = Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


    return model

def modelMultiFilters(input_shape):

    arc_params = {}
    # pre-def params
    arc_params['n_frames'] = 43
    arc_params['n_mel'] = 96
    arc_params['class_count'] = 15

    # number of filters in 1L
    arc_params['n_filters_l1'] = [32, 32, 48]

    # dfine different shapes in the 1L
    arc_params['kernel_size_l1'] = [(3, 60),
                                    (3, 30),
                                    (3, 10)]

    # always the same
    # max pooling shapes in 1L, in tuple ()
    arc_params['pool_size_l1'] = (5, 5)

    # number of filters in 2L
    arc_params['n_filters_l2'] = 256

    # dfine shape in the 2L
    arc_params['kernel_size_l2'] = (5, 7)

    arc_params['pool_size_l2'] = (4, 4)

    # --

    # input_shape = (43, 96, 1)
    channel_axis = 3 # for BN

    # this model is valid for any number of different shapes, which is hardcoded in the yaml and defined previously

    # instantiate input layer
    tf_input = Input(shape=input_shape)

    convs_1L = []  # empty list to append the diferent filter shapes' layers

    for i, ksz in enumerate(arc_params['kernel_size_l1']):
        print(i)
        print(ksz)

        conv = Conv2D(arc_params['n_filters_l1'][i], arc_params['kernel_size_l1'][i],
                      padding='same',  # IMPORTANT; easy option for now, else we have different sizes
                      data_format='channels_last',
                      kernel_initializer='uniform')
        x = conv(tf_input)
        convs_1L.append(x)

    # out_1L = keras.layers.concatenate(convs_1L, axis=channel_axis) error global name

    # so
    merge1 = Merge(mode='concat', concat_axis=channel_axis)
    out_1L = merge1(convs_1L)

    # ***got here the output of the several convolution layers with different filters, in parallel


    # create a model that takes the T-F representation as input and applies the distribution of filters as designed
    # This creates a model that includes the Input layer and the convLayers defined
    distrib_1L = Model(inputs=tf_input, outputs=out_1L)

    # create seq model, where to start adding the orevious layer and others
    model = Sequential()

    model.add(distrib_1L)
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(
        MaxPooling2D(pool_size=arc_params['pool_size_l1'], data_format="channels_last"))  # allows defining padding

    model.add(Conv2D(arc_params['n_filters_l2'], arc_params['kernel_size_l2'],
                     padding='valid',
                     data_format="channels_last",
                     kernel_initializer='uniform'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(
        MaxPooling2D(pool_size=arc_params['pool_size_l2'], data_format="channels_last"))  # allows defining padding

    model.add(Flatten())
    model.add(Dense(arc_params['class_count'], activation='softmax', kernel_initializer="uniform"))

    optimizer = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def model2Layers1Filter(input_shape):

    channel_axis = 3
    # create seq model, where to start adding the orevious layer and others
    model = Sequential()

    model.add(Conv2D(128, (5, 5), padding="valid",
                     input_shape=input_shape,data_format="channels_last",
                     kernel_initializer="uniform"))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(
        MaxPooling2D(pool_size=(5,5), data_format="channels_last"))  # allows defining padding

    model.add(Conv2D(256, (5,5),
                     padding='valid',
                     data_format="channels_last",
                     kernel_initializer='uniform'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(
        MaxPooling2D(pool_size=(2,3), data_format="channels_last"))  # allows defining padding

    model.add(Flatten())
    model.add(Dense(15, activation='softmax', kernel_initializer="uniform"))

    optimizer = Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
