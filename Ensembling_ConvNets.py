'''
Example from
https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb
'''

from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10

import numpy as np


def conv_pool_cnn(model_input):
    '''
    First model: ConvPool-CNN-C
    Paper: [Springenberg et al., 2015, Striving for Simplicity: The All Convolutional Net].
    Its descritption appears on page 4 of the linked paper https://arxiv.org/abs/1412.6806v3.
    :param model_input:
    :return:
    '''
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='conv_pool_cnn')

    return model


def all_cnn(model_input):
    '''
    Second model: ALL-CNN-C
    Paper: [Springenberg et al., 2015, Striving for Simplicity: The All Convolutional Net]
    link to paper: https://arxiv.org/abs/1412.6806
    :param model_input:
    :return:
    '''
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='all_cnn')

    return model


def nin_cnn(model_input):
    '''
    Third Model: Network In Network CNN
    Paper: [Lin et al., 2013, Network In Network]
    Link: https://arxiv.org/abs/1312.4400
    :param model_input:
    :return:
    '''
    # mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu', padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    # mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    # mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='nin_cnn')

    return model


def compile_and_train(model, num_epochs, x_train, y_train):
    '''
    For simplicity's sake, each model is compiled and trained using the same parameters.
    Using 20 epochs with a batch size of 32 (1250 steps per epoch) seems sufficient for any of the three models
    to get to some local minima. Randomly chosen 20% of the training dataset is used for validation.
    :param model:
    :param num_epochs:
    :return:
    '''
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    history = model.fit(x=x_train, y=y_train, batch_size=32,
                        epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history


def evaluate_error(model, x_test, y_test):
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error


def ensemble(models, model_input):
    '''
    Ensemble function: Ensemble model definition is very straightforward.
    It uses the same input layer thas is shared between all previous models.
    In the top layer, the ensemble computes the average of three models' outputs by using Average() merge layer.
    :param models:
    :param model_input:
    :return:
    '''

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')

    return model


def main():
    print("Ensamble Learning")
    #  Loading Trainning and Testing Dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = to_categorical(y_train, num_classes=10)  # One hot encoder

    print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape,
                                                                                                y_train.shape,
                                                                                                x_test.shape,
                                                                                                y_test.shape))
    input_shape = x_train[0, :, :, :].shape
    model_input = Input(shape=input_shape)

    # Training First ConvNet
    print('Training First CONVNET')
    conv_pool_cnn_model = conv_pool_cnn(model_input)
    _ = compile_and_train(conv_pool_cnn_model, num_epochs=20, x_train=x_train, y_train=y_train)
    error1 = evaluate_error(conv_pool_cnn_model, x_test, y_test)
    print("First ConvNet error : ", error1)

    # Training Second ConvNet
    print('Training Second CONVNET')
    all_cnn_model = all_cnn(model_input)
    _ = compile_and_train(all_cnn_model, num_epochs=20, x_train=x_train, y_train=y_train)
    error2 = evaluate_error(all_cnn_model, x_test, y_test)
    print("Second ConvNet error:", error2)

    # Training Third ConvNet
    print('Training Third CONVNET')
    nin_cnn_model = nin_cnn(model_input)
    _ = compile_and_train(nin_cnn_model, num_epochs=20, x_train=x_train, y_train=y_train)
    error3 = evaluate_error(nin_cnn_model, x_test, y_test)
    print("Third ConvNet Error: ", error3)

    # Loading Best Weights for each base Classifier
    print("Loading Weights")
    conv_pool_cnn_model.load_weights('weights/conv_pool_cnn.29-0.10.hdf5')
    all_cnn_model.load_weights('weights/all_cnn.30-0.08.hdf5')
    nin_cnn_model.load_weights('weights/nin_cnn.30-0.93.hdf5')

    models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]

    print("Ensemble Model Performance")
    ensemble_model = ensemble(models, model_input)
    error_ensemble = evaluate_error(ensemble_model)
    print("Ensemble Error: ", error_ensemble)


if __name__ == '__main__':
    main()