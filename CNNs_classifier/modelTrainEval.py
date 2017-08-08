import numpy as np
import os
import pickle

from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from utilFunctions import printProgressBar


def featureOnlineSegmentation(X, y, input_shape):
    """
    Segment the feature [nFrame, nDims] into seg parts
    :param X:
    :param y:
    :param seg:
    :return:
    """

    X_out = []

    nFrames = input_shape[0]
    seg = int(X.shape[0] / nFrames)

    for ii in xrange(seg):
        X_out.append(X[ii * nFrames:(ii + 1) * nFrames, :])

    return X_out, [y] * seg

def shuffleFilenamesLabelsInUnison(filenames, labels):
    assert len(filenames) == len(labels)
    p=np.random.permutation(len(filenames))
    return filenames[p], labels[p]

def generator(filenames, scaler, number_of_batches, file_size, input_shape, labels=None, shuffle=True):

    # print(len(filenames))

    filenames_copy = np.array(filenames[:],dtype=object)

    if labels is not None:
        labels_copy = np.copy(labels)
        labels_copy = to_categorical(labels_copy)
    else:
        labels_copy = np.zeros((len(filenames_copy), 1))

    counter = 0
    # print(filenames)

    # test shuffle
    # filenames_copy, labels_copy = shuffleFilenamesLabelsInUnison(filenames_copy, labels_copy)
    # print(filenames_copy)
    # print(labels_copy)

    while True:
        idx_start = file_size * counter
        idx_end = file_size * (counter + 1)

        X_batch = []
        y_batch = []
        # print(idx_start)
        # print(idx_end)
        for ii, fn in enumerate(filenames_copy[idx_start:idx_end]):

            # print(fn)

            # path_feature_fn = os.path.join(path_feature, fn + '.pkl')
            feature = pickle.load(open(fn, "r"))
            # labels_block = labels[idx_start:idx_end, :]
            # preprocessing
            feature = scaler.transform(feature)
            y = labels_copy[idx_start + ii, :]

            # print(feature.shape, y)

            # number of segments
            seg = feature.shape[0] / float(input_shape[0])

            if seg > 1:
                feature_list, y = featureOnlineSegmentation(feature, y, input_shape)

                for ii_f, f in enumerate(feature_list):
                    X_batch.append(f)
                    y_batch.append(y[ii_f])
            elif seg == 1:
                X_batch.append(feature)
                y_batch.append(y)
            # we don't consider the case if feature dims is less than the batch dims

        X_batch_tensor = np.zeros((len(X_batch), input_shape[0], input_shape[1], 1), dtype='float32')
        y_batch_tensor = np.zeros((len(X_batch), labels_copy.shape[1]))

        for ii in xrange(len(X_batch)):
            X_batch_tensor[ii] = np.expand_dims(X_batch[ii], axis=2)
            y_batch_tensor[ii, :] = y_batch[ii]

        # print(counter, X_batch_tensor.shape)
        counter += 1

        yield X_batch_tensor, y_batch_tensor

        if counter >= number_of_batches:
            counter = 0
            if shuffle:
                filenames_copy, labels_copy = shuffleFilenamesLabelsInUnison(filenames_copy, labels_copy)

def scoreAverage(score_pred):
    return np.mean(score_pred,axis=0)

class ModelTrainEval(object):


    def __init__(self,
                 path_model,
                 scaler,
                 file_size=32,
                 number_of_batches=None,
                 preprocessing=True,
                 input_shape=None):
        self.path_model = path_model
        self.scaler = scaler
        self.file_size = file_size # batch size if segmentation is done
        self.number_of_batches = number_of_batches
        self.preprocessing = preprocessing
        self.input_shape = input_shape

    def trainModel(self,
                   model,
                   filename_model,
                   filename_log,
                   filenames_train,
                   labels_train,
                   filenames_val,
                   labels_val,
                   epochs=100):

        # train on training set

        # model.save_weights(os.path.join(self.path_model, 'initial_weights.h5'))
        #
        # patience = 10
        # callbacks = [EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        #              ModelCheckpoint(os.path.join(filename_model),
        #                              monitor='val_loss',
        #                              save_best_only=True),
        #              CSVLogger(filename=filename_log, separator=';')]
        #
        # steps_per_epoch_train = int(np.ceil(len(filenames_train) / self.file_size))
        # steps_per_epoch_val = int(np.ceil(len(filenames_val) / self.file_size))
        #
        # generator_train = generator(filenames_train,
        #                             self.scaler,
        #                             steps_per_epoch_train,
        #                             self.file_size,
        #                             self.input_shape,
        #                             labels=labels_train)
        # generator_val = generator(filenames_val,
        #                           self.scaler,
        #                           steps_per_epoch_val,
        #                           self.file_size,
        #                           self.input_shape,
        #                           labels=labels_val)
        #
        # history = model.fit_generator(generator=generator_train,
        #                                 steps_per_epoch=steps_per_epoch_train,
        #                                 epochs=epochs,
        #                                 validation_data=generator_val,
        #                                 validation_steps=steps_per_epoch_val,
        #                                 callbacks=callbacks,
        #                                 verbose=2)
        #
        # # train again use all train and validation set
        # epochs_final = len(history.history['val_loss'])-patience
        #
        # model.load_weights(os.path.join(self.path_model, 'initial_weights.h5'))

        callbacks = [CSVLogger(filename=filename_log, separator=';')]

        epochs_final = epochs

        steps_per_epoch_train_val = int(np.ceil((len(filenames_train)+len(filenames_val)) / self.file_size))

        generator_train_val = generator(filenames_train + filenames_val,
                                        self.scaler,
                                        steps_per_epoch_train_val,
                                        self.file_size,
                                        self.input_shape,
                                        labels=np.hstack((labels_train, labels_val)))

        model.fit_generator(generator=generator_train_val,
                            steps_per_epoch=steps_per_epoch_train_val,
                            epochs=epochs_final,
                            callbacks=callbacks,
                            verbose=2)

        model.save(os.path.join(filename_model))

    def testModel(self,
                  model,
                  filenames_test,
                  file_size_test):
        steps_per_epoch_test = int(np.ceil(len(filenames_test) / file_size_test))

        generator_test = generator(filenames_test,
                                   self.scaler,
                                   steps_per_epoch_test,
                                   file_size_test,
                                   self.input_shape)

        pred_proba_test = []
        pred_test = []
        printProgressBar(0, steps_per_epoch_test, prefix='Progress:', suffix='Complete', bar_length=50)
        for ii, (X, _) in enumerate(generator_test):
            # print('predicting', ii, 'in total', steps_per_epoch_test)
            y_pred = model.predict_on_batch(X)
            # print(X.shape)
            pred_proba_test.append(y_pred)
            print(y_pred.shape)
            # print(y_pred.shape)
            pred_test.append(np.argmax(scoreAverage(y_pred)))

            print(np.argmax(scoreAverage(y_pred)))
            # pred_test.append(np.argmax(y_pred, axis=1))
            # print(np.argmax(y_pred, axis=1))
            # print(y_pred)
            # print(pred_test[ii])
            printProgressBar(ii + 1, steps_per_epoch_test, prefix='Progress:', suffix='Complete', bar_length=50)

            if ii>=steps_per_epoch_test-1: break

        # pred_test = np.hstack(pred_test)
        return pred_proba_test, pred_test

    def metricsEval(self, y_test, y_pred):
        # print(classification_report(y_test, y_pred))
        # print confusion_matrix(y_test, y_pred)
        # print("Accuracy:")
        return accuracy_score(y_test, y_pred)
        # print("Micro stats:")
        # print(precision_recall_fscore_support(y_test, y_pred, average='micro'))
        # print("Macro stats:")
        # print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

        # print(y_pred)

