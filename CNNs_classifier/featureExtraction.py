import essentia.standard as ess
import numpy as np
import time
from multiprocessing import Process

class FeatureExtraction(object):


    def __init__(self,fs,frameSize,hopSize,highFrequencyBound,numberBands):
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.fs = fs
        self.highFrequencyBound = highFrequencyBound
        self.numberBands = numberBands
        self.essentiaObjectInit()

    def essentiaObjectInit(self):
        winAnalysis = 'hann'
        self.MFCC80 = ess.MFCC(sampleRate=self.fs,
                          highFrequencyBound=self.highFrequencyBound,
                          inputSize=self.frameSize + 1,
                          numberBands=self.numberBands)

        N = 2 * self.frameSize  # padding 1 time framesize
        self.SPECTRUM = ess.Spectrum(size=N)
        self.WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - self.frameSize)

    def getMFCCBands2D(self, audio):

        '''
        mel bands feature [p[0],p[1]]
        output feature for each time stamp is a 2D matrix
        it needs the array format float32
        :param audio:
        :param p:
        :return:
        '''

        mfcc   = []
        for frame in ess.FrameGenerator(audio, frameSize=self.frameSize, hopSize=self.hopSize):
            frame           = self.WINDOW(frame)
            mXFrame         = self.SPECTRUM(frame)
            bands,mfccFrame = self.MFCC80(mXFrame)
            mfcc.append(bands)

        # the mel bands features
        feature = np.array(mfcc,dtype='float32')

        return feature

    def log_mel_feature(self,full_path_fn):
        audio               = ess.MonoLoader(downmix = 'left', filename = full_path_fn, sampleRate = self.fs)()
        feature             = self.getMFCCBands2D(audio)
        feature             = np.log(100000 * feature + 1)
        return feature

def featureSegmentation(X, seg=10):
    """
    Segment the feature [nFrame, nDims] into seg parts
    :param X:
    :param seg:
    :return:
    """

    X_out = []
    nFrames = int(X.shape[0] / seg)

    for ii in xrange(seg):
        X_out.append(X[ii * nFrames:(ii + 1) * nFrames, :])

    return X_out

if __name__ == '__main__':
    from pathName import *
    from globalParams import feature_seg
    import pickle
    import sys

    def subExtractionProcess(path_audio, path_feature, ii, fn):
        fs = 44100
        framesize = 2048
        hopsize = 1024
        highFrequencyBound = 11000
        logMel96bandsEXTRACTOR = FeatureExtraction(fs, framesize, hopsize, highFrequencyBound, 96)

        # print('calculating', ii, fn, 'audio feature for', channel_string, 'in total', len(filenames_audio))
        try:
            feature = logMel96bandsEXTRACTOR.log_mel_feature(os.path.join(path_audio, fn))
            # print(feature .shape)
            pickle.dump(feature, open(os.path.join(path_feature, fn.split('.')[0] + '.pkl'), "wb"))
        except:
            print(ii, fn, 'extraction error')

    def batchFeatureExtraction(path_audio, path_feature, channel_string):
        filenames_audio = [f for f in os.listdir(path_audio) if os.path.isfile(os.path.join(path_audio, f))]
        for ii, fn in enumerate(filenames_audio):
            if ii < 3000:
                continue

            p = Process(target=subExtractionProcess, args=(path_audio, path_feature, ii, fn, ))
            p.start()
            p.join()


    def subSegmentationProcess(path_feature, filename_base, ii, fn):
        for aug in ['-1', '-2', '-3', '1', '2', '3', 'deltas3', 'deltas11', 'deltas19']:
            #try:
            feature = pickle.load(open(os.path.join(path_feature, filename_base + '_' + aug + '.pkl'), "r"))
            feature_list = featureSegmentation(feature, feature_seg)
            for ii_X, X in enumerate(feature_list):
                pickle.dump(X,
                            open(os.path.join(path_feature, filename_base + '_' + aug + '+' + str(ii_X) + '.pkl'),
                                 "wb"))
            #except:
            #    print(ii, fn, aug, 'segmenting error')


    def batchFeatureSegmentation(filenames_audio, path_feature, channel_string, augmentation=False):
        for ii, fn in enumerate(filenames_audio):
            print('segmenting', ii, fn, 'audio feature for', channel_string, 'in total', len(filenames_audio))

            filename_base = fn.split('.')[0]
            if augmentation:
                start_time = time.time()
                subSegmentationProcess(path_feature,filename_base,ii,fn)
                end_time = time.time()
                print('Elapse time', end_time-start_time)
            else:
                try:
                    feature = pickle.load(open(os.path.join(path_feature, filename_base + '.pkl'), "r"))
                    feature_list = featureSegmentation(feature, feature_seg)
                    for ii_X, X in enumerate(feature_list):
                        pickle.dump(X, open(os.path.join(path_feature, filename_base + '+' + str(ii_X) + '.pkl'), "wb"))
                except:
                    print(ii, fn, 'segmenting error')


    # fs = 44100
    # framesize = 2048
    # hopsize = 1024
    # highFrequencyBound = 11000
    # logMel96bandsEXTRACTOR = FeatureExtraction(fs,framesize,hopsize,highFrequencyBound,96)


    # batchFeatureExtraction(logMel96bandsEXTRACTOR.log_mel_feature, path_audio_left, path_feature_CNNs_logMel_96bands_left, 'left')
    # batchFeatureExtraction(logMel96bandsEXTRACTOR.log_mel_feature, path_audio_right, path_feature_CNNs_logMel_96bands_right, 'right')
    # batchFeatureExtraction(logMel96bandsEXTRACTOR.log_mel_feature, path_audio_average, path_feature_CNNs_logMel_96bands_average, 'average')
    # batchFeatureExtraction(logMel96bandsEXTRACTOR.log_mel_feature, path_audio_difference, path_feature_CNNs_logMel_96bands_difference, 'difference')

    # batchFeatureSegmentation(logMel96bandsEXTRACTOR.featureSegmentation, path_audio_left, path_feature_CNNs_logMel_96bands_left, 'left')
    # batchFeatureSegmentation(logMel96bandsEXTRACTOR.featureSegmentation, path_audio_right, path_feature_CNNs_logMel_96bands_right,'right')
    # batchFeatureSegmentation(logMel96bandsEXTRACTOR.featureSegmentation, path_audio_average, path_feature_CNNs_logMel_96bands_average, 'average')
    # batchFeatureSegmentation(logMel96bandsEXTRACTOR.featureSegmentation, path_audio_difference, path_feature_CNNs_logMel_96bands_difference, 'difference')

    index = int(sys.argv[1])
    # Augmentation
    if index == 1:
        batchFeatureExtraction(os.path.join(path_audio_augmentation, 'left'),
                               os.path.join(path_feature_CNNs_logMel_96bands, 'left'), 'left')
    elif index == 2:
        batchFeatureExtraction(os.path.join(path_audio_augmentation, 'right'),
                               os.path.join(path_feature_CNNs_logMel_96bands, 'right'), 'right')
    elif index == 3:
        batchFeatureExtraction(os.path.join(path_audio_augmentation, 'average'),
                               os.path.join(path_feature_CNNs_logMel_96bands, 'average'), 'average')
    elif index == 4:
        batchFeatureExtraction(os.path.join(path_audio_augmentation, 'difference'),
                               os.path.join(path_feature_CNNs_logMel_96bands, 'difference'), 'difference')


    # segmentation augmentation

    #batchFeatureSegmentation(path_audio_left, path_feature_CNNs_logMel_96bands_left_augmentation, 'left', True)
    #batchFeatureSegmentation(path_audio_right, path_feature_CNNs_logMel_96bands_right_augmentation,'right', True)

    # filenames_audio_average = [f for f in os.listdir(path_audio_average) if os.path.isfile(os.path.join(path_audio_average, f))]
    # ii = 0
    # len_filenames_audio = len(filenames_audio_average)
    # while ii*100 < len_filenames_audio:
    #     filenames_audio = filenames_audio_average[ii*100:(ii+1)*100]
    #     p = Process(target=batchFeatureSegmentation,
    #                 args=(filenames_audio, path_feature_CNNs_logMel_96bands_average_augmentation,'average', True))
    #     #batchFeatureSegmentation(filenames_audio, path_feature_CNNs_logMel_96bands_average_augmentation, 'average', True)
    #     p.start()
    #     p.join()
    #     ii += 1
    # #batchFeatureSegmentation(logMel96bandsEXTRACTOR.featureSegmentation, path_audio_difference, path_feature_CNNs_logMel_96bands_difference_augmentation, 'difference', True)

    # filenames_audio_difference = [f for f in os.listdir(path_audio_difference) if
    #                            os.path.isfile(os.path.join(path_audio_difference, f))]
    # ii = 0
    # len_filenames_audio = len(filenames_audio_difference)
    # while ii * 100 < len_filenames_audio:
    #     filenames_audio = filenames_audio_difference[ii * 100:(ii + 1) * 100]
    #     p = Process(target=batchFeatureSegmentation,
    #                 args=(filenames_audio, path_feature_CNNs_logMel_96bands_difference_augmentation, 'difference', True))
    #     # batchFeatureSegmentation(filenames_audio, path_feature_CNNs_logMel_96bands_average_augmentation, 'average', True)
    #     p.start()
    #     p.join()
    #     ii+=1
