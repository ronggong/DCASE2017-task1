import essentia.standard as es
import pandas as pd
from pathName import *
import numpy
import os
from multiprocessing import Process


def convert_pool_to_dataframe(essentia_pool, filename):
    """
    convert essentia pool to pandas DataFrame
    :param essentia_pool:
    :param filename:
    :return:
    """
    pool_dict = dict()
    for desc in essentia_pool.descriptorNames():
        if 'histogram' in desc:
            # print('ignore',desc)
            continue
        if desc == 'lowlevel.gfcc.cov' or desc == 'lowlevel.gfcc.icov' \
            or desc == 'lowlevel.mfcc.cov' or desc == 'lowlevel.mfcc.icov':
            # print('ignore', desc)
            continue
        if desc.split('.')[1] == 'melbands128':
            # print('ignore', desc)
            continue

        if type(essentia_pool[desc]) is float:
            pool_dict[desc] = essentia_pool[desc]
        elif type(essentia_pool[desc]) is numpy.ndarray:
            # we have to treat multivariate descriptors differently
            for i, value in enumerate(essentia_pool[desc]):
                feature_name = "{desc_name}{desc_number}.{desc_stat}".format(
                    desc_name=desc.split('.')[0],
                    desc_number=i,
                    desc_stat=desc.split('.')[1])
                pool_dict[feature_name] = value
    return pd.DataFrame(pool_dict, index=[filename])

def statsticsCal(array, dict_seg, desc):
    m = numpy.mean(array)
    v = numpy.var(array)
    d = numpy.diff(array)
    dm = numpy.mean(d)
    dv = numpy.var(d)
    dict_seg[desc + '.mean'] = m
    dict_seg[desc + '.var'] = v
    dict_seg[desc + '.dmean'] = dm
    dict_seg[desc + '.dvar'] = dv
    return  dict_seg

def frame_pool_aggregation(essentia_frame_pool, filename):
    ii = 0
    feature_frame = pd.DataFrame()
    seg_framesize = 22
    while ii < 217/seg_framesize+1:
        dict_seg = {}
        # ignore all the useless features
        for desc in essentia_frame_pool.descriptorNames():
            if 'histogram' in desc:
                # print('ignore',desc)
                continue
            if desc == 'lowlevel.gfcc.cov' or desc == 'lowlevel.gfcc.icov' \
                or desc == 'lowlevel.mfcc.cov' or desc == 'lowlevel.mfcc.icov':
                # print('ignore', desc)
                continue
            if 'melbands128' in desc:
                # print('ignore', desc)
                continue
            if 'onset_times' in desc or \
                'bpm_intervals' in desc or \
                'metadata' in desc or \
                'beats_position' in desc or \
                    'chords_key' in desc or \
                    'chords_scale' in desc or \
                    'key_edma' in desc or \
                    'key_krumhansl' in desc or \
                    'key_temperley' in desc or \
                    'chords_progression' in desc or \
                    'rhythm' in desc or \
                    'tonal.tuning_frequency' in desc or \
                    'sfx.oddtoevenharmonicenergyratio' in desc or \
                    'tristimulus' in desc or \
                    'loudness_ebu128' in desc:
                continue

            if type(essentia_frame_pool[desc]) is float:
                continue
            if essentia_frame_pool[desc].shape[0] == 1:
                continue

            if len(essentia_frame_pool[desc].shape) == 1:
                dict_seg = statsticsCal(essentia_frame_pool[desc][seg_framesize*ii:seg_framesize*(ii+1)], dict_seg, desc)

            else:
                for jj in range(essentia_frame_pool[desc].shape[1]):
                    dict_seg = statsticsCal(essentia_frame_pool[desc][seg_framesize*ii:seg_framesize*(ii+1),jj], dict_seg, desc+str(jj))

        dataFrame_ii = pd.DataFrame(dict_seg, index=[filename + '_' + str(ii)])
        feature_frame = feature_frame.append(dataFrame_ii)
        ii += 1

    return feature_frame


def batchFeatureExtraction(extractor, path_audio, path_feature_freesound_statistics, channel_string):
    filenames_audio = [f for f in os.listdir(path_audio) if os.path.isfile(os.path.join(path_audio, f))]
    for ii, fn in enumerate(filenames_audio):
        print('calculating', ii, fn, 'audio feature for', channel_string, 'in total', len(filenames_audio))
        try:
            feature_pool, _ = extractor(os.path.join(path_audio, fn))
            feature_pd_DataFrame = convert_pool_to_dataframe(feature_pool, fn)
            feature_pd_DataFrame.to_csv(os.path.join(path_feature_freesound_statistics, fn.split('.')[0]+'.csv'))
        except:
            print(ii, fn, 'extraction error')


def batchFeatureExtractionFrame(path_audio, path_feature_freesound_statistics, channel_string):
    filenames_audio = [f for f in os.listdir(path_audio) if os.path.isfile(os.path.join(path_audio, f))]
    for ii, fn in enumerate(filenames_audio):
        if ii < 21840:
            continue
        print('calculating', ii, fn, 'audio feature for', channel_string, 'in total', len(filenames_audio))
        # try:
        extractor = es.FreesoundExtractor(lowlevelFrameSize=4096, lowlevelHopSize=2048)
        _, feature_frame_pool = extractor(os.path.join(path_audio, fn))
        feature_pd_DataFrame = frame_pool_aggregation(feature_frame_pool, fn)
        feature_pd_DataFrame.to_csv(os.path.join(path_feature_freesound_statistics, fn.split('.')[0]+'.csv'))
        # except:
        #     print(ii, fn, 'extraction error')

def subprocessFeatureExtractionFrame(path_audio, path_feature_freesound_statistics, fn):
    """
    Run this in process to avoid memory leak
    :param path_audio:
    :param path_feature_freesound_statistics:
    :param fn:
    :return:
    """
    extractor = es.FreesoundExtractor(lowlevelFrameSize=4096, lowlevelHopSize=2048)
    _, feature_frame_pool = extractor(os.path.join(path_audio, fn))
    feature_pd_DataFrame = frame_pool_aggregation(feature_frame_pool, fn)
    feature_pd_DataFrame.to_csv(os.path.join(path_feature_freesound_statistics, fn.split('.')[0] + '.csv'))

if __name__ == '__main__':
    # sample = '/media/gong/Rong Seagat/dcase/task1/TUT-acoustic-scenes-2017-development/audio/a001_0_10.wav'

    # EXTRACTOR = es.FreesoundExtractor()
    # EXTRACTOR_frame = es.FreesoundExtractor(lowlevelFrameSize=4096, lowlevelHopSize=2048)

    # batchFeatureExtractionFrame(EXTRACTOR_frame, path_audio_left, path_feature_freesound_statistics_seg_left, 'left')
    # batchFeatureExtractionFrame(EXTRACTOR_frame, path_audio_right, path_feature_freesound_statistics_seg_right, 'right')
    # batchFeatureExtractionFrame(EXTRACTOR_frame, path_audio_average, path_feature_freesound_statistics_seg_average, 'average')
    # batchFeatureExtractionFrame(EXTRACTOR_frame, path_audio_difference, path_feature_freesound_statistics_seg_difference, 'difference')

    for channel_string in ['difference']: #['left', 'right', 'average', 'difference']:
        path_audio_channel_string = os.path.join(path_dcase2017_origin, 'audio_'+channel_string)
        path_feature_freesound_statistics_channel_string = os.path.join(path_dcase2017_origin,
                                                                        'feature',
                                                                        'freesound_extractor',
                                                                        'statistics_seg_'+channel_string)
        filenames_audio = [f for f in os.listdir(path_audio_channel_string) if os.path.isfile(os.path.join(path_audio_channel_string, f))]
        for ii, fn in enumerate(filenames_audio):
            # if ii < 21840:
            #     continue
            print('calculating', ii, fn, 'audio feature for', channel_string, 'in total', len(filenames_audio))
            p = Process(target=subprocessFeatureExtractionFrame, args=(path_audio_channel_string, path_feature_freesound_statistics_channel_string,fn,))
            p.start()
            p.join()

    # batchFeatureExtractionFrame(os.path.join(path_audio, 'left'),
    #                             path_feature_freesound_statistics_seg_left, 'left')

    # batchFeatureExtractionFrame(EXTRACTOR_frame,
    #                             os.path.join(path_audio, 'right'),
    #                             path_feature_freesound_statistics_seg_left, 'right')


    # batchFeatureExtractionFrame(EXTRACTOR_frame,
    #                             os.path.join(path_audio, 'average'),
    #                             path_feature_freesound_statistics_seg_left, 'average')
    #
    #
    # batchFeatureExtractionFrame(EXTRACTOR_frame,
    #                             os.path.join(path_audio, 'difference'),
    #                             path_feature_freesound_statistics_seg_left, 'difference')