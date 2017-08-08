import essentia.standard as es
from scipy.io import wavfile
import os
from multiprocessing import Process
from pathName import *

def channelSep(filename_wav, path_dcase):
    """
    Separate stereo audio into left, right, average and difference
    :param filename_wav:
    :return:
    """
    LOADER = es.AudioLoader(filename=filename_wav)
    audio, sr, num_chan, md5, bitrate, codec = LOADER()
    filename_wav = os.path.basename(filename_wav)
    print(filename_wav)
    wavfile.write(os.path.join(path_dcase, 'audio_left', filename_wav), sr, audio[:, 0])
    wavfile.write(os.path.join(path_dcase, 'audio_right', filename_wav), sr, audio[:, 1])
    wavfile.write(os.path.join(path_dcase, 'audio_average', filename_wav), sr, audio[:, 0]/2.0+audio[:, 1]/2.0)
    wavfile.write(os.path.join(path_dcase, 'audio_difference', filename_wav), sr, audio[:, 0]-audio[:, 1])

if __name__ == "__main__":
    wavfiles = [f for f in os.listdir(path_audio_eval) if os.path.isfile(os.path.join(path_audio_eval, f))]
    for f in wavfiles:
        print('separating file',f,'in total',len(wavfiles))

        p = Process(target=channelSep, args=(os.path.join(path_audio_eval, os.path.basename(f)), path_dcase2017_evaluation,))
        p.start()
        p.join()