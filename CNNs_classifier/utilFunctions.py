# -*- coding: utf-8 -*-
import sys
import os

def evalSetupTxtReader(filename_txt, tosplit=True):
    """
    Read the .txt evaluation setup file
    :param filename_txt:
    :return:
    """
    list_filename_wav = []
    list_label = []
    with open(filename_txt) as f:
        if tosplit:
            for ii, line in enumerate(f):
                filename_wav, label = line.split()
                list_filename_wav.append(filename_wav)
                list_label.append(label)
        else:
            for line in f:
                list_filename_wav.append(line)
    return list_filename_wav, list_label

# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def filenameExtRemover(fn):
    """
    Remove the extention of a filename
    :param fn:
    :return:
    """
    filename_noext = os.path.basename(fn)
    filename_noext = filename_noext.split('.')[0]
    return filename_noext