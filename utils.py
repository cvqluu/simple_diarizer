import datetime
import subprocess

import torchaudio
from bs4 import BeautifulSoup


##################
# Audio utils
##################
def check_wav_16khz_mono(wavfile):
    """
    Checks if a wav file is 16khz and single channel
    """
    signal, fs = torchaudio.load(wavfile)

    mono = signal.shape[0] == 1
    freq = fs == 16000
    if mono and freq:
        return True
    else:
        return False


def convert_wavfile(wavfile, outfile):
    """
    Converts file to 16khz single channel mono wav
    """
    cmd = "ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(
        wavfile, outfile)
    subprocess.Popen(cmd, shell=True).wait()


##################
# TTML utils
##################

def parse_timestamp(timestr):
    starttime = datetime.datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
    date_time = datetime.datetime.strptime(timestr, "%H:%M:%S.%f")
    delta = date_time - starttime
    return delta.total_seconds()

def parse_entry(entry):
    start = parse_timestamp(entry['begin'])
    end = parse_timestamp(entry['end'])
    text = entry.text
    return {'start': start, 'end': end, 'text': text}

def parse_ttml(file):
    with open(file) as fp:
        soup = BeautifulSoup(fp, 'lxml')
    starttime = datetime.datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
    entries = soup.findAll('p')
    segments = []
    for e in entries:
        seg = parse_entry(e)
        segments.append(seg)
    return segments
