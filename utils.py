import datetime
import subprocess

import matplotlib.pyplot as plt
import numpy as np
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

##################
# Plotting utils
##################

def waveplot(signal, fs, start_idx=0, figsize=(5,3), color='tab:blue'):
    plt.figure(figsize=figsize)
    start_time = start_idx / fs
    end_time = start_time + (len(signal) / fs)

    plt.plot(np.linspace(start_time, end_time, len(signal)), signal, color=color)
    plt.xlabel('Time (s)')
    plt.xlim([start_time, end_time])

    max_amp = np.max(np.abs([np.max(signal),np.min(signal)]))
    plt.ylim([-max_amp, max_amp])

    plt.tight_layout()
    return plt.gca()

def combined_waveplot(signal, fs, segments, figsize=(20,3)):
    colors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:yellow', 'tab:red', 'tab:purple', 'tab:brown'])
    plt.figure(figsize=figsize)
    for seg in segments:
        start = seg['start_sample']
        end = seg['end_sample']
        speech = signal[start:end]
        color = colors[seg['label']]

        linelabel = 'Speaker {}'.format(seg['label'])
        plt.plot(np.linspace(seg['start'], seg['end'], len(speech)), speech, color=color, label=linelabel)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')

    plt.xlabel('Time (s)')
    plt.xlim([0,len(signal)/fs])

    xticks = np.arange(0, (len(signal)//fs)+1, 30)
    xtick_labels = [str(datetime.timedelta(seconds=int(x))) for x in xticks]
    plt.xticks(ticks=xticks, labels=xtick_labels)

    max_amp = np.max(np.abs([np.max(signal),np.min(signal)]))
    plt.ylim([-max_amp, max_amp])

    plt.tight_layout()
    return plt.gca()
