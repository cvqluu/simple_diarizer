import datetime
import subprocess
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from IPython.display import Audio, display


##################
# Audio utils
##################
def check_wav_16khz_mono(wavfile):
    """
    Returns True if a wav file is 16khz and single channel
    """
    try:
        signal, fs = torchaudio.load(wavfile)

        mono = signal.shape[0] == 1
        freq = fs == 16000
        if mono and freq:
            return True
        else:
            return False
    except:
        return False


def convert_wavfile(wavfile, outfile):
    """
    Converts file to 16khz single channel mono wav
    """
    cmd = "ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(
        wavfile, outfile
    )
    subprocess.Popen(cmd, shell=True).wait()
    return outfile


def check_ffmpeg():
    """
    Returns True if ffmpeg is installed
    """
    try:
        subprocess.check_output("ffmpeg", stderr=subprocess.STDOUT)
        return True
    except OSError as e:
        return False


##################
# Plotting utils
##################
colors = np.array(
    [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
)


def waveplot(signal, fs, start_idx=0, figsize=(5, 3), color="tab:blue"):
    """
    A waveform plot for a signal

    Inputs:
        - Signal (array): The waveform (1D)
        - fs (int): The frequency in Hz
        - start_idx (int): The starting index of the signal, changes the x axis
                    (note this does not slice the signal, this just changes the xticks)
        - figsize (tuple): Figure size tuple passed to plt.figure()
        - color: (str): The color of the waveform
    Outputs:
        - Returns the matplotlib figure
    """
    plt.figure(figsize=figsize)
    start_time = start_idx / fs
    end_time = start_time + (len(signal) / fs)

    plt.plot(np.linspace(start_time, end_time, len(signal)), signal, color=color)
    plt.xlabel("Time (s)")
    plt.xlim([start_time, end_time])

    max_amp = np.max(np.abs([np.max(signal), np.min(signal)]))
    plt.ylim([-max_amp, max_amp])

    plt.tight_layout()
    return plt.gcf()


def combined_waveplot(signal, fs, segments, figsize=(10, 3), tick_interval=60):
    """
    The full diarized waveform plot, with each speech segment coloured according to speaker

        Inputs:
            - Signal (array): The waveform (1D)
            - fs (int): The frequency in Hz
                        (should be 16000 for the models in this repo)
            - segments (list):  The diarization outputs (segment information)
            - figsize (tuple): Figsize passed into plt.figure()
            - tick_interval (float): Where to place ticks for xlabel

        Outputs:
            - The matplotlib figure
    """
    plt.figure(figsize=figsize)
    for seg in segments:
        start = seg["start_sample"]
        end = seg["end_sample"]
        speech = signal[start:end]
        color = colors[seg["label"]]

        linelabel = "Speaker {}".format(seg["label"])
        plt.plot(
            np.linspace(seg["start"], seg["end"], len(speech)),
            speech,
            color=color,
            label=linelabel,
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower right")

    plt.xlabel("Time")
    plt.xlim([0, len(signal) / fs])

    xticks = np.arange(0, (len(signal) // fs) + 1, tick_interval)
    xtick_labels = [str(datetime.timedelta(seconds=int(x))) for x in xticks]
    plt.xticks(ticks=xticks, labels=xtick_labels)

    max_amp = np.max(np.abs([np.max(signal), np.min(signal)]))
    plt.ylim([-max_amp, max_amp])

    plt.tight_layout()
    return plt.gcf()


def waveplot_perspeaker(signal, fs, segments):
    """
    Makes a waveplot for each speech segment (single speaker per plot).
    Also previews the audio of that clip

    Designed to be run in a jupyter notebook
    """
    for seg in segments:
        start = seg["start_sample"]
        end = seg["end_sample"]
        speech = signal[start:end]
        color = colors[seg["label"]]
        waveplot(speech, fs, start_idx=start, color=color)
        plt.show()
        print("Speaker {} ({}s - {}s)".format(seg["label"], seg["start"], seg["end"]))
        if "words" in seg:
            pprint(seg["words"])
        display(Audio(speech, rate=fs))
        print("=" * 40 + "\n")
