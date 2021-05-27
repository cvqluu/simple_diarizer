import subprocess

import torchaudio


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
