"""
Tested with Python versions:
    - 3.7.12
    - 3.8.12
    - 3.9.9
    - 3.10.1 - does not work
"""


from simple_diarizer.utils import (convert_wavfile, download_youtube_wav)

from simple_diarizer.diarizer import Diarizer
import tempfile

YOUTUBE_ID = "HyKmkLEtQbs"

with tempfile.TemporaryDirectory() as outdir:
    yt_file = download_youtube_wav(YOUTUBE_ID, outdir)

    wav_file = convert_wavfile(yt_file, f"{outdir}/{YOUTUBE_ID}_converted.wav")

    print(f"wav file: {wav_file}")

    diar = Diarizer(
        embed_model='ecapa', # supported types: ['xvec', 'ecapa']
        cluster_method='sc', # supported types: ['ahc', 'sc']
        window=1.5, # size of window to extract embeddings (in seconds)
        period=0.75 # hop of window (in seconds)
    )

    NUM_SPEAKERS = 2

    segments = diar.diarize(wav_file, 
                            num_speakers=NUM_SPEAKERS,
                            outfile=f"{outdir}/{YOUTUBE_ID}.rttm")

    print(segments)     