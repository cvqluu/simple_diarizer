# simple_diarizer


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nMKHOTTROwQitOXQEYq35lvv7nyTOlpe?usp=sharing)

Simplified diarization pipeline to oracle number speakers using some pretrained models.

```python
from simple_diarizer.diarizer import Diarizer

diar = Diarizer(embed_model='xvec', window=1.5, period=0.75)
segments = diar.diarize(WAV_FILE, num_speakers=NUM_SPEAKERS)

combined_waveplot(signal, fs, segments)
plt.show()
```

# Pre-trained Models

The following pretrained models are used:

 - Voice Activity Detection (VAD)
     - [Silero VAD](https://github.com/snakers4/silero-vad)
 - Deep speaker embedding extraction
     - [SpeechBrain](https://github.com/speechbrain/speechbrain)
        - [X-Vector](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)
        - [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
 - (Optional/Experimental) Speech-to-text
     - [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)
        - English ASR model

Hopefully this can be of use as a free basic tool for people transcribing interviews.

It can be checked out here, where it will try and diarize any input YouTube URL

# **[Google Colab](https://colab.research.google.com/drive/1nMKHOTTROwQitOXQEYq35lvv7nyTOlpe?usp=sharing)**
