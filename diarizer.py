import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


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
    cmd = "ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(wavfile, outfile)
    subprocess.Popen(cmd, shell=True).wait()

class Diarizer:

    def __init__(self, embed_model,
                window=1.5, period=0.75):
        
        self.vad_model, self.get_speech_ts = self.setup_VAD()
        self.embed_model = embed_model

        self.window = window
        self.period = period

    def setup_VAD(self):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad')
                                    # force_reload=True)

        (get_speech_ts,
            _, _, read_audio,
            _, _, _) = utils
        return model, get_speech_ts

    def vad(self, signal):
        """
        Runs the VAD model on the signal
        """
        return self.get_speech_ts(signal, self.vad_model)

    def windowed_embeds(self, signal, fs, window=1.5, period=0.75):
        """
        Calculates embeddings for windows across the signal

        window: length of the window, in seconds
        period: jump of the window, in seconds

        returns: embeddings, segment info
        """
        len_window = int(window * fs)
        len_period = int(period * fs)
        len_signal = signal.shape[1]

        # Get the windowed segments
        segments = []
        start = 0
        while start + len_window < len_signal:
            segments.append([start, start+len_window])
            start += len_period
        
        segments.append([start, len_signal-1])
        embeds = []

        with torch.no_grad():
            for i, j in segments:
                signal_seg = signal[:, i:j]
                seg_embed = self.embed_model.encode_batch(signal_seg)
                embeds.append(seg_embed.squeeze(0).squeeze(0).cpu().numpy())

        embeds = np.array(embeds)
        return embeds, np.array(segments)

    def recording_embeds(self, signal, fs, speech_ts):
        """
        Takes signal and VAD output (speech_ts) and produces windowed embeddings

        returns: embeddings, segment info
        """
        all_embeds = []
        all_segments = []

        for utt in tqdm(speech_ts, desc='Utterances', position=0):
            start = utt['start']
            end = utt['end']

            utt_signal = signal[:,start:end]
            utt_embeds, utt_segments = self.windowed_embeds(utt_signal, 
                                                            fs,
                                                            self.window,
                                                            self.period)
            all_embeds.append(utt_embeds)
            all_segments.append(utt_segments + start)
        
        all_embeds = np.concatenate(all_embeds, axis=0)
        all_segments = np.concatenate(all_segments, axis=0)
        return all_embeds, all_segments

    @staticmethod
    def cluster(embeds, num_speakers=2):
        """
        Uses AHC to cluster to set number of speakers
        """
        cluster_model = AgglomerativeClustering(n_clusters=num_speakers,
                                                affinity='cosine',
                                                linkage='average')
        cluster_labels = cluster_model.fit_predict(embeds)
        return cluster_labels

    @staticmethod
    def join_segments(cluster_labels, segments):
        """
        Joins up same speaker segments, resolves overlap conflicts

        Uses the midpoint for overlap conflicts
        """
        assert len(cluster_labels) == len(segments)

        new_segments = [{'start': segments[0][0], 
                         'end': segments[0][1], 
                         'label': cluster_labels[0]}]

        for l, seg in zip(cluster_labels[1:], segments[1:]):
            start = seg[0]
            end = seg[1]

            protoseg = {'start': seg[0],
                        'end': seg[1],
                        'label': l}

            if start <= new_segments[-1]['end']:
                # If segments overlap
                if l == new_segments[-1]['label']:
                    # If overlapping segment has same label
                    new_segments[-1]['end'] = end
                else:
                    # If overlapping segment has diff label
                    # Resolve by setting new start to midpoint
                    # And setting last segment end to midpoint
                    overlap = new_segments[-1]['end'] - start
                    midpoint = start + overlap//2
                    new_segments[-1]['end'] = midpoint
                    protoseg['start'] = midpoint
                    new_segments.append(protoseg)
            else:
                # If there's no overlap just append
                new_segments.append(protoseg)

        return new_segments

    @staticmethod
    def make_output_seconds(cleaned_segments, fs):
        """
        Convert cleaned segments to readable format in seconds
        """
        for seg in cleaned_segments:
            seg['start'] = seg['start']/fs
            seg['end'] = seg['end']/fs
        return cleaned_segments

    def diarize(self, 
                wav_file, 
                num_speakers=2,
                extra_info=False,
                outfile=None):
        """
        Diarize a 16khz mono wav file according to set num_speakers,

        Uses AHC to cluster
        """
        signal, fs = torchaudio.load(wav_file)
        recname = os.path.splitext(os.path.basename(wav_file))[0]
        
        assert signal.shape[0] == 1, "Audio needs to be single channel"
        assert fs == 16000, "Only 16khz audio supported"

        print('Running VAD...')
        speech_ts = self.vad(signal[0])
        print('Splitting by silence found {} utterances'.format(len(speech_ts)))
        assert len(speech_ts) >= 1, "Couldn't find any speech during VAD"

        print('Extracting embeddings...')
        embeds, segments = self.recording_embeds(signal, fs, speech_ts)

        print('Clustering to {} speakers...'.format(num_speakers))
        cluster_labels = self.cluster(embeds, num_speakers=num_speakers)

        print('Cleaning up output...')
        cleaned_segments = self.join_segments(cluster_labels, segments)
        cleaned_segments = self.make_output_seconds(cleaned_segments, fs)
        
        if outfile:
            self.rttm_output(cleaned_segments, recname, outfile=outfile)

        if not extra_info:
            return cleaned_segments
        else:
            return cleaned_segments, embeds, cleaned_segments

    @staticmethod
    def rttm_output(segments, recname, outfile=None):
        assert outfile, "Please specify an outfile"
        rttm_line = "SPEAKER {} 0 {} {} <NA> <NA> {} <NA> <NA>\n"
        with open(outfile, 'w') as fp:
            for seg in segments:
                start = seg['start']
                offset = seg['end'] - seg['start']
                label = seg['label']
                line = rttm_line.format(recname, start, offset, label)
                fp.write(line)
            
    @staticmethod
    def join_samespeaker_segments(segments, silence_tolerance=5.0):
        """
        Join up segments that belong to the same speaker, 
        even if there is a duration of silence in between them.

        If the silence is greater than silence_tolerance, does not join up
        """
        new_segments = [segments[0]]

        for seg in segments[1:]:
            if seg['label'] == new_segments[-1]['label']:
                if new_segments[-1]['end'] + silence_tolerance >= seg['start']:
                    new_segments[-1]['end'] = seg['end']
                else:
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
        return new_segments

    def match_diarization_to_transcript(self, segments, ctm_file):
        """
        Match the output of .diarize to a ctm file produced by asr
        """
        ctm_df = pd.read_csv(ctm_file, delimiter=' ', 
                                names=['utt', 'channel', 'start', 'offset', 'word', 'confidence'])
        ctm_df['end'] = ctm_df['start'] + ctm_df['offset']

        starts = ctm_df['start'].values
        ends = ctm_df['end'].values
        words = ctm_df['word'].values

        # Get the earliest start from either diar output or asr output
        earliest_start = np.min([ctm_df['start'].values[0], segments[0]['start']])

        worded_segments = self.join_samespeaker_segments(segments)
        worded_segments[0]['start'] = earliest_start
        cutoffs = []

        for seg in worded_segments:
            end_idx = np.searchsorted(ctm_df['end'].values, seg['end'], side='left') - 1
            cutoffs.append(end_idx)
        
        indexes = [[0, cutoffs[0]]]
        for c in cutoffs[1:]:
            indexes.append([indexes[-1][-1], c])
        
        indexes[-1][-1] = len(words)

        final_segments = []

        for i, seg in enumerate(worded_segments):
            s_idx, e_idx = indexes[i]
            words = ctm_df['word'].values[s_idx:e_idx]
            seg['words'] = ' '.join(words)
            if len(words) >= 1:
                final_segments.append(seg)
            else:
                print('Removed segment between {} and {} as no words were matched'.format(
                                                seg['start'], seg['end']))
        
        return final_segments

    @staticmethod
    def nice_text_output(worded_segments, outfile):
        with open(outfile, 'w') as fp:
            for seg in worded_segments:
                fp.write('[{} to {}] Speaker {}: \n'.format(round(seg['start'], 2),
                                                            round(seg['end'], 2),
                                                            seg['label']))
                fp.write('{}\n\n'.format(seg['words']))


if __name__ == "__main__":
    wavfile = sys.argv[1]
    num_speakers = int(sys.argv[2])
    outfolder = sys.argv[3]

    assert os.path.isfile(wavfile), "Couldn't find {}".format(wavfile)

    recname = os.path.splitext(os.path.basename(wavfile))[0]
    os.makedirs(outfolder, exist_ok=True)

    if check_wav_16khz_mono(wavfile):
        correct_wav = wavfile
    else: 
        correct_wav = os.path.join(outfolder, '{}_converted.wav'.format(recname))
        convert_wavfile(wavfile, correct_wav)

    # model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

    diar = Diarizer(model)
    segments = diar.diarize(correct_wav, num_speakers=num_speakers, outfile=os.path.join(outfolder, 'hyp.rttm'))

