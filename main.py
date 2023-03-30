import soundfile as sf
import matplotlib.pyplot as plt
import os,sys,time


from simple_diarizer.diarizer import Diarizer
from simple_diarizer.utils import combined_waveplot

t0 = time.time() 


diar = Diarizer(
                  embed_model='ecapa', # 'xvec' and 'ecapa' supported
                  cluster_method='NME-sc' # 'ahc' 'sc' and 'NME-sc' supported
               )

WAV_FILE,NUM_SPEAKERS,max_spk= sys.argv[1:]


if NUM_SPEAKERS == 'None':
   print('None')  
   segments = diar.diarize(WAV_FILE, num_speakers=None,max_speakers=int(max_spk))
else:
   segments = diar.diarize(WAV_FILE, num_speakers=int(NUM_SPEAKERS))


t1 = time.time()
feature_t = t1 - t0
print("Time used for extracting features:", feature_t)



json = {}
_segments = []
_speakers = {}
seg_id = 1
spk_i = 1
spk_i_dict = {}
        
for seg in segments:
        
   segment = {}
   segment["seg_id"] = seg_id

            # Ensure speaker id continuity and numbers speaker by order of appearance.
   if seg['label'] not in spk_i_dict.keys():
      spk_i_dict[seg['label']] = spk_i
      spk_i += 1

   spk_id = "spk" + str(spk_i_dict[seg['label']])
   segment["spk_id"] = spk_id
   segment["seg_begin"] = round(seg['start'])
   segment["seg_end"] = round(seg['end'])

   if spk_id not in _speakers:
      _speakers[spk_id] = {}
      _speakers[spk_id]["spk_id"] = spk_id
      _speakers[spk_id]["duration"] = seg['end']-seg['start']
      _speakers[spk_id]["nbr_seg"] = 1
   else:
      _speakers[spk_id]["duration"] += seg['end']-seg['start']
      _speakers[spk_id]["nbr_seg"] += 1

   _segments.append(segment)
   seg_id += 1

for spkstat in _speakers.values():
    spkstat["duration"] = round(spkstat["duration"])

json["speakers"] = list(_speakers.values())
json["segments"] = _segments


print(json["speakers"] )

