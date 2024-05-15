# %% [markdown]
# # Extracting HuBERT Hidden Representations

# %%
# Packages
from transformers import HubertForCTC, Wav2Vec2Processor
import numpy as np
import glob
import os
from collections import defaultdict 

# Custom Helper Libraries
from helper_scripts.TenseLax import TenseLax
from helper_scripts.AudioProcessing import AudioProcessing
from helper_scripts.Constants import *
from helper_scripts.Pathing import Pathing
for dialect_idx in range(2):
# %% [markdown]
# # Roadmap
# ```
# let n = number of samples of audio files
# let s = number of segments/phonemes to be segmented
# let k = number of encoders in LLM
# let l[i] = number of speech vectors per input sequence
# ```
# 
# 1) Select n=200 samples of audio files from a specific subset of dialects from `TIMIT`
# 2) Create the output `hidden_states`
#     - s entries
#     - each entry is a size k=25 array 
#     - each array index is a numpy array representing `l[i]` speech vectors--where each speech vector is of size 1024
# 3) Load in each of the n samples using `librosa`
# 4) 

# %% [markdown]
# # 1) Import Audio Files
# - Extract 200 path's to audio samples to save computation

# %%
# Path is TIMIT/<TEST or TRAIN>/<DIALECT>/<SPEAKER ID>/<SEGMENT ID>.wav
    DATASET_PATH = "../Timit-Database/TIMIT/"
    ALL_WAVS_PATH = os.path.join(DATASET_PATH, "*", "*", "*", "*.wav")

    speech_paths = glob.glob(ALL_WAVS_PATH)
    print(f"Importing {len(speech_paths)} speech samples")

# No need to sample since we'll do this in the post processing step
# speech_paths = AudioProcessing.select_samples(
#     speech_paths,
#     num_samples=Constants.EXPERIMENTATION.NUM_SPEECH_SAMPLES
# )
# print(f"Succesfully randomly sampled {len(speech_paths)} speech samples")

# %% [markdown]
# # 2) Create the Data Structure for Saving the Hidden States
# 
# ```
# For Each Phoneme
#     For Each Encoder 
#         For Each Sequence of 1024 Vectors
#             Append Hidden State
# ```
# 
# Hidden States maps <`phoneme string`, a list of`Hidden State Representation`>

# %%
all_hidden_states = defaultdict(list)

# %% [markdown]
# # 2) Calculate Boundaries for each Audio File

# %%
for path in speech_paths:
    # Step 1) Generate the hidden states and boundaries
    embedded_audio, num_speech_frames, sequence_length = AudioProcessing.process_audio(
        wav_path=path,
        embedding_model=Constants.EXPERIMENTATION.EMBEDDING_MODEL,
        inference_model=Constants.EXPERIMENTATION.INFERENCE_MODEL,
        sampling_rate=16000
    )

    scaled_segmentation = AudioProcessing.get_sequence_boundary(
        TIMIT_wav_path=path,
        num_speech_frames=num_speech_frames,
        num_speech_vec=sequence_length
    )

    # Step 2) Select boundaries for matching phonemes
    filtered_segmentation = AudioProcessing.filter_segmentation(
        combined_df=scaled_segmentation,
        desired_phonemes=TenseLax.getSet()
    )

    # Step 3) Place Hidden State into output matrix
    for row in filtered_segmentation.itertuples():
        _, seq_start_vec_idx, seq_end_vec_idx, phoneme = row
        
        # Step 3a) Get the Hidden States per encoder for the entire speech segment
        utterance_hidden_states = AudioProcessing.get_hidden_states(
            input_embedding=embedded_audio,
            inference_model=Constants.EXPERIMENTATION.INFERENCE_MODEL,
            start_idx=seq_start_vec_idx,
            end_idx=seq_end_vec_idx 
        )

        # Step 3b) Append hidden States to the existing hidden states for this row
        all_hidden_states[phoneme].append(
            utterance_hidden_states
        )
    

# %% [markdown]
# # 3) Save Phoneme Hidden States

# %%
for phoneme, hidden_state in all_hidden_states.items():
    combined_per_segment = np.concatenate(hidden_state, axis=1)
    print(f"{phoneme}: {combined_per_segment.shape}")
    Pathing.save_file_np(
        save_dir=Constants.PATHING.hidden_state_save_path,
        save_file_name=f"HS_{phoneme}_{combined_per_segment.shape[1]}.npy",
        to_save=combined_per_segment
    )


