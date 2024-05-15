import torch
import os
import numpy as np
import pandas as pd
from transformers import HubertForCTC, Wav2Vec2Processor
import librosa


class AudioProcessing:

    def process_audio(
            wav_path: str,
            embedding_model: Wav2Vec2Processor,
            inference_model: HubertForCTC,
            sampling_rate: int = 16000
    ) -> tuple[torch.tensor, int, int]:
        """
        Takes a path to a wav file and returns the speech embedding and the size of
        the sequence_length (number of 1024 vectors used to represent the speech length)

        The longer the speech recording, the greater the sequence_length
        TODO param descriptions
        """

        # Step 1) Load in .wav as time series
        speech, _ = librosa.load(
            wav_path, sr=sampling_rate
        )

        # Step 2) Get Model Embedding
        embedded_audio = embedding_model(
            speech,
            return_tensors="pt",  # pt means pytorch tensors
            sampling_rate=sampling_rate,
            output_hidden_states=True,  # Shouldn't be needed
            output_attentions=True,  # Shouldn't be needed
        ).input_values

        # Step 3) Get Sequence Length

        # Note: len(inference_model.hidden_states) = # Encoders
        hidden_states = inference_model(embedded_audio).hidden_states

        # We arbitrarily choose the last encoder's hidden_state
        arbitrary_hidden_state = hidden_states[-1]

        # a single hidden_state's shape is [1, seq_len, hidden_size]
        # Note: in our use case, seq_len=based on length of speech input, hidden_size=1024 for our model
        sequence_length = arbitrary_hidden_state.shape[1]
        num_speech_frames = len(speech)

        return embedded_audio, num_speech_frames, sequence_length

    def select_samples(input_arr: np.array, num_samples: int) -> np.array:
        sampled_audio = [input_arr[random_sample]  # take text array
                         for random_sample in  # Take each random sample index
                         np.random.choice(len(input_arr),  # Generate list of sample indexes
                                          size=num_samples,
                                          replace=False)]
        return sampled_audio

    def get_sequence_boundary(
            TIMIT_wav_path: str,
            num_speech_frames: int,
            num_speech_vec: int
    ) -> pd.DataFrame:
        """
        Takes a `TIMIT` .wav file and finds the corresponding .PHN file
        to translate phoneme `beginning` and `end` boundaries from
        speech frames to corresponding index of speech vector in input sequence

        ex: 
        - There is a 10 input speech vectors (each size 1024)
        - Total duration is 100 speech frames (proportional to time(ms))
        - the "i" phoneme is `start` =  50th speech frame and `end` = 55th speech frame
        - the "i" phoneme is close to the 5th speech vector (of size 1024)

        # Returns
        boundaries : `np.ndarray`
            3 dimensional `np.ndarray` = [phoneme string, start boundary, end bondary]
        """

        # Step 1) Find the .PHN file
        path_without_extension = os.path.splitext(TIMIT_wav_path)[0]
        phon_path = f"{path_without_extension}.PHN"

        # Step 2) Calculate speech vector index from speech frame boundaries (use dimensional analysis)
        vecs_per_speech_frame = float(num_speech_vec) / num_speech_frames

        # Step 3) Read in the boundaries and scale speech frames to vector index
        speech_frame_df = pd.read_csv(phon_path, sep=' ', header=None)

        # 3a) Scale the start/end boundaries
        scaled_vec_boundaries_df = (
            speech_frame_df.iloc[:, :2] * vecs_per_speech_frame).round().astype(int)

        # 3b) Keep the segmented gloss
        phon_code_df = speech_frame_df.iloc[:, 2]

        # 3c) recombine the files
        combined_df = pd.concat(
            [scaled_vec_boundaries_df, phon_code_df], axis=1
        )

        return combined_df

    def filter_segmentation(
        combined_df: pd.DataFrame,
            desired_phonemes: set
    ) -> pd.DataFrame:
        """
        Returns only the DataFrame rows whose 3rd column (`phon` transcription) is in `desired_phonemes`
        """
        phoneme_mask = combined_df.iloc[:, 2].isin(desired_phonemes)
        filtered_seg = combined_df[phoneme_mask]
        return filtered_seg

    def get_hidden_states(
            input_embedding: torch.tensor,
            inference_model: HubertForCTC,
            start_idx: int = None,
            end_idx: int = None
    ) -> np.ndarray:
        """
        Returns the hidden state for each encoder (1 to 25 for most cases)
        and each variable length number of 1024 speech vectors.

        Optionally slices out only the indexes for the specific phoneme specified by
        [start_idx, end_idx)
        Note: Both `start_idx` and `end_idx` must be specified or else all sequences of speech vecotrs will be returned
        """

        # Step 1) Get all hidden states for encoders as a list of 2D np.ndarray (seq_len, hidden_size)
        hidden_states_list = [AudioProcessing.tensor_to_np(
            encoder_state) for encoder_state in inference_model(input_embedding).hidden_states]
        
        # Step 2) Convert from a list of 2D np.ndarrys to a 3D np.ndarray
        hidden_states_combined = np.stack(hidden_states_list, axis=0)

        # Step 3) Slice out the start 
        if(start_idx is not None and end_idx is not None):
            hidden_states_combined = hidden_states_combined[:, start_idx:end_idx, :]
        print(hidden_states_combined.shape)
        return hidden_states_combined

    def tensor_to_np(tensor_to_convert: torch.tensor) -> np.ndarray:
        # Step 1) Convert Pytorch Tensor to Numpy
        np_version = tensor_to_convert.detach().numpy()

        # Step 2) Remove the first dimension of the  (1, seq_len, hidden_size)
        remove_single_dim = np.squeeze(np_version, axis=0)

        return remove_single_dim

# Test Driver:
# embedded_audio, seq_length = AudioProcessing.process_audio(
#     wav_path="/Users/kyleng/B_Organized/A_School/Ling_487/clean_code/Probe-HuBERT/TIMIT-Database/TIMIT/TEST/DR1/FAKS0/SA1.wav",
#     embedding_model=Wav2Vec2Processor.from_pretrained(
#         "facebook/wav2vec2-base-960h"),
#     inference_model=HubertForCTC.from_pretrained(
#         "facebook/hubert-large-ls960-ft", output_attentions=True, output_hidden_states=True),
#     sampling_rate=16000
# )
# print(embedded_audio.shape)

# print(seq_length)
