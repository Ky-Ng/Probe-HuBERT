import torch
import numpy as np
from transformers import HubertForCTC, Wav2Vec2Processor
import librosa

class AudioProcessing:
    foo = "bar"
    @staticmethod
    def process_audio(
            wav_path: str,
            embedding_model: Wav2Vec2Processor,
            inference_model: HubertForCTC,
            sampling_rate: int = 16000
    ) -> tuple[torch.tensor, int]:

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

        # a single hidden_state's shape is [1, seq_len=based on length of speech input, hidden_size=1024 for our model]
        sequence_length = arbitrary_hidden_state.shape[1]

        return embedded_audio, sequence_length

    @staticmethod
    def select_samples(input_arr: np.array, num_samples: int) -> np.array:
        sampled_audio = [input_arr[random_sample] # take text array
                         for random_sample in # Take each random sample index
                         np.random.choice(len(input_arr), # Generate list of sample indexes
                                          size=num_samples, 
                                          replace=False)]
        return sampled_audio
    
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
