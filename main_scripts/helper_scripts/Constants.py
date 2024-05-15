from transformers import HubertForCTC, Wav2Vec2Processor


class Constants:
    class LLM:
        # 24 Encoders + 1 input layer as an "encoder"
        NUM_ENCODERS = 25

        # Each input vectors size
        EMBEDDING_SIZE = 1024

    class AUDIO:

        # TIMIT is sampled at 16 kHz
        SAMPLE_RATE = 16000

    class EXPERIMENTATION:
        NUM_SPEECH_SAMPLES = 200

        EMBEDDING_MODEL = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h")
        
        INFERENCE_MODEL = model = HubertForCTC.from_pretrained(
            "facebook/hubert-large-ls960-ft", output_attentions=True, output_hidden_states=True)
    
    class PATHING:
        hidden_state_save_path = "../data/numpy"
        
