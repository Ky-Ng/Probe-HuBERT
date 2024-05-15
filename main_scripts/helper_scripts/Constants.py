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