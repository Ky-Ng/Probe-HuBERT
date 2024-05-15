import librosa
from transformers import HubertForCTC, Wav2Vec2Processor
import numpy as np
import glob
import matplotlib.pyplot as plt

from helper_scripts.Constants import *
from helper_scripts.TenseLax import TenseLax


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft",output_attentions=True,output_hidden_states=True)

Segs = TenseLax.getList()

for dialect in range(1, Constants.TIMIT.NUM_DIALECTS+1):
    lst=sorted(glob.glob('../TIMIT-Database/TIMIT/TEST/DR' + str(dialect) + '/*/*.WAV'))

    print(lst)
    hs = len(Segs) * [[]]
    for s in range(len(hs)):
        hs[s]= 25*[np.empty((1024,0), float)]
    
    for n in range(len(lst)):
        speech, rate = librosa.load(lst[n], sr=16000)
        input_values = processor(speech, return_tensors="pt", padding="longest", sampling_rate=rate,output_hidden_states=True,output_attentions=True).input_values
        HSlen = model(input_values).hidden_states[24].shape[1]
        nm = lst[n][:-4] + ".PHN"
        bg,ed,phn = [],[],[]
        f = open(nm, "r")
        for l in f:
            ln = l.split()
            bg.append(ln[0])
            ed.append(ln[1])
            phn.append(ln[2])
        BG = np.rint(np.array(bg,dtype=np.double)/float(len(speech))*HSlen)
        ED = np.rint(np.array(ed,dtype=np.double)/float(len(speech))*HSlen)
        for E in range(25):
            HS =  model(input_values).hidden_states[E][0,:,:].detach().numpy()
            for p in range(len(phn)): 
                if phn[p] in set(Segs):
                    hs[Segs.index(phn[p])][E] = np.append(hs[Segs.index(phn[p])][E],HS[int(BG[p]):int(ED[p]),:].T,axis=1)
        print(dialect,n/len(lst))
        
    for s in range(len(Segs)):
        np.save("../data/numpy/dialect/"+"HS_" + str(dialect) + "_" + Segs[s]+ ".npy",np.array(hs[s]))
    
    



