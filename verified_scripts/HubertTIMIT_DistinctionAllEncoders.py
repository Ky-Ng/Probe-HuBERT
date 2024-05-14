import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
IPA = ['i','ı','e','ɛ','æ','u','ʊ','o','ᴐ','a']

AllVCD = np.load("/Users/khaliliskarous/Google Drive/My Drive/TIMIT/FinalProject/CDs_D_E_T.npy")


for d in range(8):
    for token in range(20):
        disElements = []
        for e in range(25): 
            disElements.append(set(np.ndarray.tolist(np.argwhere(np.abs(AllVCD[e,:,d,token])>.5).flatten())))
        GoodFeatures = sorted(list(set.intersection(*disElements)))
        print(d,token,GoodFeatures)
            


