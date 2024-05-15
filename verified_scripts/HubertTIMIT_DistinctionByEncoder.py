import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

def cohensd(x,y):
    mnx = np.mean(x)
    mny = np.mean(y)
    sdx = np.std(x)
    sdy = np.std(y)
    return((mnx-mny)/np.sqrt((((sdx**2)+(sdy**2))/2)))

Segs = ['iy','aa']
fig, ax = plt.subplots(layout="constrained",num="")  


tn = 20
CDs_D_E_T = np.zeros((25,1024,8,tn))

for d in range(1,9):
    DistinctionAll = np.zeros((tn,25))
    HR = []
    for s in range(len(Segs)):
        HR.append(np.load('/Users/khaliliskarous/Dropbox/Ling487_24/understandHubert/' + 'HS_' + str(d) + '_' + Segs[s] + '.npy'))
    for token in range(tn):
        hr = []
        for s in range(len(Segs)):
            hr.append(HR[s][:,:,np.random.choice(HR[s].shape[2],100,replace=False)])
        CD = np.empty((25,1024))
        for e in range(25):
            for v in range(1024):
                CD[e,v] = cohensd(hr[0][e,v,:],hr[1][e,v,:])
        CDs_D_E_T[:,:,d-1,token] = CD
        
        Dist = np.abs(CD) > .5
        DistbyEnc = np.sum(Dist,axis=1)
        DistinctionAll[token,:] = 100*DistbyEnc/1024
    
    mn = np.mean(DistinctionAll,axis=0)
    sd = np.std(DistinctionAll,axis=0)
    
    plt.subplot(2,4,d)
    plt.plot(np.arange(25),mn, 'r-')
    plt.fill_between(np.arange(25), mn-sd/2, mn+sd/2,color='r',alpha=.5)
    if d == 1 | d == 5:
        plt.ylabel('% Disting. Features')
    if d > 4:
        plt.xlabel('Encoders')
    plt.title(str(d))
    print(d)
    
    
np.save('CDs_D_E_T.npy',CDs_D_E_T)