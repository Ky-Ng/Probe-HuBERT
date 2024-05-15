import numpy as np
class Analysis:
    def cohensd(x,y):
        mnx = np.mean(x)
        mny = np.mean(y)
        sdx = np.std(x)
        sdy = np.std(y)
        return((mnx-mny)/np.sqrt((((sdx**2)+(sdy**2))/2)))