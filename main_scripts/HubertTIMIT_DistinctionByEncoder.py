import numpy as np
import matplotlib.pyplot as plt
from helper_scripts.TenseLax import TenseLax
plt.close('all')


def cohensd(x, y):
    mnx = np.mean(x)
    mny = np.mean(y)
    sdx = np.std(x)
    sdy = np.std(y)
    return ((mnx-mny)/np.sqrt((((sdx**2)+(sdy**2))/2)))


for pair in TenseLax.getPairs():
    if(pair != ("uw", "uh")):
        continue
    Segs = list(pair)

    tn = 20
    CDs_D_E_T = np.zeros((25, 1024, 8, tn))

    # Create the subplot
    fig, axs = plt.subplots(2, 4)
    for d in range(1, 9):
        DistinctionAll = np.zeros((tn, 25))
        HR = []
        for s in range(len(Segs)):
            HR.append(np.load(
                f'../data/verified_cd/DR{d}/HS_{d}_{Segs[s]}.npy'))
            print(HR[s].shape)
        for token in range(tn):
            hr = []
            for s in range(len(Segs)):
                hr.append(HR[s][:, :, np.random.choice(
                    HR[s].shape[2], min(100, HR[s].shape[2]), replace=False)])
            CD = np.empty((25, 1024))
            for e in range(25):
                for v in range(1024):
                    CD[e, v] = cohensd(hr[0][e, v, :], hr[1][e, v, :])
            CDs_D_E_T[:, :, d-1, token] = CD

            Dist = np.abs(CD) > .5
            DistbyEnc = np.sum(Dist, axis=1)
            DistinctionAll[token, :] = 100*DistbyEnc/1024

        mn = np.mean(DistinctionAll, axis=0)
        sd = np.std(DistinctionAll, axis=0)

        cur_plot = axs.flatten()[d-1]

        # Plot Data
        cur_plot.plot(np.arange(25), mn, 'r-')
        cur_plot.fill_between(np.arange(25), mn-sd/2,
                              mn+sd/2, color='r', alpha=.5)
        cur_plot.set_xlim(1, 25)

        x_tick_locations = [0, 15, 25]
        cur_plot.set_xticks(x_tick_locations)
        cur_plot.set_xticklabels(x_tick_locations)

        y_tick_locations = [5, 10, 15, 20, 25]
        cur_plot.set_yticks(y_tick_locations)
        cur_plot.set_yticklabels(y_tick_locations)

        # Add Labels
        cur_plot.set_xlabel('Encoders Layer')
        cur_plot.set_ylabel('% Disting. Features')
        cur_plot.set_title(f"DR{d}")

        # Progress
        print(d)

    fig.suptitle(f"{TenseLax.getIPA(Segs[0])} vs. {TenseLax.getIPA(Segs[1])}: % Disting. Features by Encoder Layer")
    plt.tight_layout()
    plt.show()

    save_name = f'CDs_D_E_T_{Segs[0]}_{Segs[1]}'
    np.save(f'cd_comparisons/{save_name}.npy', CDs_D_E_T)
    # plt.savefig(f'../assets/percent_differentiation_graphs/{save_name}.png')
