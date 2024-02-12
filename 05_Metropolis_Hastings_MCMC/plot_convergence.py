import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import exists


def choose_color(coverage):
    # colors = ['#d1c0ce80', '#d194c6', '#d100ab']
    # colors = ['lightgray', 'gold', 'darkorange']
    colors = ['lightgray', '#62b2f3', '#8661be']
    c = np.sum(coverage)
    if c<=2:
        result = colors[0]
    elif c>=3 and c<=5:
        result = colors[1]
    elif c>=6 and c<=8:
        result = colors[2]
    else:
        print('error, c =', c)

    return result


if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(5,8))
    mcmc_conv_files = glob('mcmc_conv/*.npy')
    ax.set_xlim([0, 8.5e6])
    ax.set_ylim([1e-3, 10])
    for mcmc_conv_file in mcmc_conv_files:
        mcmc_conv = np.load(mcmc_conv_file)
        max_rhat = np.max(mcmc_conv[:,1:9], axis=1)
        mean_rhat = np.mean(mcmc_conv[:,1:9], axis=1)
        for i in range(mcmc_conv.shape[0]-1):
            seg_color = choose_color(mcmc_conv[i,9:])
            # ax.plot(mcmc_conv[i:(i+2),0], max_rhat[i:(i+2)]-1, color=seg_color, linewidth=1)
            ax.plot(mcmc_conv[i:(i+2),0], mean_rhat[i:(i+2)]-1, color=seg_color, linewidth=1)
        # if np.any(mcmc_conv<=0):
        #     print(mcmc_conv_file)
        #     print(mcmc_conv)
        # ax.plot(mcmc_conv[:,0], np.clip(mcmc_conv[:,1]-1,1e-3,None), color='gray', linewidth=1)

    # ax.set_facecolor('lightgray')
    colors = ['lightgray', '#62b2f3', '#8661be']
    ax.plot([0,0], [0,0], color=colors[0], linewidth=1, label='0 to 2')
    ax.plot([0,0], [0,0], color=colors[1], linewidth=1, label='3 to 5')
    ax.plot([0,0], [0,0], color=colors[2], linewidth=1, label='6 to 8')
    ax.text(2.8e6, 8, 'LESS CONVERGED')
    ax.text(2.8e6, 1.1e-3, 'MORE CONVERGED')
    ax.text(6.3e6, 6.2, 'recovered')
    ax.text(6.2e6, 5.2, 'parameters')

    # ax.axhline(y=0.01, color='red', linestyle='--', label='heuristic convergence threshold')
    ax.legend(bbox_to_anchor=(0.7,0.8))
    ax.set_yscale('log')
    ax.set_xlabel('MCMC iteration', fontsize=14)
    ax.set_ylabel('(mean split-$\hat{R}$) - 1', fontsize=14)
    plt.savefig('iter_versus_r_hat', bbox_inches='tight')
