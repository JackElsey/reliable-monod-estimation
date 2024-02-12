import numpy as np
import matplotlib.pyplot as plt
from glob import glob

fig, ax = plt.subplots(figsize=(5,8))
mcmc_conv_files = glob('mcmc_conv/*.npy')
ax.set_ylim([1e-3, 10])
for mcmc_conv_file in mcmc_conv_files:
    mcmc_conv = np.load(mcmc_conv_file)
    if np.any(mcmc_conv<=0):
        print(mcmc_conv_file)
        print(mcmc_conv)
    ax.plot(mcmc_conv[:,0], np.clip(mcmc_conv[:,1]-1,1e-3,None), color='gray', linewidth=1)

ax.axhline(y=0.01, color='red', linestyle='--', label='heuristic convergence threshold')
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('MCMC iteration', fontsize=14)
ax.set_ylabel('split-$\hat{R}$', fontsize=14)
plt.savefig('iter_versus_r_hat', bbox_inches='tight')
