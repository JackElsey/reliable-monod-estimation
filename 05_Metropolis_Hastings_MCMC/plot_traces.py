import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

res_file = 'mcmc_res/10060481.npy'
mcmc_arr = np.load(res_file)
seed_str = res_file[-12:-4]

for i in range(8):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(mcmc_arr[:,:,i])
    # ax.plot(mcmc_arr[::1000,:,i])
    plt.savefig('trace_plots/'+seed_str+'_'+str(i), bbox_inches='tight')
    plt.close()
