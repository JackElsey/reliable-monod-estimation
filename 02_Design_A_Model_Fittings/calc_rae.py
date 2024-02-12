import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

rae_list = list()
fit_res_files = glob('exp1_alpha/fit_res/*.npy')+glob('exp1_beta/fit_res/*.npy')+glob('exp1_gamma/fit_res/*.npy')

for fit_res_file in tqdm(fit_res_files):
    fit_res = np.load(fit_res_file)
    x_true = fit_res[1]
    x_hat = fit_res[2:]
    rae_list.append(np.abs((x_hat-x_true)/x_true))

rae = np.row_stack(rae_list)
print(np.median(rae, axis=0))

fig, ax = plt.subplots(figsize=(6,7))
ax.boxplot(rae, sym='', whis=(5,95), patch_artist=True)
ax.set_yscale('log')
ax.set_ylim((0.001, 10))
param_names = ['$u_{DCE}$', '$u_{VC}$', '$K^s_{DCE}$', '$K^s_{VC}$',
               '$K^I_{DCE}$', '$K^I_{VC}$', '$Y_{DCE}$', '$Y_{VC}$']
ax.xaxis.set_ticklabels(param_names)
ax.set_xlabel('parameter', fontsize=14)
ax.set_ylabel('relative absolute error', fontsize=14)
plt.savefig('rae', bbox_inches='tight')
