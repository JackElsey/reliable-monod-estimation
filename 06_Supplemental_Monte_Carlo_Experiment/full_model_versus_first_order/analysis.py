import numpy as np
import scipy.stats as st
from scipy.integrate import odeint
from model_fitting_exp2 import *
from glob import glob
# import multiprocessing as mp
# import time
# from scipy.optimize import differential_evolution
from os.path import exists
import matplotlib.pyplot as plt
from tqdm import tqdm

def import_samples(file):
    all_samples = np.power(10, np.load(file))

    unique_sample_list = [all_samples[0]]
    for sample in all_samples[1:]:
        if not np.array_equal(sample, unique_sample_list[-1]):
            unique_sample_list.append(sample)

    samples = np.stack(unique_sample_list)
    return samples


def calc_seed(theta):
    theta_prod = np.product(theta)
    prod_ten_exp = np.floor(np.log10(theta_prod))
    seed = int(theta_prod/np.power(10, prod_ten_exp-7))
    return seed, str(seed).zfill(8)


def compare_fit_res(theta):
    seed, seed_str = calc_seed(theta)

    if exists('fit_res/'+seed_str+'_inhib.npy'):
        fit_res_inhib = np.load('fit_res/'+seed_str+'_inhib.npy')
        fit_res_no_inhib = np.load('fit_res/'+seed_str+'_no_inhib.npy')
    else:
        # print(seed, 'fit result files missing')
        return np.zeros((2,5))

    ten_exp = np.floor(np.log10(theta))
    scale = np.power(10, ten_exp)
    x_true = theta/scale
    de_bounds_inhib = np.column_stack((x_true/10, x_true*10))
    de_bounds_no_inhib = np.column_stack((x_true[[0,1,2,3,6,7]]/10, x_true[[0,1,2,3,6,7]]*10))
    init_dce_list = [0.01, 0.05, 0.20, 0.35, 0.50]
    swsr_inhib = list()
    swsr_no_inhib = list()
    for i, init_dce in enumerate(init_dce_list):
        t95 = calc_t95(theta, init_dce=init_dce)
        gc_meas, qpcr_meas = gen_syn_data(theta, t95, seed=seed+i, init_dce=init_dce)

        def obj_fun_inhib(x):
            try:
                res = swsr(x*scale, t95, gc_meas, qpcr_meas, init_dce=init_dce)
            except:
                res = 1e12
            return res

        def obj_fun_no_inhib(x):
            theta_no_inhib = np.concatenate([
                x[:4]*scale[:4],
                np.array([1e4, 1e4]),
                x[-2:]*scale[-2:]
            ])
            try:
                res = swsr(theta_no_inhib, t95, gc_meas, qpcr_meas, init_dce=init_dce)
            except:
                res = 1e12
            return res

        swsr_inhib.append(obj_fun_inhib(fit_res_inhib[i+2]))
        swsr_no_inhib.append(obj_fun_no_inhib(fit_res_no_inhib[i]))

    return np.stack([swsr_inhib, swsr_no_inhib])


if __name__ == "__main__":
    hmc_files = glob('../*_hmc_samp.npy')
    samples = np.row_stack([import_samples(file) for file in hmc_files])
    if exists('all_swsr.npy'):
        all_swsr = np.load('all_swsr.npy')
        print(compare_fit_res(samples[1000]))
    else:
        print(samples.shape)
        print(compare_fit_res(samples[1000]))
        all_swsr_list = []
        for sample in tqdm(samples):
            all_swsr_list.append(compare_fit_res(sample))
        all_swsr = np.stack(all_swsr_list)
        np.save('all_swsr', np.stack(all_swsr_list))

    print(all_swsr.shape)
    all_swsr = all_swsr[np.nonzero(all_swsr[:,0,0])]
    print(all_swsr.shape)

    swsr_diff = all_swsr[:,1,:] - all_swsr[:,0,:]
    print(swsr_diff.shape)

    init_dce_list = [0.01, 0.05, 0.20, 0.35, 0.50]

    num_trials = swsr_diff.shape[0]
    for i, init_dce in enumerate(init_dce_list):
        print('initial DCE conc =', init_dce)
        curr_swsr_diff = swsr_diff[:,i]
        print('fraction below 0:', np.sum(curr_swsr_diff<0)/num_trials)
        print('fraction above 2.303 (alpha=0.10):', np.sum(curr_swsr_diff>2.303)/num_trials)
        print('fraction above 2.996 (alpha=0.05):', np.sum(curr_swsr_diff>2.996)/num_trials)
        print('fraction above 4.605 (alpha=0.01):', np.sum(curr_swsr_diff>4.605)/num_trials)
        print()

    inhib_proved = np.array([np.sum(swsr_diff[:,i]>2.996)/num_trials for i in range(4,-1,-1)])
    no_inhib_better = np.array([np.sum(swsr_diff[:,i]<0)/num_trials for i in range(4,-1,-1)])
    pie_data = np.row_stack([no_inhib_better, (1 - inhib_proved - no_inhib_better), inhib_proved])

    fig, ax = plt.subplots()
    colors = ['tab:red', 'tab:orange', 'tab:green']
    size_factor = 0.25
    for i in range(pie_data.shape[1]):
        ax.pie(pie_data[:,i], radius=(1+i)*size_factor, colors=colors, wedgeprops=dict(width=size_factor, edgecolor='w'))
    plt.savefig('inhib_pie', bbox_inches='tight')
