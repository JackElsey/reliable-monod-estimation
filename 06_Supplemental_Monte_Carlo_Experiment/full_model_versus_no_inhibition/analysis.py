import numpy as np
import scipy.stats as st
from scipy.integrate import odeint
from model_fitting_exp4 import *
from glob import glob
from os.path import exists
from tqdm import tqdm
import matplotlib.pyplot as plt


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

    if exists('../exp2/fit_res/'+seed_str+'_inhib.npy') and exists('fit_res/'+seed_str+'_first_order.npy'):
        fit_res_monod = np.load('../exp2/fit_res/'+seed_str+'_inhib.npy')
        fit_res_first_order = np.load('fit_res/'+seed_str+'_first_order.npy')
        fit_res_zero_order = np.load('fit_res/'+seed_str+'_zero_order.npy')
    else:
        print(seed, 'fit result files missing')
        return np.zeros((3,5))

    ten_exp = np.floor(np.log10(theta))
    scale = np.power(10, ten_exp)
    x_true = theta/scale
    # de_bounds_inhib = np.column_stack((x_true/10, x_true*10))
    # de_bounds_no_inhib = np.column_stack((x_true[[0,1,2,3,6,7]]/10, x_true[[0,1,2,3,6,7]]*10))
    init_dce_list = [0.01, 0.05, 0.20, 0.35, 0.50]
    swsr_monod = list()
    swsr_zero_order = list()
    swsr_first_order = list()
    for i, init_dce in enumerate(init_dce_list):
        t95 = calc_t95(theta, init_dce=init_dce)
        gc_meas, qpcr_meas = gen_syn_data(theta, t95, seed=seed+i, init_dce=init_dce)

        def obj_fun_monod(x):
            try:
                res = swsr(x*scale, t95, gc_meas, qpcr_meas, init_dce=init_dce)
            except:
                res = 1e12
            return res

        def obj_fun_zero_order(x):
            return 0
            # theta_zero_order = np.concatenate([
            #     x[:2]*scale[:2],
            #     np.array([0, 0, 1, 1]),
            #     x[-2:]*scale[-2:]
            # ])
            # try:
            #     res = swsr(theta_zero_order, t95, gc_meas, qpcr_meas, init_dce=init_dce)
            # except:
            #     res = 1e12
            # return res

        def obj_fun_first_order(x):
            theta_first_order = np.concatenate([
                x[:2]*scale[:2]*1e4,
                np.array([1e4, 1e4, 1e4, 1e4]),
                x[-2:]*scale[-2:]
            ])
            try:
                res = swsr(theta_first_order, t95, gc_meas, qpcr_meas, init_dce=init_dce)
            except:
                res = 1e12
            return res

        swsr_monod.append(obj_fun_monod(fit_res_monod[i+2]))
        swsr_zero_order.append(obj_fun_zero_order(fit_res_zero_order[i]))
        swsr_first_order.append(obj_fun_first_order(fit_res_first_order[i]))

    return np.stack([swsr_monod, swsr_zero_order, swsr_first_order])


if __name__ == "__main__":
    hmc_files = glob('../*_hmc_samp.npy')
    samples = np.row_stack([import_samples(file) for file in hmc_files])
    if exists('all_swsr.npy'):
        all_swsr = np.load('all_swsr.npy')
        print(samples.shape)
        print(compare_fit_res(samples[1000]))
    else:
        all_swsr_list = []
        for sample in tqdm(samples):
            curr_swsr = compare_fit_res(sample)
            if curr_swsr[0,0]>0:
                all_swsr_list.append(curr_swsr)
        all_swsr = np.stack(all_swsr_list)
        np.save('all_swsr', np.stack(all_swsr_list))

    print(all_swsr.shape)

    print(np.nanmedian(all_swsr[:,0,:], axis=0))
    print(np.nanmedian(all_swsr[:,1,:], axis=0))
    print(np.nanmedian(all_swsr[:,2,:], axis=0))
    print(np.sum(np.isnan(all_swsr)))

    swsr_diff = all_swsr[:,2,:] - all_swsr[:,0,:]
    print(np.nanmedian(swsr_diff, axis=0))

    init_dce_list = [0.01, 0.05, 0.20, 0.35, 0.50]

    num_trials = swsr_diff.shape[0]
    for i, init_dce in enumerate(init_dce_list):
        print('initial DCE conc =', init_dce)
        curr_swsr_diff = swsr_diff[:,i]
        print('fraction below 0:', np.sum(curr_swsr_diff<0)/num_trials)
        print('fraction above 2.303 (alpha=0.10):', np.nansum(curr_swsr_diff>2.303)/num_trials)
        print('fraction above 2.996 (alpha=0.05):', np.nansum(curr_swsr_diff>2.996)/num_trials)
        print('fraction above 4.605 (alpha=0.01):', np.nansum(curr_swsr_diff>4.605)/num_trials)

    monod_proved = np.array([np.sum(swsr_diff[:,i]>2.996)/num_trials for i in range(5)])
    fo_better = np.array([np.sum(swsr_diff[:,i]<0)/num_trials for i in range(5)])
    pie_data = np.row_stack([fo_better, (1 - monod_proved - fo_better), monod_proved])

    fig, ax = plt.subplots()
    colors = ['tab:red', 'tab:orange', 'tab:green']
    size_factor = 0.25
    for i in range(pie_data.shape[1]):
        ax.pie(pie_data[:,i], radius=(1+i)*size_factor, colors=colors, wedgeprops=dict(width=size_factor, edgecolor='w'))
    plt.savefig('monod_pie', bbox_inches='tight')
