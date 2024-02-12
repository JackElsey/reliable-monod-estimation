import numpy as np
from glob import glob
import arviz as az
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import exists

def calc_seed(theta):
    theta_prod = np.product(theta)
    prod_ten_exp = np.floor(np.log10(theta_prod))
    seed = int(theta_prod/np.power(10, prod_ten_exp-7))
    return seed

def load_thetas(filename='samples.npy'):
    all_samples = np.power(10, np.load(filename))

    unique_sample_list = [all_samples[0]]
    for sample in all_samples[1:]:
        if not np.array_equal(sample, unique_sample_list[-1]):
            unique_sample_list.append(sample)

    samples = np.stack(unique_sample_list)
    return samples

#
# def find_missing_theta(theta, hmc_list):
#     for i, hmc in enumerate(hmc_list):
#


if __name__ == "__main__":
    burn_in = 1000
    hmc_samp_alpha = load_thetas(filename='../alpha_hmc_samp.npy')[burn_in:]
    hmc_samp_beta = load_thetas(filename='../beta_hmc_samp.npy')[burn_in:]
    hmc_samp_gamma = load_thetas(filename='../gamma_hmc_samp.npy')[burn_in:]

    seeds_alpha = [calc_seed(theta) for theta in hmc_samp_alpha]
    seeds_beta = [calc_seed(theta) for theta in hmc_samp_beta]
    seeds_gamma = [calc_seed(theta) for theta in hmc_samp_gamma]

    seen_seeds = set()
    bad_seeds = []

    for seed in tqdm(seeds_alpha+seeds_beta+seeds_gamma):
        if seed in seen_seeds:
            bad_seeds.append(seed)
        else:
            seen_seeds.add(seed)

    print('number of bad seeds:', len(bad_seeds))

    seeds_dict = dict()
    for i, seed in enumerate(seeds_alpha):
        if seed not in bad_seeds:
            seeds_dict.update({seed:('alpha', i)})

    for i, seed in enumerate(seeds_beta):
        if seed not in bad_seeds:
            seeds_dict.update({seed:('beta', i)})

    for i, seed in enumerate(seeds_gamma):
        if seed not in bad_seeds:
            seeds_dict.update({seed:('gamma', i)})

    rae_alpha = np.zeros((hmc_samp_alpha.shape[0], 5, 8))
    rae_beta = np.zeros((hmc_samp_beta.shape[0], 5 , 8))
    rae_gamma = np.zeros((hmc_samp_gamma.shape[0], 5, 8))

    rmoe_alpha = np.zeros_like(rae_alpha)
    rmoe_beta = np.zeros_like(rae_beta)
    rmoe_gamma = np.zeros_like(rae_gamma)

    fit_res_files = glob('fit_res/*.npy')
    err_count = 0
    missing_seeds = 0
    seed_mismatch = 0
    for fit_res_file in tqdm(fit_res_files):#[:2000]):
        # try:
        fit_res_arr = np.load(fit_res_file)
        theta_true = fit_res_arr[0]

        seed = calc_seed(theta_true)
        if seed in bad_seeds:
            continue

        filename_seed = int(fit_res_file[-12:-4])
        if filename_seed!=seed:
            seed_mismatch += 1

        if seed not in seeds_dict.keys():
            missing_seeds += 1
            continue

        rmoe_file = 'rmoe/'+str(seed)+'.npy'
        if not exists(rmoe_file):
            continue

        rmoe = np.load(rmoe_file)
        if np.any(np.isnan(rmoe)):
            err_count += 1
            continue

        x_true = fit_res_arr[1]
        x_hat = fit_res_arr[2:]
        rae = np.abs((x_hat-x_true)/x_true)

        chain_name, chain_idx = seeds_dict[seed]
        if chain_name=='alpha':
            rmoe_alpha[chain_idx] = rmoe#.mean(axis=0)
            rae_alpha[chain_idx] = rae#.mean(axis=0)
        elif chain_name=='beta':
            rmoe_beta[chain_idx] = rmoe#.mean(axis=0)
            rae_beta[chain_idx] = rae#.mean(axis=0)
        elif chain_name=='gamma':
            rmoe_gamma[chain_idx] = rmoe#.mean(axis=0)
            rae_gamma[chain_idx] = rae#.mean(axis=0)

        # except:
        #     pass

    print('missing seeds:', missing_seeds)
    print('seed mismatches:', seed_mismatch)

    # print(rae_alpha.shape, rae_beta.shape, rae_gamma.shape)
    rae_alpha = rae_alpha[np.all(rae_alpha>0, axis=(1,2))]
    rae_beta = rae_beta[np.all(rae_beta>0, axis=(1,2))]
    rae_gamma = rae_gamma[np.all(rae_gamma>0, axis=(1,2))]
    # print(rae_alpha.shape, rae_beta.shape, rae_gamma.shape)

    # print(rmoe_alpha.shape, rmoe_beta.shape, rmoe_gamma.shape)
    rmoe_alpha = rmoe_alpha[np.all(rmoe_alpha>0, axis=(1,2))]
    rmoe_beta = rmoe_beta[np.all(rmoe_beta>0, axis=(1,2))]
    rmoe_gamma = rmoe_gamma[np.all(rmoe_gamma>0, axis=(1,2))]
    # print(rmoe_alpha.shape, rmoe_beta.shape, rmoe_gamma.shape)

    min_chain_len = np.min([rae_alpha.shape[0], rae_beta.shape[0], rae_gamma.shape[0]])
    rae_data = az.convert_to_dataset(np.stack([np.mean(rae_chain[:min_chain_len], axis=1) for rae_chain in [rae_alpha, rae_beta, rae_gamma]]))
    rmoe_data = az.convert_to_dataset(np.stack([np.mean(rmoe_chain[:min_chain_len], axis=1) for rmoe_chain in [rmoe_alpha, rmoe_beta, rmoe_gamma]]))
    print('RAE r-hat')
    print(az.rhat(rae_data).x.values)
    print('RAE ess')
    print(az.ess(rae_data).x.values)
    print('RMOE r-hat')
    print(az.rhat(rmoe_data).x.values)
    print('RMOE ess')
    print(az.ess(rmoe_data).x.values)

    rmoe_all = np.row_stack([np.row_stack([rmoe_chain[:,i,:] for i in range(5)]) for rmoe_chain in [rmoe_alpha, rmoe_beta, rmoe_gamma]])
    print('RMOE median')
    print(np.median(rmoe_all, axis=0))

    rae_all = np.row_stack([np.row_stack([rae_chain[:,i,:] for i in range(5)]) for rae_chain in [rae_alpha, rae_beta, rae_gamma]])
    print('RAE median')
    print(np.median(rae_all, axis=0))

    # confidence region coverage
    cr_files = glob('in_cr/*.npy')
    total_trials = len(cr_files)*5

    print('calculating confidence region coverage')
    running_sum = np.zeros((1,2))
    for cr_file in tqdm(cr_files):
        running_sum += np.load(cr_file).sum(axis=0)

    print(running_sum/total_trials)


    # confidence interval coverage
    ci_files = glob('in_ci/*.npy')
    total_trials = len(ci_files)*5

    print('calculating confidence interval coverage')
    running_sum = np.zeros((1,8))
    for ci_file in tqdm(ci_files):
        running_sum += np.load(ci_file).sum(axis=0)

    print(running_sum/total_trials)


    # RMOE box plot
    fig, ax = plt.subplots(figsize=(6,7))
    ax.boxplot(rmoe_all, sym='', whis=(5,95), patch_artist=True)
    ax.set_yscale('log')
    ax.set_ylim((10**(-2.5), 10000))
    param_names = ['$u_{DCE}$', '$u_{VC}$', '$K^s_{DCE}$', '$K^s_{VC}$',
                   '$K^I_{DCE}$', '$K^I_{VC}$', '$Y_{DCE}$', '$Y_{VC}$']
    ax.xaxis.set_ticklabels(param_names)
    ax.grid(which='both', axis='y')
    ax.set_xlabel('parameter', fontsize=14)
    ax.set_ylabel('90% margin of error relative to best fit value', fontsize=14)
    plt.savefig('rmoe', bbox_inches='tight')

    # RAE box plot
    fig, ax = plt.subplots(figsize=(6,7))
    ax.boxplot(rae_all, sym='', whis=(5,95), patch_artist=True)
    ax.set_yscale('log')
    ax.set_ylim((0.001, 10))
    param_names = ['$u_{DCE}$', '$u_{VC}$', '$K^s_{DCE}$', '$K^s_{VC}$',
                   '$K^I_{DCE}$', '$K^I_{VC}$', '$Y_{DCE}$', '$Y_{VC}$']
    ax.xaxis.set_ticklabels(param_names)
    ax.grid(which='both', axis='y')
    ax.set_xlabel('parameter', fontsize=14)
    ax.set_ylabel('relative absolute error', fontsize=14)
    plt.savefig('rae', bbox_inches='tight')
