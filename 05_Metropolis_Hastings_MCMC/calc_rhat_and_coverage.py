import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import exists
import multiprocessing as mp
import arviz as az

def get_x_true(mcmc_res_file):
    seed_str = mcmc_res_file[-12:-4]
    x_true = np.load('fit_res/'+seed_str+'.npy')[1]
    return x_true

def calc_coverage(mcmc_res_file):
    seed_str = mcmc_res_file[-12:-4]
    mcmc_conv_file = 'mcmc_conv/'+seed_str+'.npy'
    if exists(mcmc_conv_file):
        return None

    x_true = get_x_true(mcmc_res_file)
    iter_step = 200000
    # print(mcmc_res_file)
    mcmc_res = np.load(mcmc_res_file)
    if (mcmc_res.shape[0]<iter_step):
        return None
    num_calcs = (mcmc_res.shape[0]//iter_step) + 1
    iter_and_coverage = np.zeros((num_calcs, 17))

    for i, curr_iter in enumerate(list(range(iter_step, mcmc_res.shape[0], iter_step))+[mcmc_res.shape[0]]):
        # print(curr_iter)
        iter_and_coverage[i,0] = curr_iter
        data = az.convert_to_dataset(np.transpose(mcmc_res[:curr_iter],axes=(1,0,2)))
        curr_r_hat = az.rhat(data).x.values
        # print(curr_r_hat)
        iter_and_coverage[i,1:9] = curr_r_hat
        inter_bnds = np.percentile(mcmc_res[:curr_iter], (5,95), axis=(0,1))
        curr_coverage = [(inter_bnds[0,i]<x_true[i]) and (x_true[i]<inter_bnds[1,i]) for i in range(8)]
        # print(curr_coverage)
        iter_and_coverage[i,9:] = curr_coverage
    # print(iter_and_coverage)
    np.save(mcmc_conv_file, iter_and_coverage)


if __name__ == "__main__":
    # mcmc_res_files = glob('mcmc_res/*.npy')
    fit_res_file_list = sorted(glob('fit_res/*.npy'))
    file_subset = ['mcmc_res/'+file[8:] for file in fit_res_file_list[64:96]]
    # calc_coverage(mcmc_res_files[0])
    pool = mp.Pool(processes=16)
    pool.map(calc_coverage, file_subset)
    pool.terminate()
