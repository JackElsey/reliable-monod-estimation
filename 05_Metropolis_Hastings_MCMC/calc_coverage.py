import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import exists
import multiprocessing as mp


def get_x_true(mcmc_res_file):
    seed_str = mcmc_res_file[-12:-4]
    x_true = np.load('fit_res/'+seed_str+'.npy')[1]
    return x_true

def calc_coverage(mcmc_res_file):
    x_true = get_x_true(mcmc_res_file)
    iter_step = 200000
    # print(mcmc_res_file)
    mcmc_res = np.load(mcmc_res_file)
    if (mcmc_res.shape[0]<iter_step):
        return None
    seed_str = mcmc_res_file[-12:-4]
    num_calcs = (mcmc_res.shape[0]//iter_step) + 1
    iter_and_coverage = np.zeros((num_calcs, 9))
    mcmc_coverage_file = 'mcmc_coverage/'+seed_str+'.npy'
    if exists(mcmc_coverage_file):
        prev_calcs = np.load(mcmc_coverage_file)
        iter_and_coverage[:prev_calcs.shape[0]] = prev_calcs
        i = prev_calcs.shape[0]
    else:
        i = 0
    while (i+1)*iter_step<mcmc_res.shape[0]:
        curr_iter = (i+1)*iter_step
        # print(curr_iter)
        inter_bnds = np.percentile(mcmc_res[:curr_iter], (5,95), axis=(0,1))
        curr_coverage = [(inter_bnds[0,i]<x_true[i]) and (x_true[i]<inter_bnds[1,i]) for i in range(8)]
        # print(curr_coverage)
        iter_and_coverage[i,0] = curr_iter
        iter_and_coverage[i,1:] = curr_coverage
        i+=1
    inter_bnds = np.percentile(mcmc_res, (5,95), axis=(0,1))
    final_coverage = [(inter_bnds[0,i]<x_true[i]) and (x_true[i]<inter_bnds[1,i]) for i in range(8)]
    # print(mcmc_res.shape[0])
    # print(final_coverage)
    iter_and_coverage[-1,0] = mcmc_res.shape[0]
    iter_and_coverage[-1,1:] = final_coverage
    np.save(mcmc_coverage_file, iter_and_coverage)

if __name__ == "__main__":
    mcmc_res_files = glob('mcmc_res/*.npy')
    # calc_coverage(mcmc_res_files[0])
    pool = mp.Pool(processes=4)
    pool.map(calc_coverage, mcmc_res_files)
    pool.terminate()
