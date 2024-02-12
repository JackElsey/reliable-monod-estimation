import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import exists
import arviz as az
import multiprocessing as mp


def calc_r_hats(mcmc_res_file):
    iter_step = 200000
    # print(mcmc_res_file)
    mcmc_res = np.load(mcmc_res_file)
    if (mcmc_res.shape[0]<iter_step):
        return None
    seed_str = mcmc_res_file[-12:-4]
    num_calcs = (mcmc_res.shape[0]//iter_step) + 1
    iter_and_max_r_hat = np.zeros((num_calcs, 2))
    mcmc_conv_file = 'mcmc_conv/'+seed_str+'.npy'
    if exists(mcmc_conv_file):
        prev_calcs = np.load(mcmc_conv_file)
        iter_and_max_r_hat[:prev_calcs.shape[0]] = prev_calcs
        i = prev_calcs.shape[0]
    else:
        i = 0
    while (i+1)*iter_step<mcmc_res.shape[0]:
        curr_iter = (i+1)*iter_step
        # print(curr_iter)
        data = az.convert_to_dataset(np.transpose(mcmc_res[:curr_iter],axes=(1,0,2)))
        curr_r_hat = az.rhat(data).x.values
        # print(np.max(curr_r_hat))
        iter_and_max_r_hat[i] = (curr_iter, np.max(curr_r_hat))
        i+=1
    data = az.convert_to_dataset(np.transpose(mcmc_res,axes=(1,0,2)))
    final_r_hat = az.rhat(data).x.values
    # print(mcmc_res.shape[0])
    # print(np.max(final_r_hat))
    iter_and_max_r_hat[-1] = (mcmc_res.shape[0], np.max(final_r_hat))
    np.save(mcmc_conv_file, iter_and_max_r_hat)

if __name__ == "__main__":
    mcmc_res_files = glob('mcmc_res/*.npy')
    pool = mp.Pool(processes=4)
    pool.map(calc_r_hats, mcmc_res_files)
    pool.terminate()
