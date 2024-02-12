import numpy as np
from glob import glob
import arviz as az

def calc_acceptance(data_arr):
    ident_count = np.zeros(4)
    for chain_idx in range(4):
        for row_idx in range(data_arr.shape[0]-1):
            if data_arr[row_idx,chain_idx,0] == data_arr[row_idx+1,chain_idx,0]:
                ident_count[chain_idx] += 1
    return 1 - ident_count/data_arr.shape[0]

def get_x_true(mcmc_res_file):
    seed_str = mcmc_res_file[-12:-4]
    x_true = np.load('fit_res/'+seed_str+'.npy')[1]
    return x_true

if __name__ == "__main__":
    mcmc_res_list = glob('mcmc_res/*.npy')
    all_coverage_list = list()
    for file in mcmc_res_list:
        print(file)
        try:
            data_arr = np.load(file)
            acceptance_rate = calc_acceptance(data_arr)
            print(acceptance_rate)
            print(data_arr.shape[0])
            x_true = get_x_true(file)
            inter_bnds = np.percentile(data_arr, (5,95), axis=(0,1))
            coverage = [(inter_bnds[0,i]<x_true[i]) and (x_true[i]<inter_bnds[1,i]) for i in range(8)]
            all_coverage_list.append(coverage)
            print(''.join([format(inter_bnds[0,i], '.2f')+' to '+format(inter_bnds[1,i], '.2f')+' ('+format(x_true[i], '.2f')+') '+str(coverage[i])+'\n' for i in range(8)]))
            # if np.mean(acceptance_rate)>0.01:
            data = az.convert_to_dataset(np.transpose(data_arr,axes=(1,0,2)))
            print(az.rhat(data).x.values)
            print(az.ess(data).x.values)
        except:
            print('ERROR')
        print('-----------------------------------------------------------')

    print('overall coverages')
    all_coverage = np.array(all_coverage_list)
    print(np.sum(all_coverage, axis=0)/all_coverage.shape[0])
