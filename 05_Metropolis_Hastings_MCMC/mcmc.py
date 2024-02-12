import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from os.path import exists

def metropolis_hastings(swsr_fun, input_x, num_samples, num_chains=4, burn_in=500, seed=1234, prop_cov=None):
    """Returns samples with dimensions corresponding to iteration, chain, and theta.
    """

    rs = np.random.RandomState(seed)
    # num_chains = len(initial_thetas)
    num_params = input_x.size

    # ten_exp = np.floor(np.log10(input_theta))
    # scale = np.power(10, ten_exp)

    # If the covariance matrix of the proposal distribution hasn't been
    # supplied, calculate it using method from Bayesian Data Analysis p.295.
    if prop_cov is None:
        # print('input x:', input_x)
        fit_res = minimize(swsr_fun, input_x, method='BFGS', options={'disp':False})
        # print('optimized x:', fit_res.x)
        # prop_cov = fit_res.hess_inv*2.4/np.sqrt(num_params)
        prop_cov = fit_res.hess_inv*2.4/np.sqrt(num_params)/50
        # print(seed, np.max(np.abs(np.triu(corr_from_cov(prop_cov), k=1))))
        # prop_cov = fit_res.hess_inv*2.4*2.4/num_params

    # initial_thetas = np.abs(st.norm.rvs(loc=np.stack([input_theta/scale]*num_chains), scale=2, random_state=rs))
    xs = []
    for i in range(num_chains):
        all_positive = False
        while not all_positive:
            # prop_init_xs = st.multivariate_normal.rvs(mean=fit_res.x, cov=fit_res.hess_inv, random_state=rs)
            # prop_init_xs = st.multivariate_normal.rvs(mean=fit_res.x, cov=fit_res.hess_inv/3, random_state=rs)
            prop_init_xs = st.multivariate_normal.rvs(mean=fit_res.x, cov=fit_res.hess_inv/10, random_state=rs)
            all_positive = np.all(prop_init_xs>0)

        xs.append(prop_init_xs)
        # xs.append(st.multivariate_normal.rvs(mean=fit_res.x, cov=fit_res.hess_inv/10, random_state=rs))
        # xs.append(st.norm.rvs(loc=input_theta/scale, scale=0.5, random_state=rs))

    swsrs = [swsr_fun(x) for x in xs]
    # xs = [theta/scale for theta in initial_thetas]

    def proposal(x, random_state):
        return st.multivariate_normal.rvs(mean=x, cov=prop_cov, random_state=random_state)

    def calc_alpha(orig_swsr, prop_x):
        prop_swsr = swsr_fun(prop_x)
        return np.exp(orig_swsr/2 - prop_swsr/2), prop_swsr

    def iteration(orig_xs, orig_swsrs, random_state):
        prop_xs = [proposal(x, random_state) for x in orig_xs]
        prop_accepted = []
        new_xs = []
        new_swsrs = []
        for i in range(num_chains):
            u = st.uniform.rvs(loc=0, scale=1, random_state=random_state)
            alpha, prop_swsr = calc_alpha(orig_swsrs[i], prop_xs[i])
            if u<=alpha:
                prop_accepted.append(True)
                new_xs.append(prop_xs[i])
                new_swsrs.append(prop_swsr)
            else:
                prop_accepted.append(False)
                new_xs.append(orig_xs[i])
                new_swsrs.append(orig_swsrs[i])

        # print(prop_accepted)
        return new_xs, new_swsrs, prop_accepted

    def update_proposal(curr_samples, random_state):
        curr_samples = np.array(curr_samples)
        pooled_samples = np.row_stack([curr_samples[:,m,:] for m in range(curr_samples.shape[1])])
        samp_cov = np.cov(pooled_samples, rowvar=False)
        new_prop_cov = samp_cov*2.4/np.sqrt(num_params)

        def new_proposal(x, random_state):
            return st.multivariate_normal.rvs(mean=x, cov=new_prop_cov, random_state=random_state)

        return new_proposal

    filename = 'mcmc_res/'+str(seed).zfill(8)
    # if exists(filename):
    #     samples = [sample for sample in np.load(filename)]
    # else:
    samples = []

    num_accepts = [0]*num_chains
    for i in range(burn_in + num_samples):
        # if (i+1)<=burn_in and ((i+1)%100)==0:
        #     print('burn-in iteration', i+1)

        xs, swsrs, accepted = iteration(xs, swsrs, rs)
        if (i+1)>burn_in:
            samples.append(xs)
            # for j in range(num_chains):
            #     num_accepts[j] += accepted[j]
            if ((i+1)%1000)==0:
                np.save(filename, np.array(samples))
                # print(seed, 'Iteration', i+1-burn_in)
                # print('current xs:', xs)
                # print('swsr values:', swsrs)
                # print('x:', np.mean(np.mean(samples, axis=0), axis=0))
                # accept_rates = [np.round(num_accept/(i+1), 4) for num_accept in num_accepts]
                # print('acceptance rates:', accept_rates)


    return np.array(samples)


def r_hat(estimand_samples):
    """Calculates potential scale reduction factor (see Bayesian Data Analysis
    third edition p.284-285) for one estimand.
    """

    num_samples, num_chains = estimand_samples.shape
    m = int(num_chains*2)
    n = int(num_samples/2)

    # split each chain in two
    split_samples = np.column_stack(np.split(estimand_samples, 2, axis=0))

    # psi bar dot j
    chain_means = np.mean(split_samples, axis=0)

    # psi bar dot dot
    overall_mean = np.mean(split_samples)

    # between-sequence variance
    B = n/(m - 1)*np.sum(np.square(chain_means - overall_mean))

    # within-sequence variance
    s_squared = 1/(n - 1)*np.sum(np.square(split_samples - chain_means), axis=0)
    W = np.mean(s_squared)

    # marginal variance of the estimand (eq 11.3)
    var_hat_plus = (n-1)/n*W + B/n

    return np.sqrt(var_hat_plus/W) # (eq 11.4)

def r_hat_param_means(samples):
    num_samples, num_chains, num_params = samples.shape

    # list of samples for each parameter
    param_samples_list = [samples[:,:,param_ind] for param_ind in range(num_params)]

    return [r_hat(param_samples) for param_samples in param_samples_list]


def corr_from_cov(input_cov):
    diag_sqrt = np.sqrt(np.diag(input_cov))
    return input_cov/diag_sqrt/np.expand_dims(diag_sqrt, axis=1)


if __name__ == "__main__":
    n_param = 8
    n_chains = 4
    test_rs = np.random.RandomState(121212)

    def model(x, theta):
        return x@theta

    theta_true = st.uniform.rvs(loc=10, scale=10, size=n_param, random_state=test_rs)
    x_data = st.uniform.rvs(loc=0, scale=10, size=(500, n_param), random_state=test_rs)
    err_sd = 20
    y_data = st.norm.rvs(loc=model(x_data, theta_true), scale=err_sd, random_state=test_rs)

    def swsr(theta):
        y_mod = model(x_data, theta)
        resid = y_mod - y_data
        return np.sum(np.square(resid/err_sd))

    init_theta = st.norm.rvs(loc=np.stack([theta_true]*n_chains), scale=4, random_state=test_rs)
    samples = metropolis_hastings(swsr, init_theta, 10000, burn_in=10000)

    print(theta_true)
