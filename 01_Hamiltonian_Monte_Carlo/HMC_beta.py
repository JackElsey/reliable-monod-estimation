import numpy as np
import scipy.stats as st
from scipy.integrate import odeint
import multiprocessing as mp
import time

# Some code has been adapted from a tutorial by Colin Carroll.
# https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/

def ode_fn(y, t, theta, Qa=0):
#     y.clip(0, None, out=y)
    y[y<0] = 0 # faster
    c_dce = y[0]
    c_vc = y[1]
    c_eth = y[2]
    x_dhc = y[3]
    v_a = y[4]
    v_g = y[5]

    umax_dce = theta[0]
    umax_vc = theta[1]

    Ks_dce = theta[2]
    Ks_vc = theta[3]

    KI_dce = theta[4]
    KI_vc = theta[5]

    Y_dce = theta[6]
    Y_vc = theta[7]

    H_dce = 2.06
    H_vc  = 0.445
    H_eth = 0.0471

    kdl_dce = 5.00E-4
    kdl_vc = 1.68E-4
    kdl_eth = 9.36E-6

    kb_dhc = 0.01

    R = 8.314E-3
    T = 298.15

    r_dce_numer = umax_dce*x_dhc*c_dce
    I_dce = 1 + c_vc/KI_vc
    r_dce_denom = c_dce + Ks_dce*I_dce
    r_dce = r_dce_numer/r_dce_denom

    r_vc_numer = umax_vc*x_dhc*c_vc
    I_vc = 1 + c_dce/KI_dce
    r_vc_denom = c_vc + Ks_vc*I_vc
    r_vc = r_vc_numer/r_vc_denom

    R_dce = -r_dce
    R_vc = r_dce - r_vc
    R_eth = r_vc

    def dcdt(R_i, c_i, H_i, kdl_i):
        hrt = H_i*R*T
        mrd = v_a*R_i
        samp_loss = c_i*Qa/hrt
        diff_loss = kdl_i*c_i
        numer = mrd - samp_loss - diff_loss
        denom = v_a + v_g/hrt
        return numer/denom

    ddt_c_dce = dcdt(R_dce, c_dce, H_dce, kdl_dce)
    ddt_c_vc = dcdt(R_vc, c_vc, H_vc, kdl_vc)
    ddt_c_eth = dcdt(R_eth, c_eth, H_eth, kdl_eth)

    dhc_growth = Y_dce*r_dce + Y_vc*r_vc
    dhc_decay = kb_dhc*x_dhc
    ddt_x_dhc = dhc_growth - dhc_decay

    ddt_v_a = -Qa
    ddt_v_g = Qa

    return np.array([ddt_c_dce, ddt_c_vc, ddt_c_eth,
                     ddt_x_dhc, ddt_v_a, ddt_v_g])


def bump(x, a, b):
    def f(t):
        res = np.zeros(t.shape)
        res[t>0] = np.exp(-1/t[t>0])
        return res

    def g(t):
        return f(t)/(f(t)+f(1-t))

    return 1 - g((np.square(x) - a**2)/(b**2 - a**2))


def better_bump(x, min_val, max_val, trans):
    shift = (min_val + max_val)/2
    a = max_val - shift
    b = a + trans
    return bump(x-shift, a, b) + 1E-10


def calc_tot_amts(y):
    H_1D = np.array([2.06, 0.445, 0.0471])
    H = np.expand_dims(H_1D, axis=0)
    R = 8.314E-3
    T = 298.15
    HRT = H*R*T
    C = y[:, :3]
    V_a = np.expand_dims(y[:, 4], axis=1)
    V_g = np.expand_dims(y[:, 5], axis=1)
    V_g_equiv = V_g/HRT

    n_a = V_a*C
    n_g = V_g_equiv*C

    return n_a + n_g


def dechlor_time(y, h):
    n = calc_tot_amts(y)
    n_t = np.sum(n, axis=1)
    eth_frac = n[:, -1]/n_t
    i = np.searchsorted(eth_frac, 0.95)
    return i*h


def dechlor_time_dce(y, h):
    n = calc_tot_amts(y)
    n_t = np.sum(n, axis=1)
    vc_frac = n[:, 1]/n_t
    i = np.searchsorted(vc_frac, 0.95)
    return i*h


def bound_bump(theta):
    bounds = np.array([[5.50E-05, 3.20E+01],
                       [8.00E-06, 4.84E+00],
                       [6.00E-04, 3.83E-01],
                       [7.00E-04, 6.09E-01],
                       [1.80E-04, 9.51E-01],
                       [5.45E-03, 3.83E-01],
                       [1.10E+00, 1.43E+03],
                       [2.27E+00, 6.88E+04]])

    log_bounds = np.log10(bounds)
    log_theta = np.log10(theta)

    res = np.zeros(8)
    for i in range(8):
        start = log_bounds[i, 0]
        end = log_bounds[i, 1]
        trans = np.abs(0.1*end)
        res[i] = better_bump(log_theta[i], start, end, trans)

    return np.prod(res)


def tot_dechlor_bump(theta):
    t_init = 0
    t_final = 300
    n_t = 201
    h = t_final/(n_t-1)
    y0 = np.array([0.15, 0.0, 0.0, 1E-1, 0.125, 0.035])
    t_eval=np.linspace(t_init, t_final, n_t)
    try:
        ivp_soln = odeint(
            ode_fn,
            y0,
            t_eval,
            args=(theta,),
            rtol=1E-3,
            atol=1E-6,
            hmin=1E-10,
            mxstep=100000,
            full_output=True
        )
    except:
        # print('tot fail 1')
        return 1E-10, 0

    if ivp_soln[1]['tcur'][-1]<t_final:
        # print('tot fail 2')
        return 1E-10, 0
    else:
        y = np.clip(ivp_soln[0], 0, None)
        t95 = dechlor_time(y, h)
        bump_start = 40
        bump_final = 250
        bump_trans = 30
        return better_bump(t95, bump_start, bump_final, bump_trans), t95


def dce_bump(theta):
    mask = np.array([1, 0, 1, 1, 1, 10000, 1, 1])
    masked_theta = theta*mask # set VC u-max to zero, turn off VC inhibition of DCE MRD
    t_init = 0
    t_final = 150
    n_t = 101
    h = t_final/(n_t-1)
    y0 = np.array([0.15, 0.0, 0.0, 1E-1, 0.125, 0.035])
    t_eval=np.linspace(t_init, t_final, n_t)
    try:
        ivp_soln = odeint(
            ode_fn,
            y0,
            t_eval,
            args=(masked_theta,),
            rtol=1E-3,
            atol=1E-6,
            hmin=1E-10,
            mxstep=100000,
            full_output=True
        )
    except:
        # print('dce fail 1')
        return 1E-10

    if ivp_soln[1]['tcur'][-1]<t_final:
        # print('dce fail 2')
        return 1E-10
    else:
        y = np.clip(ivp_soln[0], 0, None)
        t95 = dechlor_time_dce(y, h)
        bump_start = 25
        bump_final = 125
        bump_trans = 20
        return better_bump(t95, bump_start, bump_final, bump_trans)


def vc_bump(theta):
    t_init = 0
    t_final = 150
    n_t = 101
    h = t_final/(n_t-1)
    y0 = np.array([0.0, 0.11, 0.0, 1E-1, 0.125, 0.035])
    t_eval=np.linspace(t_init, t_final, n_t)
    try:
        ivp_soln = odeint(
            ode_fn,
            y0,
            t_eval,
            args=(theta,),
            rtol=1E-3,
            atol=1E-6,
            hmin=1E-10,
            mxstep=100000,
            full_output=True
        )
    except:
        # print('vc fail 1')
        return 1E-10

    if ivp_soln[1]['tcur'][-1]<t_final:
        # print('vc fail 2')
        return 1E-10
    else:
        y = np.clip(ivp_soln[0], 0, None)
        t95 = dechlor_time(y, h)
        bump_start = 25
        bump_final = 125
        bump_trans = 20
        return better_bump(t95, bump_start, bump_final, bump_trans)


def combined_bump(theta):
    tot_dechlor_bump_val, t95 = tot_dechlor_bump(theta)
    return bound_bump(theta)*tot_dechlor_bump_val*dce_bump(theta)*vc_bump(theta), t95


def neg_unnormalized_log_prob(log_theta):
    theta = np.power(10, log_theta)
    combined_bump_val, t95 = combined_bump(theta)
    return -np.log(combined_bump_val)#, t95


def dVdq(log_theta):
    h = 0.02
    log_thetas = np.row_stack([log_theta]*16)
    log_thetas[:8] -= h*np.eye(8)
    log_thetas[8:] += h*np.eye(8)
    pool = mp.Pool(processes=None)
    fin_diffs = np.array(pool.map(neg_unnormalized_log_prob, log_thetas))
    pool.terminate()
    return (fin_diffs[8:] - fin_diffs[:8])/(2*h)


def hamiltonian_monte_carlo(n_samples, initial_position, path_len=0.05, step_size=0.01):
    """Run Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    # autograd magic
    # dVdq = grad(negative_log_prob)

    # collect all our samples in a list
    samples = [initial_position]
    num_accept = 0

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    # tqdm_iterator = tqdm(momentum.rvs(size=size), ncols=0)
    start_time = time.perf_counter()
    for p0 in momentum.rvs(size=size):
#     for p0 in momentum.rvs(size=size)[54:]:
        # tqdm_iterator.set_description('num_accept='+str(num_accept))
        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            samples[-1],
            p0,
            path_len=path_len,
            step_size=step_size,
        )

        # Check Metropolis acceptance criterion
        start_log_p = neg_unnormalized_log_prob(samples[-1]) - np.sum(momentum.logpdf(p0))
        new_neg_unnormalized_log_prob = neg_unnormalized_log_prob(q_new)
        new_log_p = new_neg_unnormalized_log_prob - np.sum(momentum.logpdf(p_new))
        if np.log(np.random.rand()) < start_log_p - new_log_p:
#             print('accept', end=' ')
            num_accept += 1
            samples.append(q_new)
            np.save('samples', np.array(samples))
            print(num_accept, len(samples), q_new)
            print(time.perf_counter()-start_time)
            # tqdm.write(str(samples[-1]))
        else:
            print('reject')
            samples.append(np.copy(samples[-1]))
            print(time.perf_counter()-start_time)

    return np.array(samples)

def leapfrog(q, p, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    global curr_q
    q, p = np.copy(q), np.copy(p)
    curr_q = q
    dVdq_val = dVdq(q)
    p -= step_size * dVdq_val / 2  # half step
    for i in range(int(path_len / step_size) - 1):
        q += step_size * p  # whole step
        curr_q = q
        try:
            dVdq_val = dVdq(q)
        except:
            dVdq_val = np.zeros(8)
        p -= step_size * dVdq_val  # whole step
    q += step_size * p  # whole step
    curr_q = q
    try:
        dVdq_val = dVdq(q)
    except:
        dVdq_val = np.zeros(8)
    p -= step_size * dVdq_val / 2  # half step

    # momentum flip at end
    return q, -p


# theta0 = np.array([1.66E-2, 9.64E-4, 1.950E-01, 1.769E-01, 1.950E-01,
#                    1.769E-01, 9.50E+1, 2.25E+2])
# start = np.log10(theta0)

np.random.seed(2345)
num_samples = 30000
# start = np.array([-1.86482545, -4.69845645, -2.71830259, -2.67394926, -3.38613001, -1.90043456, 1.66335406, 3.8135393])

# print(dVdq(start))
# start_time = time.perf_counter()
# for i in range(50):
#     _ = dVdq(start)
# print((time.perf_counter()-start_time)/50)
#
# start_time = time.perf_counter()
# for i in range(50*16):
#     _ = neg_unnormalized_log_prob(start)
# print((time.perf_counter()-start_time)/50)

prev_samples = np.load('all_samples.npy')
start = prev_samples[-1]
samples = hamiltonian_monte_carlo(num_samples, start, path_len=1.0, step_size=0.0050)


# np.save('samples', samples)
# np.save('t95s', t95s)
