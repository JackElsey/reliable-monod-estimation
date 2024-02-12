import numpy as np
import scipy.stats as st
from scipy.integrate import odeint
import multiprocessing as mp
import time
from scipy.optimize import differential_evolution
from os.path import exists

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

def gc_err_sd(conc):
    cov = 0.1
    return cov*conc

def qpcr_err_sd(giga_conc):
    conc = 1e9*giga_conc
    a = 2.26
    b = 0.891
    sd = a*np.power(conc, b)
    return sd/1e9

def gen_syn_data(theta, t95, seed=1234, init_dce=0.15):
    t_samp = np.linspace(0, t95, 21)
    Qa = 0.021/t95
    y0 = np.array([init_dce, 0.0, 0.0, 1E-1, 0.125, 0.035])

    ivp_soln = odeint(
        ode_fn,
        y0,
        t_samp,
        args=(theta, Qa),
        rtol=1E-3,
        atol=1E-6,
        hmin=1E-10,
        mxstep=100000,
        full_output=True
    )
    y = np.clip(ivp_soln[0], 1e-6, None)

    rs = np.random.RandomState(seed)

    gc_meas_list = []
    qpcr_meas_list = []
    for i in range(3):
        gc_meas = st.norm.rvs(loc=y[:,:3], scale=gc_err_sd(y[:,:3]), random_state=rs)
        # gc_meas = rng.normal(loc=y[:3], scale=gc_err_sd(y[:3]))
        gc_meas[gc_meas<1e-3] = 0
        gc_meas_list.append(gc_meas)
        qpcr_meas = st.norm.rvs(loc=y[::4, 3], scale=qpcr_err_sd(y[::4, 3]), random_state=rs)
        qpcr_meas_list.append(qpcr_meas)

    return np.stack(gc_meas_list), np.stack(qpcr_meas_list)

def swsr(theta, t95, gc_meas, qpcr_meas, init_dce=0.15):
    t_samp = np.linspace(0, t95, 21)
    Qa = 0.021/t95
    y0 = np.array([init_dce, 0.0, 0.0, 1E-1, 0.125, 0.035])
    ivp_soln = odeint(
        ode_fn,
        y0,
        t_samp,
        args=(theta, Qa),
        rtol=1E-3,
        atol=1E-6,
        hmin=1E-10,
        mxstep=100000,
        full_output=True
    )
    y = np.clip(ivp_soln[0], 1e-6, None)

    gc_resid = gc_meas - np.stack([y[:, :3]]*3)
    qpcr_resid = qpcr_meas - np.stack([y[::4, 3]]*3)

    gc_wsr = np.square(gc_resid[gc_meas>0]/gc_err_sd(gc_meas[gc_meas>0]))
    qpcr_wsr = np.square(qpcr_resid/qpcr_err_sd(qpcr_meas))

    return np.sum(gc_wsr) + np.sum(qpcr_wsr)


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


def calc_t95(theta, init_dce=0.15):
    t_init = 0
    t_final = 300
    n_t = 201
    h = t_final/(n_t-1)
    y0 = np.array([init_dce, 0.0, 0.0, 1E-1, 0.125, 0.035])
    t_eval=np.linspace(t_init, t_final, n_t)
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
    y = np.clip(ivp_soln[0], 0, None)
    t95 = dechlor_time(y, h)
    return t95

def run_trial(theta):
    theta_prod = np.product(theta)
    prod_ten_exp = np.floor(np.log10(theta_prod))
    seed = int(theta_prod/np.power(10, prod_ten_exp-7))

    if exists('fit_res/'+str(seed).zfill(8)+'_inhib.npy'):
        return None

    ten_exp = np.floor(np.log10(theta))
    scale = np.power(10, ten_exp)
    x_true = theta/scale

    init_dce_list = [0.01, 0.05, 0.20, 0.35, 0.50]
    x_hat_list_inhib = list()
    x_hat_list_no_inhib = list()
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

        de_bounds_inhib = np.column_stack((x_true/10, x_true*10))
        de_bounds_no_inhib = np.column_stack((x_true[[0,1,2,3,6,7]]/10, x_true[[0,1,2,3,6,7]]*10))

        de_res_inhib = differential_evolution(
            obj_fun_inhib,
            bounds=de_bounds_inhib,
            disp=False,
            popsize=15,
            seed=seed + i
        )
        x_hat_list_inhib.append(de_res_inhib.x)

        de_res_no_inhib = differential_evolution(
            obj_fun_no_inhib,
            bounds=de_bounds_no_inhib,
            disp=False,
            popsize=15,
            seed=seed + i
        )
        x_hat_list_no_inhib.append(de_res_no_inhib.x)

    filename = 'fit_res/'+str(seed).zfill(8)
    np.save(filename+'_inhib', np.stack([theta, x_true]+x_hat_list_inhib))

    np.save(filename+'_no_inhib', np.stack(x_hat_list_no_inhib))

all_samples = np.power(10, np.load('samples.npy'))

unique_sample_list = [all_samples[0]]
for sample in all_samples[1:]:
    if not np.array_equal(sample, unique_sample_list[-1]):
        unique_sample_list.append(sample)

samples = np.stack(unique_sample_list)

print('starting', samples.shape[0], 'optimizations...')
start_time = time.perf_counter()

pool = mp.Pool(processes=None)
pool.map(run_trial, samples)
pool.terminate()

elapsed_time = time.perf_counter() - start_time

print('optimizations complete')
print('total elapsed time:', elapsed_time)
print('time per parameter point:', elapsed_time/samples.shape[0])
