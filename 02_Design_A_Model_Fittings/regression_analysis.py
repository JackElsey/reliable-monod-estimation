import numpy as np
import scipy.stats as st
from scipy.integrate import odeint
from scipy.optimize import minimize
import time
import multiprocessing as mp

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

def gen_syn_data(theta, t95, seed=1234):
    t_samp = np.linspace(0, t95, 21)
    Qa = 0.021/t95
    y0 = np.array([0.15, 0.0, 0.0, 1E-1, 0.125, 0.035])

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
    gc_stdev_list = []
    qpcr_meas_list = []
    qpcr_stdev_list = []
    for i in range(3):
        gc_meas = st.norm.rvs(loc=y[:,:3], scale=gc_err_sd(y[:,:3]), random_state=rs)
        gc_meas[gc_meas<1e-3] = 0
        gc_meas_no_init = gc_meas[1:]
        gc_meas_list.append(gc_meas_no_init)
        gc_stdev_list.append(gc_err_sd(gc_meas_no_init))
        qpcr_meas = st.norm.rvs(loc=y[::4, 3], scale=qpcr_err_sd(y[::4, 3]), random_state=rs)
        qpcr_meas_no_init = qpcr_meas[1:]
        qpcr_meas_list.append(qpcr_meas_no_init)
        qpcr_stdev_list.append(qpcr_err_sd(qpcr_meas_no_init))

    return np.stack(gc_meas_list), np.stack(qpcr_meas_list), np.stack(gc_stdev_list), np.stack(qpcr_stdev_list)

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


def calc_t95(theta):
    t_init = 0
    t_final = 300
    n_t = 201
    h = t_final/(n_t-1)
    y0 = np.array([0.15, 0.0, 0.0, 1E-1, 0.125, 0.035])
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


def calc_seed(theta):
    theta_prod = np.product(theta)
    prod_ten_exp = np.floor(np.log10(theta_prod))
    seed = int(theta_prod/np.power(10, prod_ten_exp-7))
    return seed


def gen_flat_model_and_data(theta, seed):
    t95 = calc_t95(theta)
    gc_meas, qpcr_meas, gc_stdev, qpcr_stdev = gen_syn_data(theta, t95, seed=seed)
    gc_meas_flat = gc_meas.flatten()
    gc_meas_masked = gc_meas_flat[gc_meas_flat>0]
    gc_stdev_flat = gc_stdev.flatten()
    gc_stdev_masked = gc_stdev_flat[gc_meas_flat>0]
    qpcr_meas_flat = qpcr_meas.flatten()
    qpcr_stdev_flat = qpcr_stdev.flatten()
    y_data = np.concatenate([gc_meas_masked, qpcr_meas_flat])
    x_data = np.arange(y_data.size)
    meas_stdev = np.concatenate([gc_stdev_masked, qpcr_stdev_flat])

    def mrd_model(x, theta):
        t_samp = np.linspace(0, t95, 21)
        y0 = np.array([0.15, 0.0, 0.0, 1E-1, 0.125, 0.035])
        Qa = 0.021/t95
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

        gc_mod_flat = np.stack([y[1:,:3]]*3).flatten()
        qpcr_mod_flat = np.stack([y[4::4, 3]]*3).flatten()

        gc_mod_masked = gc_mod_flat[gc_meas_flat>0]

        y_mod = np.concatenate([gc_mod_masked, qpcr_mod_flat])

        return y_mod

    return mrd_model, x_data, y_data, meas_stdev


def est_jacobian(fun, x, theta):
    """Estimate the jacobian with finite difference approximations."""

    perturb = 0.01
    res = np.zeros((x.size, theta.size))

    # Iterate over parameters with a for loop so non-vectorizable functions
    # can be used as input.
    for i in range(theta.size):
        h = theta[i]*perturb

        theta_p1 = theta.copy()
        theta_p1[i] += h
        theta_p2 = theta.copy()
        theta_p2[i] += 2*h
        theta_m1 = theta.copy()
        theta_m1[i] -= h
        theta_m2 = theta.copy()
        theta_m2[i] -= 2*h

        fun_p1 = fun(x, theta_p1)
        fun_p2 = fun(x, theta_p2)
        fun_m1 = fun(x, theta_m1)
        fun_m2 = fun(x, theta_m2)

        # Calculate 2nd order finite difference approx. of partial derivative.
        res[:,i] = (8*fun_p1 - fun_p2 - 8*fun_m1 + fun_m2)/(12*h)

    return res

def swsr(theta, model, x_data, y_data, meas_stdev):
    """Calculate sum of weighted squared residuals (objective function)."""

    # Calculate variance of additive zero-mean Gaussian measurement error.
    meas_var = np.square(meas_stdev)

    y_mod = model(x_data, theta)
    resid = y_data - y_mod
    return np.sum(np.square(resid)/meas_var)

def analyze_regression(model, x_data, y_data, meas_stdev, theta_hat,
                       theta_true, conf_level=0.90, normalize=True):
    """Calculate approximate and exact confidence regions for calibration
       parameters.

    Input:

    model -- The model that was fit to the data. Must be of the form
    y_pred = model(x, theta).

    x_data -- Vector of independent variable values.

    y_data -- Vector of dependent variable measurements.

    meas_stdev -- Standard deviation of zero-mean additive Gaussian measurement
    errors. Must be vector of same size as x_data and y_data.

    theta_hat -- Best-fit point estimates for model calibration parameters
    theta.

    theta_true -- True values of model calibration parameters that were used to
    generate synthetic data.

    conf_level -- Confidence level (i.e., 1 - alpha) for confidence regions.

    plot_dir -- Filename of parameter-space plot. No plot is generated if this
    argument is not supplied.

    Output:

    in_approx_cr -- binary value indicating whether true parameter point
    theta_true is contained within approximate confidence region ellipse

    in_exact_cr -- binary value indicating whether true parameter point
    theta_true is contained within exact confidence region
    """

    n_meas = x_data.size # number of measurements
    n_par = theta_hat.size # number of calibration parameters theta
    dof = n_meas - n_par # degrees of freedom
    swsr_args = (model, x_data, y_data, meas_stdev) # obj. function arguments

    alpha = 1 - conf_level # significance level

    # Calculate minimum objective function value.
    min_swsr = swsr(theta_hat, model, x_data, y_data, meas_stdev)

    # Calculate bounds of ellipse approximating confidence region using method
    # explained in Draper and Smith (1998) pp. 221-223 for weighted least
    # squares, except that the matrix X is replaced by the jacobian from above.
    if normalize:
        theta_power = np.floor(np.log10(theta_hat))
        norm_diag = np.diag(np.power(10, theta_power)) # normalizing diag matrix
    else:
        norm_diag = np.identity(theta_hat.size)

    def norm_model(x, norm_theta):
        return model(x, norm_diag@norm_theta)

    # Estimate jacobian of model predictions wrt model parameters theta.
    theta_hat_mant = theta_hat@np.linalg.inv(norm_diag) # mantissa
    jacob = est_jacobian(norm_model, x_data, theta_hat_mant)

    Q = np.linalg.inv(np.diag(meas_stdev))@jacob
    A = Q.T@Q
    A_inv = np.linalg.inv(A)
    #
    # eig_val, eig_vec = np.linalg.eig(A_inv)

    # Z = y_data/meas_stdev
    F = n_par/(n_meas-n_par)*st.f.ppf(conf_level, n_par, dof)
    approx_thres = F*min_swsr

    # T = eig_vec@np.diag(np.sqrt(np.abs(eig_val)))
    # omega = np.linspace(0,2*np.pi,100)
    # d = np.vstack((np.cos(omega),np.sin(omega)))
    # trans_d = T@d
    # bnd_coef = np.sqrt(approx_thres)
    # ellipse = norm_diag@(bnd_coef*trans_d) + np.tile(np.expand_dims(theta_hat,1),(1,trans_d.shape[1]))

    # Determine if point describing true parameter values is enclosed in
    # approximate confidence region.
    theta_true_mant = theta_true@np.linalg.inv(norm_diag)
    diff = theta_hat_mant - theta_true_mant
    if (diff.T@Q.T@Q@diff) <= approx_thres:
        in_approx_cr = 1
    else:
        in_approx_cr = 0

    # Using methods described in Bates and Watts (1988), calculate threshold
    # objective function value describing boundary of exact confidence region.
    exact_thres = min_swsr*(1+F)

    # Determine if point describing true parameter values is enclosed in
    # exact confidence region.
    if swsr(theta_true, model, x_data, y_data, meas_stdev)<=exact_thres:
        in_exact_cr = 1
    else:
        in_exact_cr = 0

    return in_approx_cr, in_exact_cr, A_inv


def load_thetas(filename='samples.npy'):
    all_samples = np.power(10, np.load(filename))

    unique_sample_list = [all_samples[0]]
    for sample in all_samples[1:]:
        if not np.array_equal(sample, unique_sample_list[-1]):
            unique_sample_list.append(sample)

    samples = np.stack(unique_sample_list)
    return samples


def load_theta_hats(theta):
    seed = calc_seed(theta)
    ten_exp = np.floor(np.log10(theta))
    scale = np.power(10, ten_exp)
    fit_res_arr = np.load('exp1_alpha/fit_res/'+str(seed)+'.npy')
    return fit_res_arr[-5:]*scale


def mrd_ra(theta):
    try:
        seed = calc_seed(theta)
        theta_hats = load_theta_hats(theta)
        in_cr_list = list()
        cov_list = list()
        for i, theta_hat in enumerate(theta_hats):
            mrd_model, x_data, y_data, meas_stdev= gen_flat_model_and_data(theta, seed=seed+i)
            res = analyze_regression(mrd_model, x_data, y_data, meas_stdev, theta_hat, theta, conf_level=0.90)
            in_cr_list.append(res[:2])
            cov_list.append(res[-1])
        np.save('ra_res/'+str(seed)+'_in_cr', np.stack(in_cr_list))
        np.save('ra_res/'+str(seed)+'_cov', np.stack(cov_list))
    except:
        pass


def grad_optim_ra(theta):
    ten_exp = np.floor(np.log10(theta))
    scale = np.power(10, ten_exp)
    x_true = theta/scale

    try:
        seed = calc_seed(theta)
        theta_hats = load_theta_hats(theta)
        res = np.zeros((5,2))
        for i, theta_hat in enumerate(theta_hats):
            mrd_model, x_data, y_data, meas_stdev= gen_flat_model_and_data(theta, seed=seed+i)

            def obj_fun(x):
                res = swsr(x*scale, mrd_model, x_data, y_data, meas_stdev)
                return res

            optim_res = minimize(obj_fun, x_true)
            theta_hat = optim_res.x*scale
            res[i] = analyze_regression(mrd_model, x_data, y_data, meas_stdev, theta_hat, theta, conf_level=0.90)

        np.save('grad_ra_res/'+str(seed), res)
    except:
        pass


if __name__ == "__main__":
    thetas = load_thetas()
    print(thetas.shape)
    start_time = time.perf_counter()

    pool = mp.Pool(processes=4)
    # pool.map(mrd_ra, thetas)
    pool.map(grad_optim_ra, thetas)
    pool.terminate()

    elapsed_time = time.perf_counter() - start_time
    print(elapsed_time)

    # tot_points = 0
    # tot_in_approx_cr = 0
    # tot_in_exact_cr = 0
    # start = perf_counter()
    # for theta0 in thetas[:20]:
    #     seed0 = calc_seed(theta0)
    #     print(seed0)
    #     try:
    #         theta_hats0 = load_theta_hats(theta0)
    #         tot_points += 1
    #         for i, theta_hat0 in enumerate(theta_hats0):
    #             mrd_model0, x_data0, y_data0, meas_stdev0= gen_flat_model_and_data(theta0, seed=seed0+i)
    #             # print(swsr(theta_hat0, mrd_model0, x_data0, y_data0, meas_stdev0))
    #             # print(swsr(theta0, mrd_model0, x_data0, y_data0, meas_stdev0))
    #             res = analyze_regression(mrd_model0, x_data0, y_data0, meas_stdev0, theta_hat0, theta0, conf_level=0.90)
    #             print(res)
    #             tot_in_approx_cr += res[0]
    #             tot_in_exact_cr += res[1]
    #     except:
    #         print('model fits not found')
    #
    # print(perf_counter() - start)
    # print(tot_points, tot_in_approx_cr, tot_in_exact_cr)
