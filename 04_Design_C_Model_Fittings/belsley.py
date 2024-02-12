import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from tqdm import tqdm


def belsley(x):
    x_scaled = x/np.sum(np.square(x), axis=0)
    u, s, vh = np.linalg.svd(x_scaled)
    cond_ind = np.max(s)/s # condition indices
    a = np.square(vh/np.expand_dims(s, axis=1))
    col_sums = np.sum(a, axis=0)
    vdp = a/np.expand_dims(col_sums, axis=0) # variance decomp proportions
    return cond_ind, vdp


def belsley_analysis(Q, ci_thres=30, vdp_thres=0.5):
    ci, vdp = belsley(Q)
    two_above_vdp_thres = np.sum(vdp>=vdp_thres, axis=1)>=2
    collin_row = np.logical_and(two_above_vdp_thres, ci>=ci_thres)
    if np.any(collin_row):
        row_idx, col_idx = np.where(vdp[collin_row]>vdp_thres)
        param_collin = np.split(col_idx, np.cumsum(np.bincount(row_idx))[:-1])
        param_collin_tuples = [tuple(param) for param in param_collin]
        return dict(zip(param_collin_tuples, ci[collin_row]))
    else:
        return dict()


def perform_belsley_analyses(Q_filelist):
    all_res = dict()
    error_count = 0
    no_collin_trials = 0
    total_trials = len(Q_filelist)

    for Q_file in tqdm(Q_filelist):
        try:
            indiv_res = belsley_analysis(np.load(Q_file))

            if len(indiv_res.keys())==0:
                no_collin_trials += 1

            for params, ci in indiv_res.items():
                if params in all_res.keys():
                    all_res[params].append(ci)
                else:
                    all_res.update({params:[ci]})
        except:
            error_count += 1

    print('trials without collinearity:', no_collin_trials)
    return all_res, error_count, total_trials


def plot_belsley_analyses(all_res, plot_collin=20):
    res_dict = all_res[0]
    total_trials = all_res[2]
    num_collin_types = len(res_dict.keys())
    trial_count_dict = {params:len(ci) for params, ci in res_dict.items()}
    collin_list, trial_count_list = zip(*trial_count_dict.items())
    sorted_collin_list =  [collin_list[idx] for idx in np.argsort(trial_count_list)[::-1]]
    sorted_trial_list = sorted(trial_count_list)[::-1]
    total_collin = np.sum(trial_count_list)
    trial_frac = np.array(sorted_trial_list[:plot_collin])/total_trials
    sorted_ci_list = [np.array(res_dict[collin]) for collin in sorted_collin_list]

    ci_percentile_05 = np.min([np.percentile(ci, 5) for ci in sorted_ci_list[:plot_collin]])
    ci_percentile_95 = np.max([np.percentile(ci, 95) for ci in sorted_ci_list[:plot_collin]])
    plot_ci_min = np.power(10, np.floor(np.log10(ci_percentile_05)))
    plot_ci_max = np.power(10, np.ceil(np.log10(ci_percentile_95)))

    print('fraction in top ten:', np.sum(trial_frac[:10]))
    avg_collin_param = np.sum(np.array([len(param) for param in sorted_collin_list])*np.array(sorted_trial_list)/total_collin)
    print('average collinear parameters:', avg_collin_param)
    print('total trials:', total_trials)
    avg_median_ci = np.sum([np.median(ci)*len(ci) for ci in sorted_ci_list])/total_collin
    print('average median ci:', avg_median_ci)

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(7,plot_collin/20*9), gridspec_kw={'width_ratios': [3, 8, 3]})
    ax[0].set_ylim([plot_collin-0.5, -0.5])

    ax[0].set_xlabel('fraction of trials')
    ax[1].set_xlabel('collinear parameters')
    ax[2].set_xlabel('condition index')

    for i in range(3):
        ax[i].xaxis.tick_top()
        ax[i].yaxis.set_visible(False)
        ax[i].xaxis.set_label_position('top')

    ax[0].barh(np.arange(plot_collin), trial_frac)
    ax[0].set_xlim(np.around(trial_frac.max()+0.05, decimals=1), 0)
    # ax[0].set_xlim(0.5, 0)

    ax[1].set_xlim([-1, 8])
    for i in range(8):
        ax[1].axvline(x=i, color='gray', zorder=1, alpha=0.5)

    for i in range(plot_collin):
        ax[1].scatter(sorted_collin_list[i], [i]*len(sorted_collin_list[i]), marker='s', color='black', zorder=2)

    ax[1].xaxis.set_ticks(np.arange(8))
    param_names = ['$u_{DCE}$',
                   '$u_{VC}$',
                   '$K^s_{DCE}$',
                   '$K^s_{VC}$',
                   '$K^I_{DCE}$',
                   '$K^I_{VC}$',
                   '$Y_{DCE}$',
                   '$Y_{VC}$']
    ax[1].xaxis.set_ticklabels(param_names)

    bplot = ax[2].boxplot(sorted_ci_list[:plot_collin], vert=False, sym='', whis=(5, 95), positions=np.arange(plot_collin), patch_artist=True)
    for patch in bplot['boxes']:
        patch.set_facecolor('white')
    ax[2].set_xscale('log')
#     ax[2].set_xlim(plot_ci_min, plot_ci_max)
    ax[2].set_xlim(1e1, 1e8)
    ax[2].grid(True, which='both')
#     ax2_xticks = np.power(10, np.arange(np.log10(plot_ci_min), np.log10(plot_ci_max)+1))
    ax2_xticks = np.power(10, np.arange(1, 8))
    ax[2].xaxis.set_minor_locator(FixedLocator(ax2_xticks))
    ax[2].xaxis.set_ticklabels([], minor=True)
#     ax[2].xaxis.set_ticks([plot_ci_min, plot_ci_max])
    ax[2].xaxis.set_ticks([1e1, 1e8])

    # fig.suptitle(name, fontsize=14, x=0.51)
    plt.savefig('collinearities', bbox_inches='tight')


def variance_boxplot(Q_files):
    print('calculating covariance matrices...')
    var_list = list()
    num_covar_err = 0
    num_load_err = 0
    for i, Q_file in enumerate(tqdm(Q_files)):
        try:
            Q = np.load(Q_file, allow_pickle=True)
        except:
            num_load_err += 1
            continue
        covar = np.linalg.inv(Q.T@Q)
        if np.any(np.isnan(covar)):
            num_covar_err += 1
        else:
            var_list.append(np.diag(covar))

    print('...calculations complete')
    print('numerical errors:', num_covar_err)
    print('loading errors:', num_load_err)

    fig, ax = plt.subplots(figsize=(4,5))
    ax.set_yscale('log')
    ax.boxplot(np.stack(var_list), sym='', whis=(5,95), patch_artist=True)
    param_names = ['$u_{DCE}$', '$u_{VC}$', '$K^s_{DCE}$', '$K^s_{VC}$',
                   '$K^I_{DCE}$', '$K^I_{VC}$', '$Y_{DCE}$', '$Y_{VC}$']
    ax.xaxis.set_ticklabels(param_names)
    ax.set_xlabel('parameter', fontsize=14)
    ax.set_ylabel('CRLB diagonal element value', fontsize=14)
    ax.yaxis.grid(which='major')
    ax.axhline(y=6.25, color='green', linestyle='--')
    plt.savefig('var_plot', bbox_inches='tight')

if __name__ == "__main__":
    Q_files = glob('Q/*.npy')
    print('plotting variance boxplot...')
    variance_boxplot(Q_files)
    print('...plotting complete')
    print('performing Belsley analyses...')
    belsley_res = perform_belsley_analyses(Q_files)
    plot_belsley_analyses(belsley_res)
    print('...Belsley analyses complete, plot generated')
