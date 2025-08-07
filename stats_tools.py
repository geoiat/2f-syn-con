import numpy as np
from scipy.stats import binned_statistic, linregress, bootstrap
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.linalg import lstsq
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm


def curate_data(data_mat, dt, n_step_smooth=1, cell_vec=None):
    print('Curation... ', end='', flush=True)
    # book-keeping
    n_syn, n_t = data_mat.shape
    n_dt = n_t - 1

    # storage for clean data
    clean_dict = {}

    # save copy of raw data
    clean_dict['raw'] = data_mat

    # save vector with cell indices
    if cell_vec is not None:
        clean_dict['cell_vec'] = cell_vec
    else:
        clean_dict['cell_vec'] = np.ones(n_syn, dtype=int)

    # list for the evaluated step sizes
    dt_lst = []

    # loop through all possible step sizes
    for n_steps in range(1, n_t):

        # lists to store all postive and negative dw
        pos_lst = []
        neg_lst = []

        # calculate last possible start point
        t_last = np.minimum(n_steps, n_t - n_steps)

        # loop though all possible start points
        for i_start in range(t_last):
            w = data_mat[:,i_start:n_t:n_steps]
            dw = np.diff(w, axis=1)

            # discard last time step, so w and dw have the same shape
            w = w[:,:-1]

            # remove nan values and flatten
            idx_finite = np.isfinite(dw)
            if not idx_finite.any():
                continue

            w_finite = w[idx_finite]
            dw_finite = dw[idx_finite]

            wdw_finite = np.array([w_finite, dw_finite]).T

            # separate positive and negative dw
            idx_pos = dw_finite>0
            idx_neg = dw_finite<0
            pos_lst.append(wdw_finite[idx_pos,:])
            neg_lst.append(wdw_finite[idx_neg,:])

        # continue if no measurements with this step size
        if not (pos_lst and neg_lst):
            continue

        # make lists to matrices
        pos_mat = np.concatenate(pos_lst, axis=0)
        neg_mat = np.concatenate(neg_lst, axis=0)

        # put in a dict
        clean_dict[n_steps] = {}
        clean_dict[n_steps]['+'] = pos_mat
        clean_dict[n_steps]['-'] = neg_mat

        # save step size
        dt_lst.append(n_steps)

    # save vector with time intervals in hours
    clean_dict['dt'] = np.array(dt_lst)*dt #np.arange(1, n_t)*dt

    # all w standardized
    w_indep = data_mat[:,0:n_t:n_step_smooth]
    w_save = w_indep[w_indep>0]
    w_save_norm = lognormalize(w_save)
    clean_dict['w'] = w_save_norm

    # all positive dw standardized
    # dw_indep = np.diff(w_indep, axis=1)
    # dw_pos = dw_indep[dw_indep>0]
    # dw_neg = np.abs(dw_indep[dw_indep<0])
    # dw_pos_norm = lognormalize(dw_pos)
    # dw_neg_norm = lognormalize(dw_neg)
    # clean_dict['dw'] = {}
    # clean_dict['dw']['+'] = dw_pos_norm
    # clean_dict['dw']['-'] = dw_neg_norm

    print('Done!')

    return clean_dict


def slope_vs_dt(data_dict, n_resamples, seed, slope='conv'):
    print('Slopes... ', end='', flush=True)
    rng = np.random.default_rng(seed)
    n_steps = data_dict['dt'].size
    step_vec = np.array([key for key in data_dict if isinstance(key, int)])

    # storage
    data_dict['slopes'] = {}

    # loop over ltp and ltd data
    for sign in ['+', '-']:
        slope_mean = []
        slope_std = []
        dt = []

        # loop over all step sizes dt for dw(dt)
        for i_step in range(n_steps):
            step = step_vec[i_step]
            data_mat = data_dict[step][sign]

            # skip this dt if number of measurements is less than 200
            n_rows, n_cols = data_mat.shape
            if n_rows < 50: # <-------------- this sets minimum number of (w,dw) measurements required, default=200
                continue

            # calculate the bootstrapped slope
            mean, std = boot_slope(data_mat, n_resamples=n_resamples, rng=rng, slope=slope)
            slope_mean.append(mean)
            slope_std.append(std)
            dt.append(data_dict['dt'][i_step])

        # put results back in data dictionary
        data_dict['slopes'][sign] = np.array([dt, slope_mean, slope_std]).T

    data_dict['slopes']['n_resamples'] = n_resamples

    print('Done!')

    return data_dict


def boot_slope(data_mat, n_resamples, rng, slope):
    n_data, _ = data_mat.shape
    slope_vec = np.zeros(n_resamples)

    # loop over all bootstrap samples
    for i_resample in range(n_resamples):
        # draw the new bootstrap
        idx_resamples = rng.integers(low=0, high=n_data, size=n_data)
        data_resample = data_mat[idx_resamples,:]

        # calculate slope for the bootstrap sample
        if slope == 'pow':
            slope_vec[i_resample] = pow_slope(data_resample)
        elif slope == 'conv':
            slope_vec[i_resample] = conv_slope(data_resample)
        elif slope == 'bin':
            slope_vec[i_resample] = bin_slope(data_resample)
        elif slope == 'raw':
            slope_vec[i_resample] = raw_slope(data_resample)

    # return mean and std across bootstrap samples
    return slope_vec.mean(), slope_vec.std()


def conv_slope(data_mat):
    # book-keeping
    w = data_mat[:,0]
    dw = np.abs(data_mat[:,1])
    n_data = w.size

    # set window size for moving average
    n_width = np.maximum(round(n_data/20), 10)

    # sorting before moving average
    idx_sort = np.argsort(w)
    w_sorted = w[idx_sort]
    dw_sorted = dw[idx_sort]

    # apply moving average
    dw_filt = uniform_filter1d(dw_sorted, size=n_width, mode='constant', cval=-1e10)

    # remove points based on data padding
    idx_filt = dw_filt>0
    dw_filt = dw_filt[idx_filt]
    w_filt = w_sorted[idx_filt]

    # fit line to moving average
    regr_filt = linregress(np.log10(w_filt), np.log10(dw_filt))

    # return slope
    return regr_filt.slope


def bin_slope(data_mat):
    # book-keeping
    w = data_mat[:,0]
    dw = np.abs(data_mat[:,1])
    n_data = w.size

    # number of bins
    n_bin = round(n_data/20)

    # the bin edges so that each bin has about the same number of data points
    bin_equal = np.interp(np.linspace(0, n_data, n_bin + 1), np.linspace(0, n_data, n_data), np.sort(np.log10(w)))

    # get the binned averages
    bin_stats, bin_edges, bin_number = binned_statistic(np.log10(w), dw, bins=bin_equal, statistic='mean')

    # calculate bin middles
    bin_middle = (10**bin_edges[:-1] + 10**bin_edges[1:])/2

    # fit line to binned averages
    regr_bin = linregress(np.log10(bin_middle), np.log10(bin_stats))

    # fit power law to binned averages in normal scale
    #popt, pcov = curve_fit(f=f, jac=J, xdata=bin_middle, ydata=bin_stats, p0=np.array([1, 1]), method='lm')

    # return slope
    return regr_bin.slope


def pow_slope(data_mat):
    # book-keeping
    w = data_mat[:,0]
    dw = np.abs(data_mat[:,1])

    # fit power law to all data in normal scale
    popt, _ = curve_fit(f=f, jac=J, xdata=w, ydata=dw, p0=np.array([1., 1.]), method='lm')

    # return slope
    return popt[1]


def raw_slope(data_mat):
    # book-keeping
    w = data_mat[:,0]
    dw = np.abs(data_mat[:,1])

    # fit line to log of data as it is
    regr_raw = linregress(np.log10(w), np.log10(dw))

    return regr_raw.slope


# power law
def f(x, a, b):
    return a*x**b

# Jacobian of power law
def J(x, a, b):
    return np.array([x**b, a*x**(b)*np.log(x)]).T


# DEPRECATED remove data points that are very far from the mean relative the std (in log scale)
# threshold given in unit of sigmas
# def remove_log_outliers(data_mat, threshold):
#     log_w = np.log10(data_mat[:,0])
#     log_dw = np.log10(data_mat[:,1])
#     idx_w =  (log_w < log_w.mean()-threshold*log_w.std()) | (log_w > log_w.mean()+threshold*log_w.std())
#     idx_dw = (log_dw < log_dw.mean()-threshold*log_dw.std()) | (log_dw > log_dw.mean()+threshold*log_dw.std())
#     idx_out = idx_w | idx_dw
#     return data_mat[~idx_out,:], idx_out.sum()

# remove data points that are very far from the mean relative the std (in log scale)
# threshold given in unit of sigmas
def remove_log_outliers(data_mat, sigmas):
    # logarithmize data
    log_data_mat = np.log10(data_mat)
    # z-transform
    z_mat = np.abs(log_data_mat - log_data_mat.mean(axis=0))/log_data_mat.std(axis=0)
    # we combine the outliersin both the independent and dependent variable
    idx_out = np.any(z_mat > sigmas, axis=1)
    return data_mat[~idx_out,:], data_mat[idx_out,:]


# remove data points that are very far from the mean relative the std
# threshold given in unit of sigmas
def remove_outliers(data_mat, sigmas):
    # z-transform
    z_mat = np.abs(data_mat - np.mean(data_mat, axis=0))/(np.std(data_mat, axis=0) + 1e-20)
    # we combine the outliersin both the independent and dependent variable
    idx_out = np.any(z_mat > sigmas, axis=1)
    return data_mat[~idx_out,:], data_mat[idx_out,:]


def get_extended_data(data_mat, interaction_degree, add_bias):
    # extend input varaibles with bias and cross-terms
    kernel_transformer = PolynomialFeatures(interaction_degree, include_bias=add_bias, interaction_only=True)
    return kernel_transformer.fit_transform(data_mat)
    


# standardize vector in logarithmic scale
def lognormalize(x):
    log_x = np.log10(x)
    log_x_norm = (log_x - log_x.mean())/log_x.std()
    return 10**(log_x_norm)


# extract CV for each neuron with bootstrap
def cv(data_dict, n_resamples, n_steps, seed):
    print('CVs... ', end='', flush=True)
    rng = np.random.default_rng(seed)
    data_mat = data_dict['raw']
    cell_vec = data_dict['cell_vec']
    n_cells = int(cell_vec.max())

    # extract data only for surviving synapses
    idx_alive = data_mat[:,n_steps-1]>0
    data_mat = data_mat[idx_alive,:n_steps]
    cell_vec = cell_vec[idx_alive]

    # z_vec = np.arange(1, 5.2, 0.2)
    # n_z = z_vec.size
    norm_vec = np.arange(0.2, 2.6, 0.1)
    n_norms = norm_vec.size

    # storage
    cv_all = np.zeros((n_norms,n_cells,n_resamples))
    cv_means = np.zeros((n_norms,3))
    norm_min = np.zeros(2)

    # loop over all norms in terms of z
    for i_z in range(n_norms):
        norm = norm_vec[i_z] #2/z_vec[i_z]
        cv_mat = np.zeros((n_cells,n_resamples))
        cv_mat.fill(np.nan)

        # loop over all cells in the raw data
        for i_cell in range(n_cells):
            #print(z_vec[i_z], i_cell)
            idx_cell = cell_vec==(i_cell+1)
            cell_data = data_mat[idx_cell,:]

            if cell_data.size==0:
                continue

            cell_cv = boot_cv(cell_data, norm=norm, n_resamples=n_resamples, rng=rng)
            cv_mat[i_cell,:] = cell_cv

        # save all CVs
        cv_all[i_z,:,:] = cv_mat#.flatten()

    cv_all = (cv_all - cv_all.min(axis=0, keepdims=True))/(cv_all.max(axis=0, keepdims=True) - cv_all.min(axis=0, keepdims=True))
    # cv_all = cv_all/cv_all.min(axis=0)
    # cv_all = (cv_all - cv_all.mean(axis=0))/cv_all.std(axis=0)

    cv_all_mean = np.mean(cv_all, axis=2)
    cv_all_var = np.var(cv_all, axis=2)
    cv_all_weights = 1/cv_all_var
    cv_weighted_mean = np.sum(cv_all_mean*cv_all_weights, axis=1)/np.sum(cv_all_weights, axis=1)
    cv_weighted_sem = 1/np.sqrt( np.sum(cv_all_weights, axis=1) )

    # save the mean and std of CVs across cells
    cv_means[:,0] = norm_vec
    cv_means[:,1] = cv_weighted_mean #np.nanmean(cv_all, axis=1)
    cv_means[:,2] = cv_weighted_sem #np.nanstd(cv_all, axis=1)#/np.sqrt(n_cells)

    # calculate bootstrapped minimum norm
    # cv_all = cv_all[ :,~np.all(np.isnan(cv_all), axis=0) ]
    idx_min = np.nanargmin(cv_all, axis=0, keepdims=True)
    norm_min_vec = norm_vec[idx_min]
    norm_all_mean = np.mean(norm_min_vec, axis=2)
    norm_all_var = np.var(norm_min_vec, axis=2)
    norm_all_weights = 1/norm_all_var
    norm_weighted_mean = np.sum(norm_all_mean*norm_all_weights, axis=1)/np.sum(norm_all_weights, axis=1)
    norm_weighted_sem = 1/np.sqrt( np.sum(norm_all_weights, axis=1) )
    norm_min[0] = norm_weighted_mean #np.nanmean(norm_min_vec)
    norm_min[1] = norm_weighted_sem #np.nanstd(norm_min_vec)#/np.sqrt(n_cells)

    # save CV matrix to dict
    data_dict['cv'] = cv_means
    data_dict['qmin'] = norm_min
    data_dict['cv_all'] = cv_all

    print('Done!')

    return data_dict


# extract CV for data coming from a single neuron with bootstrap
def cv_single(data_dict, n_resamples, n_steps, seed):
    print('CVs... ', end='', flush=True)
    rng = np.random.default_rng(seed)
    data_mat = data_dict['raw']

    # extract data only for surviving synapses
    idx_alive = data_mat[:,n_steps-1]>0
    data_mat = data_mat[idx_alive,:n_steps]

    # z_vec = np.arange(1, 5.2, 0.2)
    # n_z = z_vec.size
    norm_vec = np.arange(0.1, 4.5, 0.1)
    n_norms = norm_vec.size

    # storage
    cv_all = np.zeros((n_norms,n_resamples))
    cv_means = np.zeros((n_norms,3))
    norm_min = np.zeros(2)

    # loop over all norms in terms of z
    for i_z in range(n_norms):
        norm = norm_vec[i_z] #2/z_vec[i_z]
        cv_vec = boot_cv(data_mat, norm=norm, n_resamples=n_resamples, rng=rng)
        cv_all[i_z,:] = cv_vec#.flatten()

    cv_all = (cv_all - cv_all.min(axis=0, keepdims=True))/(cv_all.max(axis=0, keepdims=True) - cv_all.min(axis=0, keepdims=True))
    # cv_all = cv_all/cv_all.min(axis=0)
    # cv_all = (cv_all - cv_all.mean(axis=0))/cv_all.std(axis=0)

    cv_all_mean = np.mean(cv_all, axis=1)
    cv_all_std = np.std(cv_all, axis=1)

    # save the mean and std of CVs across cells
    cv_means[:,0] = norm_vec
    cv_means[:,1] = cv_all_mean #np.nanmean(cv_all, axis=1)
    cv_means[:,2] = cv_all_std #np.nanstd(cv_all, axis=1)#/np.sqrt(n_cells)

    # calculate bootstrapped minimum norm
    # cv_all = cv_all[ :,~np.all(np.isnan(cv_all), axis=0) ]
    idx_min = np.nanargmin(cv_all, axis=0, keepdims=True)
    norm_min_vec = norm_vec[idx_min]
    norm_all_mean = np.mean(norm_min_vec, axis=1)
    norm_all_std = np.std(norm_min_vec, axis=1)
    norm_min[0] = norm_all_mean #np.nanmean(norm_min_vec)
    norm_min[1] = norm_all_std #np.nanstd(norm_min_vec)#/np.sqrt(n_cells)

    #norm_min[0] = norm_vec[np.nanargmin(cv_all_mean)] #np.nanmean(norm_min_vec)

    # save CV matrix to dict
    data_dict['cv'] = cv_means
    data_dict['qmin'] = norm_min
    data_dict['cv_all'] = cv_all

    print('Done!')

    return data_dict


def boot_cv(data_mat, norm, n_resamples, rng):
    n_syn, n_t = data_mat.shape
    norm_mat = np.zeros((n_resamples,n_t))

    # first sample is the original data
    norm_mat[0,:] = np.nansum(data_mat**norm, axis=0)**(1/norm)

    if n_resamples > 1:
        # loop over all bootstrap resamples
        for i_resample in range(1,n_resamples):
            # draw the new bootstrap
            idx_resamples = rng.integers(low=0, high=n_syn, size=(n_syn,n_t))

            # create a bootstrapped neuron with n_syn measurements over n_t time points
            data_resample = np.take_along_axis(data_mat, idx_resamples, axis=0)

            # calculate number of synapses at each time point
            # n_syn_vs_t = np.isfinite(data_resample).sum(axis=0)
            # n_syn_vs_t = np.where(n_syn_vs_t==0, np.nan, n_syn_vs_t)

            # calculate the norm at each time point and put in the norm matrix
            norm_mat[i_resample,:] = np.nansum(data_resample**norm, axis=0)**(1/norm)#/(n_syn_vs_t)**(1/norm) #np.linalg.norm(data_resample, ord=norm, axis=0)

    # calculate cv for all bootrapped neurons
    cv_vec = np.nanstd(norm_mat, axis=1)/np.nanmean(norm_mat, axis=1)

    return cv_vec


# extract norm of spine fluorescence over time
def norm_vs_t(data):
    cell_vec = data[:,0]
    n_cells = int(cell_vec.max())

    data = data[:,1:]
    _, n_t = data.shape

    z_vec = np.arange(1, 10, 0.2)
    n_z = z_vec.size

    norm_dict = {}

    # loop over all norms in terms of z
    for i_z in range(n_z):
        norm = 2/z_vec[i_z]
        norm_mat = np.zeros((n_cells,n_t))

        # loop over all cells in the raw data
        for i_cell in range(n_cells):
            idx_cell = cell_vec==(i_cell+1)
            cell_data = data[idx_cell,:]

            # calculate number of synapses at each time point
            n_syn = np.isfinite(cell_data).sum(axis=0)
            n_syn = np.where(n_syn==0, np.nan, n_syn)

            # calculate the norm of spine intensities for each time point
            norm_mat[i_cell,:] = np.nansum(cell_data**norm, axis=0)**(1/norm)/(n_syn)**(1/norm)

        # store norm matrix
        norm_dict['L'+str(norm)] = norm_mat

    return norm_dict


# bootstrapping data for OLS for sleep consolidation data
def boot_ols(data_mat, interaction_degree, n_resamples, seed):
    rng = np.random.default_rng(seed)
    n_data = data_mat.shape[0] # we assume the columns are (x1, x2, y)

    X = data_mat[:,:-1]
    y = data_mat[:,-1]

    # extend input variables with bias and cross-terms
    kernel_transformer = PolynomialFeatures(interaction_degree, include_bias=True, interaction_only=True)
    X = kernel_transformer.fit_transform(X)
    n_coeff = X.shape[1]

    # output matrix with coefficients
    coeff_mat = np.zeros((n_resamples, n_coeff))

    # loop over all bootstrap resamples
    for i_resample in range(n_resamples):
        # draw the new bootstrap
        idx_resamples = rng.integers(low=0, high=n_data, size=n_data)

        # create a bootstrapped neuron with n_syn measurements over n_t time points
        X_resample = X[idx_resamples,:] #np.take_along_axis(X, idx_resamples, axis=0)
        y_resample = y[idx_resamples]

        # least-squares fit
        coeff_vec, _, _, _ = lstsq(X_resample, y_resample)

        # save coeffs
        coeff_mat[i_resample,:] = coeff_vec
    
    return coeff_mat


# bootstrapping residuals for ANCOVA for sleep consolidation data
def boot_residuals(data_mat, interaction_degree, n_resamples, seed):
    rng = np.random.default_rng(seed)
    n_data = data_mat.shape[0] # we assume the columns are (x1, x2, y)

    X = data_mat[:,:-1]
    y = data_mat[:,-1]

    # extend input variables with bias and cross-terms
    kernel_transformer = PolynomialFeatures(interaction_degree, include_bias=True, interaction_only=True)
    X = kernel_transformer.fit_transform(X)
    n_coeff = X.shape[1]

    # output matrix with coefficients
    coeff_mat = np.zeros((n_resamples, n_coeff))

    # compute residuals
    coeff_vec, _, _, _ = lstsq(X, y)
    coeff_mat[0,:] = coeff_vec
    y_hat = np.matmul(X, coeff_vec)
    res = y - y_hat

    # loop over all bootstrap resamples
    for i_resample in range(1,n_resamples):
        # draw the new bootstrap
        idx_resamples = rng.integers(low=0, high=n_data, size=n_data)

        # create a bootstrapped neuron with n_syn measurements over n_t time points
        res_resample = res[idx_resamples]
        y_new = y_hat + res_resample

        # least-squares fit
        coeff_vec, _, _, _ = lstsq(X, y_new)

        # save coeffs
        coeff_mat[i_resample,:] = coeff_vec
    
    return coeff_mat


# bootstrapping for ANCOVA using GLM with Binomial and probit
def boot_glm(data_mat, interaction_degree, n_resamples, seed):
    rng = np.random.default_rng(seed)
    n_data = data_mat.shape[0] # we assume the columns are (x1, x2, y)

    X = data_mat[:,:-1]
    y = data_mat[:,-1]

    # extend input variables with bias and cross-terms
    kernel_transformer = PolynomialFeatures(interaction_degree, include_bias=True, interaction_only=True)
    X = kernel_transformer.fit_transform(X)
    n_coeff = X.shape[1]

    # output matrix with coefficients
    coeff_mat = np.zeros((n_resamples, n_coeff))

    # fit GLM
    link = sm.genmod.families.links.probit()
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial(link=link)).fit()
    coeff_mat[0,:] = glm_binom.params

    # loop over all bootstrap resamples
    for i_resample in range(1,n_resamples):
        # draw the new bootstrap
        idx_resamples = rng.integers(low=0, high=n_data, size=n_data)

        # create a bootstrapped neuron with n_syn measurements over n_t time points
        X_resample = X[idx_resamples,:] #np.take_along_axis(X, idx_resamples, axis=0)
        y_resample = y[idx_resamples]

        # fit GLM
        glm_binom = sm.GLM(y_resample, X_resample, family=sm.families.Binomial(link=link)).fit()

        # save coeffs
        coeff_mat[i_resample,:] = glm_binom.params
    
    return coeff_mat



# bootstrapped power law for sleep consolidation data
def boot_power(data_mat, n_resamples, seed):
    rng = np.random.default_rng(seed)
    n_data = data_mat.shape[0] # we assume the columns are (x1, x2, y)
    coeff_mat = np.zeros((n_resamples, 3))

    # prepare data matrix for fitting
    x = data_mat[:,0]
    y = data_mat[:,1]

    bounds = ([-np.inf, -1., -np.inf], [np.inf, 1, np.inf])
    guess = [1, -0.5, -1]
    
    # loop over all bootstrap resamples
    for i_resample in range(n_resamples):
        if i_resample%50==0:
            print(f'resample {(i_resample):3}')

        # draw the new bootstrap
        idx_resamples = rng.integers(low=0, high=n_data, size=n_data)

        # create a bootstrapped neuron with n_syn measurements over n_t time points
        x_resample = x[idx_resamples] #np.take_along_axis(X, idx_resamples, axis=0)
        y_resample = y[idx_resamples]

        # curve-fit
        coeff_vec, _ = curve_fit(f=f2, jac=J2, xdata=x_resample, ydata=y_resample, p0=guess, bounds=bounds, method='trf', max_nfev=n_data*1000)
        # regr = linregress(x_resample, y_resample)

        # save coeffs
        coeff_mat[i_resample,:] = coeff_vec
        # coeff_mat[i_resample,:] = [regr.intercept, regr.slope]
    
    return coeff_mat


# power law
def f2(x, a, b, c):
    return a*x**b + c

# Jacobian of power law
def J2(x, a, b, c):
    return np.array([x**b, a*x**(b)*np.log(x), x**0]).T








