import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from auxiliary import process_inputs_for_metrics

'''
Functions to compute and plot the univariate and multivariate versions of the
Giacomini-White (GW) test for Conditional Predictive Ability
'''

def GW(p_real, p_pred_1, p_pred_2, norm=1, version='univariate'):
    """Perform the one-sided GW test
    
    The test compares the Conditional Predictive Accuracy of two forecasts
    ``p_pred_1`` and ``p_pred_2``. The null H0 is that the CPA of errors ``p_pred_1``
    is higher (better) or equal to the errors of ``p_pred_2`` vs. the alternative H1
    that the CPA of ``p_pred_2`` is higher. Rejecting H0 means that the forecasts
    ``p_pred_2`` are significantly more accurate than forecasts ``p_pred_1``.
    (Note that this is an informal definition. For a formal one we refer 
    `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_)

    Parameters
    ----------
    p_real : numpy.ndarray
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the real market
        prices
    p_pred_1 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the first forecast
    p_pred_2 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the second forecast
    norm : int, optional
        Norm used to compute the loss differential series. At the moment, this value must either
        be 1 (for the norm-1) or 2 (for the norm-2).
    version : str, optional
        Version of the test as defined in
        `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_. It can have two values:
        ``'univariate'`` or ``'multivariate'``
    Returns
    -------
    float, numpy.ndarray
        The p-value after performing the test. It is a float in the case of the multivariate test
        and a numpy array with a p-value per hour for the univariate test

    """
    # Checking that all time series have the same shape
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the errors of each forecast
    loss1 = p_real - p_pred_1
    loss2 = p_real - p_pred_2
    tau = 1 # Test is only implemented for a single-step forecasts
    if norm == 1:
        d = np.abs(loss1) - np.abs(loss2)
    else:
        d = loss1**2 - loss2**2
    TT = np.max(d.shape)

    # Conditional Predictive Ability test
    if version == 'univariate':
        GWstat = np.inf * np.ones((np.min(d.shape), ))
        for h in range(24):
            instruments = np.stack([np.ones_like(d[:-tau, h]), d[:-tau, h]])
            dh = d[tau:, h]
            T = TT - tau
            
            instruments = np.array(instruments, ndmin=2)

            reg = np.ones_like(instruments) * -999
            for jj in range(instruments.shape[0]):
                reg[jj, :] = instruments[jj, :] * dh
        
            if tau == 1:
                betas = np.linalg.lstsq(reg.T, np.ones(T), rcond=None)[0]
                err = np.ones((T, 1)) - np.dot(reg.T, betas)
                r2 = 1 - np.mean(err**2)
                GWstat[h] = T * r2
            else:
                raise NotImplementedError('Only one step forecasts are implemented')

    elif version == 'multivariate':
        d = d.mean(axis=1)
        instruments = np.stack([np.ones_like(d[:-tau]), d[:-tau]])
        d = d[tau:]
        T = TT - tau
        
        instruments = np.array(instruments, ndmin=2)

        reg = np.ones_like(instruments) * -999
        for jj in range(instruments.shape[0]):
            reg[jj, :] = instruments[jj, :] * d
    
        if tau == 1:
            betas = np.linalg.lstsq(reg.T, np.ones(T), rcond=None)[0]
            err = np.ones((T, 1)) - np.dot(reg.T, betas)
            r2 = 1 - np.mean(err**2)
            GWstat = T * r2
        else:
            raise NotImplementedError('Only one step forecasts are implemented')
    
    GWstat *= np.sign(np.mean(d, axis=0))
    
    q = reg.shape[0]
    pval = 1 - scipy.stats.chi2.cdf(GWstat, q)
    return pval

def plot_multivariate_GW_test(real_price, forecasts, norm=1, title='GW test', savefig=False, path=''):
    """Plotting the results of comparing forecasts using the multivariate GW test. 
    
    The resulting plot is a heat map in a chessboard shape. It represents the p-value
    of the null hypothesis of the forecast in the y-axis being significantly more
    accurate than the forecast in the x-axis. In other words, p-values close to 0
    represent cases where the forecast in the x-axis is significantly more accurate
    than the forecast in the y-axis.
    
    Parameters
    ----------
    real_price : pandas.DataFrame
        Dataframe that contains the real prices
    forecasts : TYPE
        Dataframe that contains the forecasts of different models. The column names are the 
        forecast/model names. The number of datapoints should equal the number of datapoints
        in ``real_price``.
    norm : int, optional
        Norm used to compute the loss differential series. At the moment, this value must either
        be 1 (for the norm-1) or 2 (for the norm-2).
    title : str, optional
        Title of the generated plot
    savefig : bool, optional
        Boolean that selects whether the figure should be saved in the current folder
    path : str, optional
        Path to save the figure. Only necessary when `savefig=True`
    
    """

    # Computing the multivariate GW test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = GW(p_real=real_price.values.reshape(-1, 24), 
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1, 24), 
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1, 24), 
                                                  norm=norm, version='multivariate')

    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts.columns)), forecasts.columns, rotation=90.)
    plt.yticks(range(len(forecasts.columns)), forecasts.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    plt.title('(b) '+title)
    plt.tight_layout()
    plt.gca().set_xticks([x-0.5 for x in range(0, len(forecasts.columns))], minor=True)
    plt.gca().set_yticks([y-0.5 for y in range(0, len(forecasts.columns))], minor=True)
    plt.grid(which = 'minor',color='w', linestyle='-', linewidth=0.5)
    plt.tick_params(which='minor', bottom=False, left=False)


    if savefig:
        plt.savefig(f'{path}{title}.png', dpi=300)
        plt.savefig(f'{path}{title}.eps')
    plt.show()

"""
Functions to compute and plot the univariate and multivariate versions of the Diebold-Mariano (DM) test.
"""

def DM(p_real, p_pred_1, p_pred_2, norm=1, version='univariate'):
    """Function that performs the one-sided DM test in the contex of electricity price forecasting
    
    The test compares whether there is a difference in predictive accuracy between two forecast 
    ``p_pred_1`` and ``p_pred_2``. Particularly, the one-sided DM test evaluates the null hypothesis H0 
    of the forecasting errors of  ``p_pred_2`` being larger (worse) than the forecasting
    errors ``p_pred_1`` vs the alternative hypothesis H1 of the errors of ``p_pred_2`` being smaller (better).
    Hence, rejecting H0 means that the forecast ``p_pred_2`` is significantly more accurate
    that forecast ``p_pred_1``. (Note that this is an informal definition. For a formal one we refer to 
    `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_)

    Two versions of the test are possible:

        1. A univariate version with as many independent tests performed as prices per day, i.e. 24
        tests in most day-ahead electricity markets.

        2. A multivariate with the test performed jointly for all hours using the multivariate 
        loss differential series (see this 
        `article <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_ for details.

    
    Parameters
    ----------
    p_real : numpy.ndarray
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the real market
        prices
    p_pred_1 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the first forecast
    p_pred_2 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the second forecast
    norm : int, optional
        Norm used to compute the loss differential series. At the moment, this value must either
        be 1 (for the norm-1) or 2 (for the norm-2).
    version : str, optional
        Version of the test as defined in 
        `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_. It can have two values:
        ``'univariate`` or ``'multivariate``      
    Returns
    -------
    float, numpy.ndarray
        The p-value after performing the test. It is a float in the case of the multivariate test
        and a numpy array with a p-value per hour for the univariate test

    """

    # Checking that all time series have the same shape
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the errors of each forecast
    errors_pred_1 = p_real - p_pred_1
    errors_pred_2 = p_real - p_pred_2

    # Computing the test statistic
    if version == 'univariate':

        # Computing the loss differential series for the univariate test
        if norm == 1:
            d = np.abs(errors_pred_1) - np.abs(errors_pred_2)
        if norm == 2:
            d = errors_pred_1**2 - errors_pred_2**2

        # Computing the loss differential size
        N = d.shape[0]

        # Computing the test statistic
        mean_d = np.mean(d, axis=0)
        var_d = np.var(d, ddof=0, axis=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    elif version == 'multivariate':

        # Computing the loss differential series for the multivariate test
        if norm == 1:
            d = np.mean(np.abs(errors_pred_1), axis=1) - np.mean(np.abs(errors_pred_2), axis=1)
        if norm == 2:
            d = np.mean(errors_pred_1**2, axis=1) - np.mean(errors_pred_2**2, axis=1)

        # Computing the loss differential size
        N = d.size

        # Computing the test statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)
        
    p_value = 1 - stats.norm.cdf(DM_stat)

    return p_value

def plot_multivariate_DM_test(real_price, forecasts, norm=1, title='DM test', savefig=False, path=''):
    """Plotting the results of comparing forecasts using the multivariate DM test. 
    
    The resulting plot is a heat map in a chessboard shape. It represents the p-value
    of the null hypothesis of the forecast in the y-axis being significantly more
    accurate than the forecast in the x-axis. In other words, p-values close to 0
    represent cases where the forecast in the x-axis is significantly more accurate
    than the forecast in the y-axis.
    
    Parameters
    ----------
    real_price : pandas.DataFrame
        Dataframe that contains the real prices
    forecasts : TYPE
        Dataframe that contains the forecasts of different models. The column names are the 
        forecast/model names. The number of datapoints should equal the number of datapoints
        in ``real_price``.
    norm : int, optional
        Norm used to compute the loss differential series. At the moment, this value must either
        be 1 (for the norm-1) or 2 (for the norm-2).
    title : str, optional
        Title of the generated plot
    savefig : bool, optional
        Boolean that selects whether the figure should be saved in the current folder
    path : str, optional
        Path to save the figure. Only necessary when `savefig=True`
    
    """

    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = DM(p_real=real_price.values.reshape(-1, 24), 
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1, 24), 
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1, 24), 
                                                  norm=norm, version='multivariate')

    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts.columns)), forecasts.columns, rotation=90.)
    plt.yticks(range(len(forecasts.columns)), forecasts.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    plt.title('(a) '+title)
    plt.tight_layout()
    plt.gca().set_xticks([x-0.5 for x in range(0, len(forecasts.columns))], minor=True)
    plt.gca().set_yticks([y-0.5 for y in range(0, len(forecasts.columns))], minor=True)
    plt.grid(which = 'minor',color='w', linestyle='-', linewidth=0.5)
    plt.tick_params(which='minor', bottom=False, left=False)


    if savefig:
        plt.savefig(f'{path}{title}.png', dpi=300)
        plt.savefig(f'{path}{title}.eps')

    plt.show()
