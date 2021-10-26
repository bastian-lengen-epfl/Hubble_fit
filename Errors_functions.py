import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot

def errors_distribution(errors: pd.DataFrame, name: str, fig_dir: str = "./Figure"):
    # Create the figure directory
    err_dir = fig_dir+"/Errors"
    if not os.path.exists(err_dir):
        print("I will create the %s directory for you." %err_dir)
        os.mkdir(err_dir)

    # Histogram plot of the errors
    plt.hist(errors, 40, density=True)
    x = np.linspace(-1, 1, 100)
    mu, std, skew, kur = np.mean(errors), np.std(errors), stats.skew(errors), stats.kurtosis(errors)
    plt.plot(x, stats.norm.pdf(x, mu, std), marker='', ls='-')
    plt.savefig("%s/%s_histogram.jpg" %(err_dir, name), dpi=100)
    plt.close()

    # Can displays stats if wanted
    # print('mean = %f, std = %f, skw = %f, kur = %f' % (mu, std, skew, kur))

    #  QQ-plot
    qqplot(errors, line='s')
    plt.savefig("%s/%s_qqplot.jpg" %(err_dir, name), dpi=100)
    plt.close()


    # D’Agostino’s K^2 Test
    stat, p = stats.normaltest(errors)
    print('\nD’Agostino’s K^2 Test on the errors distribution: Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    return