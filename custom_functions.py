# Functions required to run the Airbnb_Analysis Notebook
import numpy
import statistics
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Function to calculate proportion of all unique values in a column
def get_prop(df, col):
    '''
    INPUT:
    df - the pandas dataframe whose column has to be plotted
    col - the column name to be plotted
    
    OUTPUT:
    prop_df - pandas series with each unique value found on df[col]
              and its respective proportion
    '''
    
    prop_df = df[col].value_counts().sort_values()/(df.shape[0]-sum(df[col].isnull()))
    return prop_df

# Function to delete a column where all values are equal
def single_value_drop(df):
    '''
    INPUT:
    df - the pandas dataframe whose columns ought to be checked
    
    OUTPUT:
    clean_df - pandas dataframe without single-value columns
    '''
    
    clean_df = df
    
    for col in df.columns:
        if len(df[col].unique()) == 1:
            clean_df = clean_df.drop(col, axis = 1)
            print('The following column was deleted: {}'. format(col))
            
    return clean_df

# Function to clean and convert numbers saved as strings
def clean_price(x):
    '''
    INPUT:
    x - string to be cleaned '$*,*'
    
    OUTPUT:
    x - if x is already cleaned and saved as a float
    x_clean - x cleaned and saved as float
    '''
    try:
        x_clean =float(x.replace('$', '').replace(',', ''))
        return x_clean
    except:
        return x
    
 # Function to clean percentage values saved as strings
def clean_perc(x):
    '''
    INPUT:
    x - string to be cleaned '*%'
    
    OUTPUT:
    x - if x is already cleaned and saved as a float
    x_clean - x cleaned and saved as float
    '''
    try:
        return float(x.replace('%', ''))/100
    except:
        return x
    
# Creating function to compare two samples in a t-test
def two_tail_t_test(reg_df, super_df, col, alpha):
    
    '''
    INPUT:
    reg_df - the first pandas dataframe to compare in a t-test
    super_df - the second pandas dataframe to compare in a t-test
    col - the column name to be compared
    size - how big should the bins in your histogram be
    
    OUTPUT:
    Displays a double-histogram plot and the 25%, 50% and 75% percentiles 
    '''

    # T-test: Comparing Superhosts and regular hosts

    # Calculating regular host parameters
    sample_r = reg_df.shape[0]
    mean_sample_r = reg_df[col].mean()
    stdev_sample_r = statistics.stdev(reg_df[col])
    sum_squares_r = reg_df[col].var(ddof=1)*(sample_r-1)

    # Calculating superhost parameters
    sample_s = super_df.shape[0]
    mean_sample_s = super_df[col].mean()
    stdev_sample_s = statistics.stdev(super_df[col])
    sum_squares_s = super_df[col].var(ddof=1)*(sample_s-1)

    # Independent samples t-test 
    # Degrees of freedom:
    deg_f = sample_r + sample_s - 2

    # Pooling variances for a more precise result
    pool_var = (sum_squares_r + sum_squares_s)/deg_f

    # Corrected Standard Error
    st_error = numpy.sqrt((pool_var/sample_r) + (pool_var/sample_s))

    # Calculating t-statistics
    t = (mean_sample_s - mean_sample_r)/st_error

    # For the alpha value we find our t-critical value (two-tailed)
    t_crit = -stats.t.ppf(alpha/2, deg_f)

    return t, t_crit

# Function to plot a double histogram 
def hist_generator(reg_df, super_df, col, size, max_range_99=False, percentil=True, xmin=0):
    
    '''
    INPUT:
    reg_df - the first pandas dataframe whose column you want to plot
    super_df - the second pandas dataframe whose column you want to plot
    col - the column name you want to plot
    size - how big should the bins in your histogram be
    max_range_99 - if true, sets the x-axis maximum value to the 99% quartile.
                   Default is false to show all values.
    percentil - True (default) to plot percentile values and axis
    xmin - minimal x value on the plot. Zero as default.
    
    OUTPUT:
    Displays a double-histogram plot and the 25%, 50% and 75% percentiles 
    '''
    
    # Basis for the analysis
    super_price = super_df[col]
    reg_price = reg_df[col]
        
    if max_range_99 == True:
        # We'll show 99% of the data to exclude outliers
        max_range = int(max(reg_price.quantile(0.99), 
                    super_price.quantile(0.99)))
    else:
        # Max Range is the maximum value of the distributions
        max_range = max(int(super_price.max()) + 1,
                        int(reg_price.max()) + 1)

    # Defining bin size to optimize visualization
    bin_size = list(numpy.arange(xmin, max_range + 1, size))

    # Defining weights for each distribution
    # that is, the basis for the percentage - samples have different sizes!
    basis_super = numpy.ones(len(super_price)) / len(super_price)
    basis_reg = numpy.ones(len(reg_price)) / len(reg_price)

    # Plotting the histogram
    fig = plt.figure(figsize = (15,10))
    plt.hist([reg_price, super_price], weights = [basis_reg, basis_super], alpha=0.5, bins=bin_size)

    # Formatting percentages, legends and axis labels
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1));
    plt.legend(['Regular hosts', 'Superhosts']);
    plt.ylabel('Percentage of listings (%)');
    plt.xlabel(col.capitalize());
    plt.xticks(list(numpy.arange(xmin, max_range, size)));
    plt.title(col.capitalize() + ': Superhosts vs. Regular hosts' );
    plt.grid(False);

    # Adding 25%, 50% and 75% Percentiles
    if percentil == True:
        reg_quantile = reg_price.quantile([0.25,0.5,0.75])
        super_quantile = super_price.quantile([0.25,0.5,0.75])
        min_ylim, max_ylim = plt.ylim()
        delta = 0.01

        for idx, item in enumerate(reg_quantile):
            plt.axvline(item, color='C0', linestyle='dashed', linewidth=1)
            plt.text(item + 10, max_ylim - delta, '{}. Quartile: {:.2f}'.format(idx + 1, item), color ='C0')
            delta += 0.01

        for idx, item in enumerate(super_quantile):
            plt.axvline(item,  color='darkorange', linestyle='dashed', linewidth=1)
            plt.text(item + 10, max_ylim - delta, '{}. Quartile: {:.2f}'.format(idx + 1, item), color = 'darkorange')
            delta += 0.01
    
    plt.show()
    
# Function to plot a cummulative histogram
def cummulative_hist(reg_df, super_df, col, size, max_range_99=False, xmin=0):
    '''
    INPUT:
    reg_df - the first pandas dataframe whose column you want to plot
    super_df - the second pandas dataframe whose column you want to plot
    col - the column name you want to plot
    size - how big should the bins in your cummulative histogram be
    max_range_99 - if true, sets the x-axis maximum value to the 99% quartile.
                   Default is false to show all values.
    xmin - minimal x value on the plot. Zero as default.
    
    OUTPUT:
    Displays a cummulative histogram plot for both data frames' column.
    '''
    
    # Basis for the analysis
    reg_price = reg_df[col]
    super_price = super_df[col]

    if max_range_99 == True:
        # We'll show 99% of the data to exclude outliers
        max_range = int(max(reg_price.quantile(0.99), 
                    super_price.quantile(0.99)))
    else:
        # Max Range is the maximum value of the distributions
        max_range = max(int(super_price.max()) + 1,
                        int(reg_price.max()) + 1)
    
    # Defining bin size to optimize visualization
    bin_size = list(numpy.arange(xmin, max_range, size))

    # Plotting the histogram
    fig = plt.figure(figsize = (15,10))
    plt.hist([reg_price, super_price], density = True, 
             histtype='step', fill=False, cumulative=True, bins=bin_size, color = ['C0', 'darkorange'])

    # Formatting percentages, legends and axis labels
    plt.grid(True, alpha=0.5)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1));
    plt.legend(['Superhosts', 'Regular hosts']);
    plt.ylabel('Percentage of listings - Cummulative (%)');
    plt.xlabel(col.capitalize());
    plt.title(col.capitalize() +': Superhosts vs. Regular hosts');