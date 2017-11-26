import pandas as pd
import matplotlib.pyplot as plt
from z_test.z_test import z_test


# 4.
def find_mismatch(ab_cell, landing_page_cell):
    """Checks if A/B test treatment/control encoding is accurate
    Parameters
    ----------
    ab_cell: str
        Treatment/Control encoding
    landing_page_cell: str
        Page version
    Returns
    -------
    int: indicator of match (1) or mismatch (0)
    """

    if ab_cell == 'treatment' and landing_page_cell == 'new_page':
        return 0
    elif ab_cell == 'control' and landing_page_cell == 'old_page':
        return 0
    else:
        return 1


# 5.
def calc_conversion_rate(data, page_type):
    """Counts proportion of tatal visits resulting in a conversion
    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns 'converted' (1=yes, 0=no)
        and 'landing_page' with values "new_page" or "old_page"
    page_type: str ("new" or "old")
        corresponding to the "new_page"/"old_page"
    Returns
    -------
    float: proportion of total visits converted for input page_type
    """

    total_vis = data[data['landing_page'] == page_type + '_page']
    converted = total_vis[total_vis['converted'] == 1].shape[0]
    return float(converted) / total_vis.shape[0], total_vis.shape[0]


# 8.
def plot_pval(data):
    """plots p-value based on hourly testing of running A/B test
    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns 'hour' converted' and 'landing_page'
    Returns
    -------
    None: A plot is produced buy no axis object is returned
    """

    pval_lst = []
    datetime = data.ts.astype('datetime64[s]')
    hour = datetime.apply(lambda x: x.hour)
    data['hour'] = hour
    # Run the test as the data accumulates hourly
    for hr in hour.unique():
        hr_data = data[data['hour'] <= hr]
        # data for old landing page
        old = hr_data[hr_data['landing_page'] == 'old_page']['converted']
        old_nrow = old.shape[0]
        old_conversion = old.mean()
        # data for new landing page
        new = hr_data[hr_data['landing_page'] == 'new_page']['converted']
        new_nrow = new.shape[0]
        new_conversion = new.mean()
        # Run the z-test
        p_val = z_test(old_conversion, new_conversion,
                       old_nrow, new_nrow, effect_size=0.001,
                       two_tailed=True, alpha=.05)
        pval_lst.append(p_val[1])

    # Make the plot
    plt.plot(pval_lst, marker='o')
    plt.ylabel('p-value', fontweight='bold', fontsize=14)
    plt.xlabel('Hour in the day', fontweight='bold', fontsize=14)
    plt.axhline(0.05, linestyle='--', color='r')


# 9.
def read_country_and_merge(data, filename):
    """Reads in csv file with 'country' and 'user_id' columns and merges with data
    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns 'user_id'
    filename: str
        Path to csv file with columns named 'user_id' and 'country'
    Returns
    -------
    Pandas DataFrame: orginal input 'data' merged with the csv file at 'filename'
    """
    country = pd.read_csv(filename)
    merged_df = pd.merge(data, country, left_on='user_id',
                         right_on='user_id', how='left')
    merged_df['country'] = merged_df['country'].map(str)
    return merged_df


def run_test(data):
    """Does an A/B test based on 'calc_conversion_rate' and 'z_test.z_test'
    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns 'converted' (1=yes, 0=no)
        and 'landing_page' with values "new_page" or "old_page"
    Returns
    -------
    tuple: p_val, old_conversion, new_conversion
        The p-value from an A/B test along with the A and B conversion rates
    """

    new_convert, new_nrow = calc_conversion_rate(data, 'new')
    old_conversion, new_nrow = calc_conversion_rate(data, 'old')
    #alpha needs to be reduced to account for 4 countries
    modified_alpha = 0.05 / 4
    p_val = z_test(old_conversion, new_conversion,
       old_nrow, new_nrow, effect_size = 0.001, alpha = modified_alpha)[1]

    return p_val, old_conversion, new_conversion


def run_country_test(data):
    """Runs a separate A/B test based on 'run_test' for each country
    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns
        'converted',  'landing_page', and 'country'
    Returns
    -------
    None: Conversion rate difference (new-old) and
        test p-value are printed for each country
    """
    results = {}
    for country in data['country'].unique():
        country_df = data[data['country'] == country]
        p_val, old_conversion, new_conversion = run_test(country_df)
        results[country] = [p_val, new_conversion - old_conversion]

    for country, lst in results.iteritems():
        p_val, conversion_diff = lst
        print '%s | conversion increase: %s | p_val: %s' % (country, conversion_diff, p_val)


if __name__ == '__main__':


    data = pd.read_csv('data/experiment.csv')

    # 4.
    print "\nProblem #4 results:\n"
    # inconsistent, dirty data:
    print 'ab column counts:'
    print data['ab'].value_counts()
    print 'landing_page column counts:'
    print data['landing_page'].value_counts()

    print 'Dropping treatment / control and landing page mismatch...'
    # axis=1 means iterate the rows in the dataframe
    find_mismatch_on_row = lambda row: find_mismatch(row['ab'], row['landing_page'])
    data['mismatch'] = data.apply(find_mismatch_on_row, axis=1)

    mismatched = data[data['mismatch'] == 1]
    percent_mismatched = (len(mismatched) / (len(data['mismatch']) * 1.) * 100)
    print 'Percentage mismatched:', percent_mismatched

    data = data[data['mismatch'] == 0]

    duplicated = data.duplicated()
    percent_duplicated = (sum(duplicated) / (len(duplicated) * 1.) * 100)
    print 'Percentage duplicated:', percent_duplicated
    # no duplicates: good

    # 5.
    print "\nProblem #5 results:\n"
    # Get the parameters needed for z_test()
    old_conversion, old_nrow = calc_conversion_rate(data, 'old')
    new_conversion, new_nrow = calc_conversion_rate(data, 'new')

    print old_conversion, old_nrow, new_conversion, new_nrow

    z_test(old_conversion, new_conversion, old_nrow, new_nrow,
           effect_size=0.001, two_tailed=False, alpha=.05)

    # 8.
    print "\nProblem #8 results:\n"
    plot_pval(data)
    plt.show()

    # 9.
    print "\nProblem #9 results:\n"
    merged_df = read_country_and_merge(data, 'data/country.csv')
    run_country_test(merged_df)

    print ""
