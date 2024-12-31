# %%

import gc
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')
pd.set_option('display.max_columns', 30)

# %%


def read_data(filename: str, dir: str = 'Data'):
    """
    read_data: Reads a parquet file from the data directory.

    Arguments:
        filename: Must be a .parquet file.

    Keyword Arguments:
        dir: (default: {'Data'})

        If the data is in a subdiretory of the working
        directory, dir is the name of the subdirectory.

        If the data is within the same directory, dir should
        be an empty string.

    Returns:
        The data stored in the .parquet file in pandas
        dataframe type.
    """
    file = os.path.join(os.getcwd(), dir, filename)
    print(file)
    df = pd.read_parquet(file, engine='pyarrow')
    return df


def base36decode(number):
    """
    base36decode: Converts the order_number in the errands df from
    base-36 format to base-10, so that it can be cross-referenced
    with the orders df.

    Arguments:
        number: the order_number column with data in base-36 format
        from the errands df.

    Returns:
        The input data converted to base-10.
    """
    return int(number, 36)


def preprocess_data(errands_df, orders_df):
    """
    preprocess_data: Calls read_data to the errands.parquet and
    orders.parquet files and receives their respective dataframes.

    Converts date columns in both dataframes to datetime format.

    Calls base36decode to convert the order_number column in
    errands_df to base-10 and adds a column named order_id
    with the converted values to df.

    Merges the errands_df with orders_df based on the order_id
    column, creating a new df named errands_in_orders_df.

    Arguments:
        errands_df -- pandas dataframe type from errands.parquet
        received as output from the read_data function.
        orders_df -- pandas dataframe type from orders.parquet
        received as output from the read_data function.

    Returns:
        Processed version of the input data.
    """
    orders_df = orders_df.rename(str.lower, axis='columns')
    errands_df = errands_df.rename(columns={'created': 'errand_created_at'})

    orders_df['order_created_at'] = pd.to_datetime(orders_df['order_created_at'])
    errands_df['errand_created_at'] = pd.to_datetime(errands_df['errand_created_at'])

    errands_df['order_id'] = errands_df['order_number'].apply(base36decode)

    errands_in_orders_df = errands_df.merge(orders_df, how='inner', on='order_id')

    return (orders_df, errands_df, errands_in_orders_df)


[orders, errands, errands_in_orders] = preprocess_data(
    read_data('errands.parquet'), read_data('orders.parquet')
)


gc.collect()

# %%


def find_test_errands(errands_df):
    """
    Create a new dataframe only with errands named as 'is_test_errand_subset'.
    Drop errands flagged as test from input dataframe (errands_in_orders), and drop
    the 'is_test'  column from it.

    Parameters
    ----------
    errands_df : pandas dataframe
        errands_in_orders.

    Returns
    -------
    is_test_errand_subset : pandas dataframe
        Dataframe containing only test errands.
    test_errands_removed : pandas dataframe
        The input dataframe without order ids from test errands and the 'is_test'
        columns dropped.

    """
    is_test_errand_subset = errands_df.loc[
        (errands_df['is_test_errand'] == True)
    ].reset_index(drop=True)

    is_test_errand_subset = is_test_errand_subset.drop(columns=['is_test_errand'])

    test_errands_removed = errands_df.loc[
        (errands_df['is_test_errand'] == False)
    ].reset_index(drop=True)

    test_errands_removed = test_errands_removed.drop(columns=['is_test_errand'])
    return is_test_errand_subset, test_errands_removed


test_errands, errands_in_orders = find_test_errands(errands_in_orders)


# %%
# Mapps order IDs from the errands_in_orders dataset into the
# orders dataset and creating a column of booleans indicating
# orders that resulted in errands

unique_ids = errands_in_orders['order_id'].unique()
set_unique_ids = set(unique_ids)
orders['has_errands'] = orders['order_id'].map(
    lambda order_id: order_id in set_unique_ids
)

# %%
# Renaming ordering device named as "iPod" to "Unknow"

orders['device'] = orders['device'].str.replace(
    'iPod', 'Unknown', case=False, regex=False
)

# %%


def errands_counts_per_order(df, column: str):
    """
    Calculate the number of errands per order id and plot a bar plot with about
    90% of the orders with errands.

    Parameters
    ----------
    df : pandas dataframe
        The orders dataframe.
    column : str
        The order_id columns in the orders dataframe.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figures axes to plot it outside the function.
    counts_per_order : pandas series
        The value count result for the order_id column.
    percentage : pandas series
        The proportion of errands count in counts_per_order multiplied by 100.

    """
    counts_per_order = (
        df[column].value_counts(normalize=False).sort_values(ascending=False)
    )
    percentage = (
        counts_per_order.value_counts(normalize=True).sort_values(ascending=False) * 100
    )
    quantile_lim = percentage.quantile(0.90)
    print(quantile_lim)
    sns.set_theme()
    fig = plt.figure()
    # sns.set_context('talk', font_scale=1.0)
    sns.set_style(
        "darkgrid",
        {
            'axes.labelcolor': '.25',
            'xtick.color': '.25',
            'ytick.color': '.25',
            'text.color': '.25',
            'font.family': ['sans-serif'],
            'font.sans-serif': ['Inter', 'Liberation Sans', 'sans-serif'],
            'patch.force_edgecolor': False,
        },
    )
    ax = sns.barplot(percentage[percentage.ge(quantile_lim)])
    ax.set_xlabel('Number of errands')
    ax.set_ylabel('Orders with errands [%]')
    # ax.set_yscale('log')

    outliers = percentage[percentage.lt(quantile_lim)]
    print(outliers)

    return (fig, counts_per_order, percentage)


[figure, errands_count, errands_percentage] = errands_counts_per_order(
    errands_in_orders, 'order_id'
)


figure.savefig('errands_count_per_order.png', dpi=300, bbox_inches='tight')

# %%
# Errands in orders quantification

errands_count_average = errands_count.agg(['std', 'mean', 'max', 'min','median'])
print(errands_count_average)

bulk_errands_total = sum(errands_percentage.ge(errands_percentage.quantile(0.9)))
outlier_errands_total = sum(errands_percentage.lt(errands_percentage.quantile(0.9)))
# ~90% of orders have errands count below 12

orders_with_below_twelve_errands = len(
    errands_count.loc[errands_count.le(bulk_errands_total)]
)

orders_with_above_eleven_errands = len(
    errands_count.loc[errands_count.gt(bulk_errands_total)]
)

total_orders_with_errands = (
    orders_with_below_twelve_errands + orders_with_above_eleven_errands
)

# %%
# Adding a column with errands count by order ID in the orders df
orders_with_errands = orders.merge(errands_count, how='inner', on='order_id')
# len(orders_with_errands) agrees with total_orders_with_errands obtained in the cell above

# Adding a column with errands count by order ID in the errands df
errands_in_orders = errands_in_orders.merge(errands_count, how='inner', on='order_id')


# %%
# Removing order IDs with more than 11 errands

errands_in_order_subset = errands_in_orders[(errands_in_orders['count'] < 12)]
errands_in_order_subset = errands_in_order_subset.drop(
    ['errand_id', 'pnr', 'client_entry_type', 'origin_country', 'destination_country'],
    axis=1,
)

# %%


def get_errand_stats(df, size: int = 6):
    """
    Calculate basic statistics for the top values count for the listed columns.

    Parameters
    ----------
    df : pandas dataframe
         errands_in_order_subset.

    size : int, optional
        The number of top values count to keep.
        The default is 6.

    Returns
    -------
    stats : dictionary
        A dictionary with the calculated values.
    """
    discrete_value_columns = [
        'errand_type',
        'errand_action',
        'errand_category',
        'errand_channel',
        'booking_system',
        'site_country',
        'brand',
        'partner',
        'customer_group_type',
        'device',
        'currency',
        'revenue',
        'cancel_reason',
        'change_reason',
        'booking_system_source_type',
    ]
    stats = {
        column: df[column]
        .value_counts(normalize=True)
        .sort_values(ascending=False)
        .head(size)
        for column in discrete_value_columns
    }
    return stats


global_stats = get_errand_stats(errands_in_order_subset)


errands_aspects = errands_in_order_subset.groupby('errand_channel')[['errand_category','errand_type','errand_action']].value_counts().sort_values(ascending=False)
errands_aspects = errands_aspects.to_frame()
# %%


def plot_heat_map(df, column_a: str, column_b: str, plot: bool, top: int = 6):
    """
    Use the pandas crosstab function to calculate a matrix with the percentage of
    orders with errands for the top values count in column_a and, if chosen, plots
    the heat map using seaborn.

    Parameters
    ----------
    df : pandas dataframe
        errands_in_orders_subset.
    column_a : str
        A column with categorical data in df to make the vertical axes of the heat map.
    column_b : str
        A column with numerical data, normally the 'count' coulumn.
    plot : bool
        If the function should plot the heat map or not.
    top : int, optional
        The number of top values count to keep in column_a. The default is 6.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure axes to save the plot outside the function.
    heat_map : pandas dataframe
        Data frame with the crosstab results.

    """
    top_indexes = (
        df[column_a].value_counts(ascending=False).head(top)
    )
    heat_map = pd.crosstab(df[column_a], df[column_b], normalize='all') * 100
    heat_map = heat_map.loc[top_indexes.index]

    if plot:
        sns.set_theme()
        fig = plt.figure(figsize=(15, 6.5), layout='constrained')
        sns.set_context("talk", font_scale=1.15)
        sns.set_style(
            "darkgrid",
            {
                'axes.labelcolor': '.25',
                'xtick.color': '.25',
                'ytick.color': '.25',
                'text.color': '.25',
                'font.family': ['sans-serif'],
                'font.sans-serif': ['Inter', 'Liberation Sans', 'sans-serif'],
                'patch.force_edgecolor': False,
            },
        )
        ax = sns.heatmap(
            heat_map,
            annot=True,
            linewidth=0.5,
            cmap=sns.cubehelix_palette(as_cmap=True),
            cbar=False,
        )
        ax.set(
            xlabel='Number of errands per order ID',
            ylabel=''
            # f'{column_a.replace('_', ' ').title()}',
        )
        ax.set_title('Color: Percentage of errands normalized by all values')
        ax.tick_params('y', labelrotation=0)
        return fig, heat_map
    else:
        return None, heat_map


figure, errands_count_map = plot_heat_map(
    errands_in_order_subset, 'errand_channel', 'count', True, 5
)
# figure.savefig('change_reason_by_errand_count.png', dpi=300, bbox_inches='tight')
# errands_count_map

gc.collect()
