import base64
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class TransactionsVisualizer:
    
    def __init__(self, html_path, account_df, 
                 transactions_df, bid_col, 
                 x_col='transaction_datetime', 
                 y_col='real_ending_balance',
                 target_col='target',
                 random_state=12345, 
                 scatter_filter=None ):
        """
        usage for the MCD model:
        - https://gitlab.com/sofiinc/data-science-risk/money-risk-models/-/blob/master/check-risk/notebooks/iter-0-20200920/6-pattern-investigation.ipynb
        A copy is stored in the rdsutils/examples folder
        
        @params html_path: str - where to save the .html file
        @params account_df: pd.DataFrame
            - account level df
            - each row represents an account, with id as the index
        @params transactions_df: pd.DataFrame
            - transactions level df
        @params bid_col: str
            - id column, default = business_account_number
        @params x_col: str
            - col to plot on the x-axis of the left plot
            - default: transaction_datetime
        @params y_col: str
            - col to plot on the y-axis of the left plot
            - default: real_ending_balance
        @params target_col: str
            - col that contains targets
        @params random_state: int
            - random seed
        @params scatter_filer: function
            - filter for transactions_df, what is left will be scatter plotted
            - e.g. account balance (line plot) vs. deposits (scatter plot)
                   use the function to filter out the non-deposits 
                   from the transactions_df
        """
        self.writer = HTMLPlotWriter(html_path)
        self.account_df = account_df
        self.transactions_df = transactions_df
        self.bid_col = bid_col
        self.x_col = x_col
        self.y_col = y_col
        self.target_col = target_col
        self.random_state = random_state
        self.scatter_filter = scatter_filter
        
    
    def run(self, n_samples=None, figsize=(18,6), 
            indeterminate_col=None):
        """
        
        """
        plt.ioff()
        df_ = self.account_df
        trans_df = self.transactions_df
        
        if n_samples is not None:
            df_ = df_.sample(n=n_samples, random_state=self.random_state)
        
        for bid, acct_row in tqdm(df_.iterrows()):
            acct_snapshot = acct_row.reset_index()
            acct_snapshot.columns = ['attr', 'type', 'values']
            
            # init plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # get transactions for that bid
            df = trans_df[trans_df[self.bid_col] == bid].copy()
            df.sort_values(by=[self.x_col], ascending=True, inplace=True)
            
            # plot transaction trajectory on the left
            plot_transactions_trajectory(df, target_col=self.target_col,
                                         fig=fig, ax=ax1, title=f'ID: {bid} - {self.y_col}',
                                         indeterminate_col=indeterminate_col, 
                                         scatter_filter=self.scatter_filter)
            
            # plot table on the right
            plot_dataframe(acct_snapshot, fig=fig, ax=ax2)
            self.writer.write(fig, message=f'ID: {bid}')
            plt.close()
        
        self.writer.close()


class HTMLPlotWriter:
    
    def __init__(self, html_path):
        self.html_path = html_path
        self.f_out = open(html_path, 'w+')
    
    def write(self, fig, message):
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html = f'<br>{message}<br>' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        self.f_out.write(html)
    
    def close(self):
        self.f_out.close()
        
    
def plot_dataframe(df, fig=None, ax=None, figsize=None,
                   fontsize=14, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns,
                     loc='center', **kwargs)
    
    fig.tight_layout()
    table.set_fontsize(fontsize)
    return fig, ax


def plot_transactions_trajectory(df, x_col='transaction_datetime',
                                 y_col='real_ending_balance', target_col='target',
                                 fig=None, ax=None, title=None, palette=None, 
                                 figsize=None, indeterminate_col=None, 
                                 scatter_filter=None, **kwargs):
    """
    plot account trajectory of df on ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if palette is None:
        palette = sns.color_palette(n_colors=df.business_account_number.nunique()+3)
        
    sns.lineplot(y=y_col, x=x_col, 
                 data=df, alpha=0.8, color=palette[2], label=y_col, ax=ax)
    
    ## TODO for scatter, only plot the write type
    if scatter_filter is not None:
        df = scatter_filter(df)
    
    try:
        if indeterminate_col is None:
            # label non target deposits
            sns.scatterplot(y=y_col, x=x_col, 
                            data=df[~df[target_col]], label='not target', ax=ax, 
                            color=palette[1], alpha=1)
            # label target deposits
            sns.scatterplot(y=y_col, x=x_col, 
                            data=df[df[target_col]], label='target', ax=ax, 
                            color=palette[0], alpha=1)
        else:
            # label non target deposits
            sns.scatterplot(y=y_col, x=x_col, 
                            data=df[(~df[target_col]) & (~df[indeterminate_col])], 
                            label='not target', ax=ax, 
                            color=palette[1], alpha=1)
            # label target deposits
            sns.scatterplot(y=y_col, x=x_col, 
                            data=df[(df[target_col]) & (~df[indeterminate_col])], 
                            label='target', ax=ax, 
                            color=palette[0], alpha=1)
             # label indeterminate deposits
            sns.scatterplot(y=y_col, x=x_col, 
                            data=df[df[indeterminate_col]], 
                            label='indeterminate', ax=ax, 
                            color=palette[3], alpha=1)
    except:
        print('Error, look into this')
    
    
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    return fig, ax