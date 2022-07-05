import sys
from .woe import WOE_Transform


def get_woe_(df, features, label_cols, method='tree', 
            num_features=None, num_bin_start=40, 
            min_samples_leaf=500, min_iv=0.01, display=0, 
            woe_fit_params={}):
    """ WOE wrapper for rdsutils.woe 

    For detailed usage or customization, checkout the original repo
    """
    woe = WOE_Transform(method=method, 
                        num_bin_start=num_bin_start,
                        min_samples_leaf=min_samples_leaf,
                        min_iv=min_iv)

    if len(label_cols) > 1 and isinstance(label_cols, list):
        print("Currently does not support multiple labels")
        sys.exit(1)
    label = label_cols[0] if isinstance(label_cols, list)\
                          else label_cols

    if num_features is None:
        print("producing WOE for all columns with numerical dtype")
        data = df[features]._get_numeric_data()
    else:
        data = df[num_features]

    woe.fit(data, df[label].astype(int), 
            display=display, **woe_fit_params)

    print("WOE fitted")
    return woe