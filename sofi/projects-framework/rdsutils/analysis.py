import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix 


def get_misclassified_high_risk_rows(df, risk_col='transaction_amount', pred_col='pred'):
    """
    Get transactions that is:
    - sorted by high risk desc
    - misclassified 
    
    Parameters
    ----------
    df: dataframe
    
    risk_col: str or list 
        column of data used to quantify risk, will be sorted in desc order
    
    pred_col: str
        model prediction, sorted in ascending order -> indicate misclassification
    
    Returns
    -------
    df: dataframe
        sorted df
    """
    df = df.copy()
    if not isinstance(risk_col, list):
        risk_col = [risk_col]
    
    df['_tmp'] = 1-df[pred_col] 
    df = df.sort_values(by=risk_col+['_tmp'], ascending=False)
    df.drop('_tmp', inplace=True, axis=1)
    return df


def build_summary(target, pred, percentiles=np.linspace(90, 100, 10, endpoint=False)):
    """ build model behaviors by thresholds 

    Designed for Fraud models, but works for all binary classifications.
    
    Parameter
    ---------
    target: list or np.array
        sequence of true targets
        
    pred: list or np.array
        sequence of model predicted probabilities
        
    percentiles: list or np.array
        sequence of percentiles, default: np.linspace(90,100,10,endpoint=False)
    
    Return
    ------
    df: dataframe
        Table with descriptive stats.
    """
    df = []
    target = pd.Series(np.array(target) == 1)
    pred = pd.Series(pred)
    for thresh, pctl in [(np.percentile(pred, pctl), pctl) for pctl in percentiles]:
        pred_tmp = pred >= thresh
        rep = classification_report(y_true=target, y_pred=pred_tmp, output_dict=True)
        conf = confusion_matrix(y_true=target, y_pred=pred_tmp)
        df.append([pctl, thresh, (1 - rep['True']['precision']) * 100, rep['True']['recall'] * 100,
                  sum(conf[:, 1]), conf[1][1], conf[0][1], conf[1][0]])
    return pd.DataFrame(df, columns=['Percentile', 'Threshold', 'False Positive Rate (%)', 
                                     'Fraud Capture Rate (%)', '#Above Threshold', 
                                     '#Fraudulent Above Threshold', '#Good Above Threshold',
                                     '#Fraudulent Below Threshold'])


