import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns
plt.style.use('ggplot')

def score_gain(Y,Y_weight,X,decile=10,diretion=0):
    YY=Y.reset_index(drop=True)
    YY_weight=Y_weight.reset_index(drop=True)
    XX=X.reset_index(drop=True)
    
    bins=pd.qcut(XX,decile,retbins=False,labels=False,duplicates="drop")
    Y_bins=pd.DataFrame(bins,index=YY.index)
    Y_model=pd.DataFrame(XX,index=YY.index)
    Y_new=YY*YY_weight
        
    Y_final=pd.concat([Y_new,YY_weight,Y_model,Y_bins],axis=1) 
    Y_final.columns=['true','weight','score','bins']
    Y_final.loc[Y_final.score==0,'bins']=-99     
    s_min=round(Y_final.groupby('bins')['score'].min(),4)
    s_max=round(Y_final.groupby('bins')['score'].max(),4)
    s_count=Y_final.groupby('bins')['weight'].sum()
    s_pred=round(Y_final.groupby('bins')['score'].mean(),4)
    #s_rate=round(Y_final.groupby('bins')['true'].mean(),4)
    s_numtarget=round(Y_final.groupby('bins')['true'].sum(),1)
    s_rate=s_numtarget/s_count
    g_table=pd.concat([s_count,s_min,s_max,s_pred,s_rate,s_numtarget],axis=1)    
    g_table.columns=["#accts","min","max","pred_rate","actual_rate","target_num"]
        
    if diretion==1:
        g_table=g_table.sort_values(by="min",ascending=False)
    elif diretion==0:
        g_table=g_table.sort_values(by="min",ascending=True)
        
    g_table['nontarget_num']=g_table['#accts']-g_table['target_num']
        
    g_table['pct_cum_acct']=round(g_table['#accts'].cumsum()/g_table['#accts'].sum(),3)
    g_table['pct_cum_target']=round(g_table['target_num'].cumsum()/g_table['target_num'].sum(),3)
    g_table['pct_cum_nontarget']=round(g_table['nontarget_num'].cumsum()/g_table['nontarget_num'].sum(),3)
    g_table['reverse_target_num']=g_table.target_num.values[::-1]
    g_table['reverse_total_num']=g_table['#accts'].values[::-1]
    g_table['cum_bads']=round(g_table['reverse_target_num'].cumsum()/g_table['reverse_total_num'].cumsum(),3)
    g_table['cum_acct']=round(g_table['reverse_total_num'].cumsum()/g_table['reverse_total_num'].sum(),3)
    g_table['KS']=(g_table['pct_cum_target'] - g_table['pct_cum_nontarget'])*100
        
    g_table.reset_index(inplace=True,drop=True)
    g_table=g_table[["pct_cum_acct","#accts","min","max","target_num",'pct_cum_target','pct_cum_nontarget',"actual_rate",'KS','cum_bads','cum_acct']]
        
    return g_table
    




# for model prediction and becnhamrk   
def model_score_eval(Y_true,model,score=[],score2=[],Y_weight=[],decile=10, if_plot = 1):

    if if_plot == 1:
        fig = plt.figure(figsize=(16,14))
        #ax0 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(221)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

    
    if len(Y_weight)!=len(Y_true):
        Y_weight=pd.Series(np.ones(len(Y_true)))
        
    if len(model)==len(Y_true):
        auc_bm=roc_auc_score(Y_true,model,sample_weight=Y_weight)
        fpr, tpr, threshold = roc_curve(Y_true,model,sample_weight=Y_weight)
        roc_auc = auc(fpr, tpr)
        model_table=score_gain(Y_true,Y_weight,model,decile,1)
        return_auc = auc_bm
        return_ks = model_table['KS'].max()
        
        if if_plot == 1:
            print(model.name,"AUC: ",round(auc_bm,5), " KS: ", round(model_table['KS'].max(),3))

            model_table_t=model_table[['pct_cum_acct','pct_cum_target',
                                       'actual_rate','cum_bads','cum_acct']]
            model_table_t.columns=['pct_cum_acct',
                                   model.name+'_pct_cum_target',
                                   model.name+'_actual_rate',
                                   model.name+'_cum_bads',
                                   'cum_acct']
            table_total= model_table_t.copy()
            pos=list(range(len(table_total[model.name+'_actual_rate']))) 
            ax1.bar(pos,table_total[model.name+'_actual_rate'],0.25,label=model.name)
            ax3.plot(fpr, tpr,  label = '{} AUC = {}'.format(model.name,round(roc_auc,3)))
            ax4.plot(table_total['cum_acct'],
                     table_total[model.name+'_cum_bads'],
                     label=model.name)
            pct_cum_acct=table_total['pct_cum_acct']
            pct_cum_acct.at[-1]=0
            pct_cum_target=table_total[model.name+'_pct_cum_target']
            pct_cum_target.at[-1]=0
            ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=model.name)  
    else:
        print('model value lenghth does not match with Y_true')
    
    
    if len(score)==len(Y_true):
        auc_bm=1-roc_auc_score(Y_true,score,sample_weight=Y_weight)
        fpr, tpr, threshold = roc_curve(1-Y_true,score,sample_weight=Y_weight)
        roc_auc = auc(fpr, tpr)
        score_table=score_gain(Y_true,Y_weight,score,decile,0)

        return_auc = auc_bm
        return_ks = score_table['KS'].max()
        if if_plot == 1:
            print(score.name,"AUC: ",round(auc_bm,5), " KS: ", round(score_table['KS'].max(),3))

            score_table_t=score_table[['pct_cum_target','actual_rate','cum_bads']]
            score_table_t.columns=[score.name+'_pct_cum_target',score.name+'_actual_rate',score.name+'_cum_bads']
            table_total=pd.concat([table_total,score_table_t],axis=1)

            ax1.bar([p +0.25 for p in pos],table_total[score.name+'_actual_rate'],0.25,label=score.name)
            ax4.plot(table_total['cum_acct'],table_total[score.name+'_cum_bads'],label=score.name)
            pct_cum_acct=table_total['pct_cum_acct']
            pct_cum_acct.at[-1]=0
            pct_cum_target=table_total[score.name+'_pct_cum_target']
            pct_cum_target.at[-1]=0
            ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=score.name)


            ax3.plot(fpr, tpr,  label = '{} AUC = {}'.format(score.name,round(roc_auc,3)))
        
    
    
    if len(score2)==len(Y_true):
        auc_bm2=1-roc_auc_score(Y_true,score2,sample_weight=Y_weight)
        fpr, tpr, threshold = roc_curve(1-Y_true,score2,sample_weight=Y_weight)
        roc_auc = auc(fpr, tpr)
        score_table2=score_gain(Y_true,Y_weight,score2,decile,0)
        
        return_auc = auc_bm2
        return_ks = score_table2['KS'].max()
        if if_plot == 1:
            print(score2.name,"AUC: ",round(auc_bm2,5), " KS: ", round(score_table2['KS'].max(),3))

            score_table2_t=score_table2[['pct_cum_target','actual_rate','cum_bads']]
            score_table2_t.columns=[score2.name+'_pct_cum_target',score2.name+'_actual_rate',score2.name+'_cum_bads']

            table_total=pd.concat([table_total,score_table2_t],axis=1)
            ax1.bar([p + 0.5 for p in pos],table_total[score2.name+'_actual_rate'],0.25,label=score2.name)
            ax4.plot(table_total['cum_acct'],table_total[score2.name+'_cum_bads'],label=score2.name)
            pct_cum_acct=table_total['pct_cum_acct']
            pct_cum_acct.at[-1]=0
            pct_cum_target=table_total[score2.name+'_pct_cum_target']
            pct_cum_target.at[-1]=0
            ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=score2.name)
            ax3.plot(fpr, tpr,   label = '{} AUC = {}'.format(score2.name,round(roc_auc,3)))
    
    
    #s_pct_cum_acct=score_table['pct_cum_acct']
    #s_pct_cum_acct.at[-1]=0
    
    #s_pct_cum_target=score_table['pct_cum_target']
    #s_pct_cum_target.at[-1]=0
    
    #s_pct_cum_nontarget=score_table['pct_cum_nontarget']
    #s_pct_cum_nontarget.at[-1]=0
    
    
    #ax0.plot(s_pct_cum_acct.sort_index(),s_pct_cum_target.sort_index(),'--r',label="Target")
    #ax0.plot(s_pct_cum_acct.sort_index(),s_pct_cum_nontarget.sort_index(),'--b',label="Non-Target")
    
    if if_plot == 1:
        ax2.plot(pct_cum_acct.sort_index(),pct_cum_acct.sort_index(),':',label='baseline')

        print()
        print("Gains Table--Model:")
        model_table_print=model_table[['pct_cum_acct','#accts','min','max','target_num','pct_cum_target','pct_cum_nontarget','actual_rate','KS']].copy()

        model_table_print['pct_cum_acct']=model_table_print['pct_cum_acct'].map('{:,.1%}'.format)
        model_table_print['pct_cum_target']=model_table_print['pct_cum_target'].map('{:,.1%}'.format)
        model_table_print['pct_cum_nontarget']=model_table_print['pct_cum_nontarget'].map('{:,.1%}'.format)
        model_table_print['actual_rate']=model_table_print['actual_rate'].map('{:,.1%}'.format)

        display( model_table_print)
        print()
        print()

        #ax0.legend()
        #ax0.set_xlabel('% Total Population')
        #ax0.set_ylabel('% Cum Target')
        #ax0.set_title('Model KS')
        #ax0.set_ylim([0,1])
        #ax0.set_xlim([0,1])
        #ax0.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        #ax0.xaxis.set_major_formatter(PercentFormatter(xmax=1))



        ax1.legend()
        ax1.set_xlabel('Percentile')
        ax1.set_ylabel('Target Rate')
        ax1.set_title('Bad Rate by percentile')
        ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))


        ax2.legend()
        ax2.set_xlabel('% Total Population')
        ax2.set_ylabel('% Cum Target')
        ax2.set_title('Gains Table (Capture Rate)')
        ax2.set_ylim([0,1])
        ax2.set_xlim([0,1])
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1))




        print()
    #     fig, ax = plt.subplots(figsize=(12,8))
    #     ax.plot(model_table.index, model_table.pct_cum_target,color='mediumvioletred')
    # #     g=sns.lineplot(model_table_print.index, model_table_print.pct_cum_target, color='gray')
    #     for ixx, row in model_table.iterrows():
    #         plt.text(ixx,row.pct_cum_target*1.03,'{:,.0%}'.format(row.pct_cum_target), color='black', ha='center')

        ax3.legend(loc = 'lower right')
        ax3.plot([0, 1], [0, 1],'k--')
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        ax3.set_xlabel('True Positive Rate')
        ax3.set_ylabel('False Positive Rate')
        ax3.set_title('ROC Curve')

        ax4.legend()
        ax4.set_xlabel('Approval Rate')
        ax4.set_ylabel('Bad Rate')
        ax4.set_title('Efficient Fonterior')

        ax4.set_xlim([0,1])



        plt.show()
    
    return return_auc, return_ks
    
    
def ROC_curve(model, df_train, lable, features, weight, df_test1 = None, df_test2 = None):
    
    fpr_train, tpr_train, thresholds = roc_curve(df_train[lable], model.predict_proba(df_train[features])[:, 1], sample_weight = df_train[weight] )
    train_auc = roc_auc_score(df_train[lable], model.predict_proba(df_train[features])[:, 1], sample_weight = df_train[weight])
    
    plt.style.use('seaborn-white')
    plt.figure()
    
    plt.plot(fpr_train, tpr_train, 'k--')
    plt.plot(fpr_train, tpr_train, 
             color='darkorange',
             lw = 2,
             label='Training ROC (area = %0.3f)' % train_auc)

        
    if (df_test1 is not None):
    
        fpr_test1, tpr_test1, thresholds = roc_curve(df_test1[lable], model.predict_proba(df_test1[features])[:, 1], sample_weight = df_test1[weight] )
        test1_auc = roc_auc_score(df_test1[lable], model.predict_proba(df_test1[features])[:, 1], sample_weight = df_test1[weight])

        plt.plot(fpr_test1, tpr_test1, 'k--')
        plt.plot(fpr_test1, tpr_test1, 
                 color='darkblue',
                 lw = 2,
                 label='Testing 1 ROC (area = %0.3f)' % test1_auc)
        
    if (df_test1 is not None):
    
        fpr_test2, tpr_test2, thresholds = roc_curve(df_test2[lable], model.predict_proba(df_test2[features])[:, 1], sample_weight = df_test2[weight] )
        test2_auc = roc_auc_score(df_test2[lable], model.predict_proba(df_test2[features])[:, 1], sample_weight = df_test2[weight])

        plt.plot(fpr_test2, tpr_test2, 'k--')
        plt.plot(fpr_test2, tpr_test2, 
                 color='darkred',
                 lw = 2,
                 label='Testing 2 ROC (area = %0.3f)' % test2_auc)        
           
    plt.plot([0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    

def model_eval(model, df_train, lable, features, weight=None, df_test1=None, df_test2 = None):

    sns.set(rc={'figure.figsize':(12,8)})
    ROC_curve(model, df_train, lable, features, weight, df_test1, df_test2)
        
    df_train['pred'] = model.predict_proba(df_train[model.feature_name_])[:,1]
    _auc, _ks = model_score_eval(df_train[lable],df_train['pred'],Y_weight=df_train[weight],decile=20, if_plot = -1)
    results = pd.DataFrame(data = {'data':['training'], 'auc': [_auc], 'ks': [_ks]})

    
    if df_test1 is not None:
        df_test1['pred'] = model.predict_proba(df_test1[model.feature_name_])[:,1]
        _auc, _ks = model_score_eval(df_test1[lable],df_test1['pred'],Y_weight=df_test1[weight],decile=20, if_plot = -1)
        results = results.append({'data':'test 1', 'auc': _auc, 'ks': _ks}, ignore_index=True)

    if df_test2 is not None:
        df_test2['pred'] = model.predict_proba(df_test2[model.feature_name_])[:,1]
        _auc, _ks = model_score_eval(df_test2[lable],df_test2['pred'],Y_weight=df_test2[weight],decile=20, if_plot = -1)
        results = results.append({'data':'test 2', 'auc': _auc, 'ks': _ks}, ignore_index=True)

    return results
    
