import pandas as pd
from sklearn.metrics import roc_auc_score
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

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
    g_table['KS']=(g_table['pct_cum_target'] - g_table['pct_cum_nontarget'])*100
        
    g_table.reset_index(inplace=True,drop=True)
    g_table=g_table[["pct_cum_acct","#accts","min","max","target_num",'pct_cum_target','pct_cum_nontarget',"actual_rate",'KS']]
        
    return g_table
    



    
### for score only,
def score_eval(Y_true,score,score2=[],Y_weight=[],decile=10):
 
    fig = plt.figure(figsize=(12,10))
    #ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(221)
    
    if len(Y_weight)!=len(Y_true):
        Y_weight=pd.Series(np.ones(len(Y_true)))
        
    if len(score)==len(Y_true):
        auc_bm=1-roc_auc_score(Y_true,score,sample_weight=Y_weight)
        score_table=score_gain(Y_true,Y_weight,score,decile,0)
        print(score.name,"AUC: ",round(auc_bm,5), " KS: ", round(score_table['KS'].max(),3))
        
        score_table_t=score_table[['pct_cum_acct','pct_cum_target','actual_rate']]
        score_table_t.columns=['pct_cum_acct',score.name+'_pct_cum_target',score.name+'_actual_rate']
        table_total= score_table_t.copy()
        
        ax1.plot(table_total['pct_cum_acct'],table_total[score.name+'_actual_rate'],label=score.name)
        
        pct_cum_acct=table_total['pct_cum_acct']
        pct_cum_acct.at[-1]=0
        pct_cum_target=table_total[score.name+'_pct_cum_target']
        pct_cum_target.at[-1]=0
        ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=score.name)
        
    if len(score2)==len(Y_true):
        auc_bm2=1-roc_auc_score(Y_true,score2,sample_weight=Y_weight)
        score_table2=score_gain(Y_true,Y_weight,score2,decile,0)
        print(score2.name,"AUC: ",round(auc_bm2,5), " KS: ", round(score_table2['KS'].max(),3))
        
        score_table2_t=score_table2[['pct_cum_target','actual_rate']]
        score_table2_t.columns=[score2.name+'_pct_cum_target',score2.name+'_actual_rate']
        table_total=pd.concat([table_total,score_table2_t],axis=1)
        ax1.plot(table_total['pct_cum_acct'],table_total[score2.name+'_actual_rate'],'y',label=score2.name)
        
        pct_cum_acct=table_total['pct_cum_acct']
        pct_cum_acct.at[-1]=0
        pct_cum_target=table_total[score2.name+'_pct_cum_target']
        pct_cum_target.at[-1]=0
        ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=score2.name)
    
    
    s_pct_cum_acct=score_table['pct_cum_acct']
    s_pct_cum_acct.at[-1]=0
    
    s_pct_cum_target=score_table['pct_cum_target']
    s_pct_cum_target.at[-1]=0
    
    s_pct_cum_nontarget=score_table['pct_cum_nontarget']
    s_pct_cum_nontarget.at[-1]=0
    
    
    #ax0.plot(s_pct_cum_acct.sort_index(),s_pct_cum_target.sort_index(),'--r',label="Target")
    #ax0.plot(s_pct_cum_acct.sort_index(),s_pct_cum_nontarget.sort_index(),'--b',label="Non-Target")
    
    
    ax2.plot(s_pct_cum_acct.sort_index(),s_pct_cum_acct.sort_index(),':',label='baseline')
    
    print()
    print("Gains Table--Model:")
    score_table_print=score_table.copy()
    
    score_table_print['pct_cum_acct']=score_table_print['pct_cum_acct'].map('{:,.1%}'.format)
    score_table_print['pct_cum_target']=score_table_print['pct_cum_target'].map('{:,.1%}'.format)
    score_table_print['pct_cum_nontarget']=score_table_print['pct_cum_nontarget'].map('{:,.1%}'.format)
    score_table_print['actual_rate']=score_table_print['actual_rate'].map('{:,.1%}'.format)
    
    print(score_table_print)
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
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    
    ax2.legend()
    ax2.set_xlabel('% Total Population')
    ax2.set_ylabel('% Cum Target')
    ax2.set_title('Gains Table')
    ax2.set_ylim([0,1])
    ax2.set_xlim([0,1])
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    plt.show()
    print()


    
# for model prediction and becnhamrk   
def model_score_eval(Y_true,model,score=[],score2=[],Y_weight=[],decile=10):
 
    fig = plt.figure(figsize=(14,12))
    #ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(221)
    
    if len(Y_weight)!=len(Y_true):
        Y_weight=pd.Series(np.ones(len(Y_true)))
        
    if len(model)==len(Y_true):
        auc_bm=roc_auc_score(Y_true,model,sample_weight=Y_weight)
        model_table=score_gain(Y_true,Y_weight,model,decile,1)
        print(model.name,"AUC: ",round(auc_bm,5), " KS: ", round(model_table['KS'].max(),3))
        
        model_table_t=model_table[['pct_cum_acct','pct_cum_target','actual_rate']]
        model_table_t.columns=['pct_cum_acct',model.name+'_pct_cum_target',model.name+'_actual_rate']
        table_total= model_table_t.copy()
        
        ax1.plot(table_total['pct_cum_acct'],table_total[model.name+'_actual_rate'],label=model.name)
        
        pct_cum_acct=table_total['pct_cum_acct']
        pct_cum_acct.at[-1]=0
        pct_cum_target=table_total[model.name+'_pct_cum_target']
        pct_cum_target.at[-1]=0
        ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=model.name)        
    else:
        print('model value lenghth does not match with Y_true')
    
    
    if len(score)==len(Y_true):
        auc_bm=1-roc_auc_score(Y_true,score,sample_weight=Y_weight)
        score_table=score_gain(Y_true,Y_weight,score,decile,0)
        print(score.name,"AUC: ",round(auc_bm,5), " KS: ", round(score_table['KS'].max(),3))
        
        score_table_t=score_table[['pct_cum_target','actual_rate']]
        score_table_t.columns=[score.name+'_pct_cum_target',score.name+'_actual_rate']
        table_total=pd.concat([table_total,score_table_t],axis=1)
        
        ax1.plot(table_total['pct_cum_acct'],table_total[score.name+'_actual_rate'],label=score.name)
        
        pct_cum_acct=table_total['pct_cum_acct']
        pct_cum_acct.at[-1]=0
        pct_cum_target=table_total[score.name+'_pct_cum_target']
        pct_cum_target.at[-1]=0
        ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=score.name)
    
    
    if len(score2)==len(Y_true):
        auc_bm2=1-roc_auc_score(Y_true,score2,sample_weight=Y_weight)
        score_table2=score_gain(Y_true,Y_weight,score2,decile,0)
        print(score2.name,"AUC: ",round(auc_bm2,5), " KS: ", round(score_table2['KS'].max(),3))
        
        score_table2_t=score_table2[['pct_cum_target','actual_rate']]
        score_table2_t.columns=[score2.name+'_pct_cum_target',score2.name+'_actual_rate']
        
        table_total=pd.concat([table_total,score_table2_t],axis=1)
        ax1.plot(table_total['pct_cum_acct'],table_total[score2.name+'_actual_rate'],'y',label=score2.name)
        
        pct_cum_acct=table_total['pct_cum_acct']
        pct_cum_acct.at[-1]=0
        pct_cum_target=table_total[score2.name+'_pct_cum_target']
        pct_cum_target.at[-1]=0
        ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=score2.name)
    
    
    #s_pct_cum_acct=score_table['pct_cum_acct']
    #s_pct_cum_acct.at[-1]=0
    
    #s_pct_cum_target=score_table['pct_cum_target']
    #s_pct_cum_target.at[-1]=0
    
    #s_pct_cum_nontarget=score_table['pct_cum_nontarget']
    #s_pct_cum_nontarget.at[-1]=0
    
    
    #ax0.plot(s_pct_cum_acct.sort_index(),s_pct_cum_target.sort_index(),'--r',label="Target")
    #ax0.plot(s_pct_cum_acct.sort_index(),s_pct_cum_nontarget.sort_index(),'--b',label="Non-Target")
    
    
    ax2.plot(pct_cum_acct.sort_index(),pct_cum_acct.sort_index(),':',label='baseline')
    
    print()
    print("Gains Table--Model:")
    model_table_print=model_table.copy()
    
    model_table_print['pct_cum_acct']=model_table_print['pct_cum_acct'].map('{:,.1%}'.format)
    model_table_print['pct_cum_target']=model_table_print['pct_cum_target'].map('{:,.1%}'.format)
    model_table_print['pct_cum_nontarget']=model_table_print['pct_cum_nontarget'].map('{:,.1%}'.format)
    model_table_print['actual_rate']=model_table_print['actual_rate'].map('{:,.1%}'.format)
    
    print( model_table_print)
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
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    
    ax2.legend()
    ax2.set_xlabel('% Total Population')
    ax2.set_ylabel('% Cum Target')
    ax2.set_title('Gains Table')
    ax2.set_ylim([0,1])
    ax2.set_xlim([0,1])
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    plt.show()
    print()
    
    
### for score only,
def score_eval_ks(Y_true,score,score2=[],Y_weight=[],decile=10,up=0):
 
    fig = plt.figure(figsize=(12,10))
    #ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(221)
    
    if len(Y_weight)!=len(Y_true):
        Y_weight=pd.Series(np.ones(len(Y_true)))
        
    if len(score)==len(Y_true):
        #auc_bm=1-roc_auc_score(Y_true,score,sample_weight=Y_weight)
        score_table=score_gain(Y_true,Y_weight,score,decile,up)
        print(" KS: ", round(score_table['KS'].max(),3))
        
        score_table_t=score_table[['pct_cum_acct','pct_cum_target','actual_rate']]
        score_table_t.columns=['pct_cum_acct',score.name+'_pct_cum_target',score.name+'_actual_rate']
        table_total= score_table_t.copy()
        
        ax1.plot(table_total['pct_cum_acct'],table_total[score.name+'_actual_rate'],label=score.name)
        
        pct_cum_acct=table_total['pct_cum_acct']
        pct_cum_acct.at[-1]=0
        pct_cum_target=table_total[score.name+'_pct_cum_target']
        pct_cum_target.at[-1]=0
        ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=score.name)
        
    if len(score2)==len(Y_true):
        score_table2=score_gain(Y_true,Y_weight,score2,decile,up)
        print(score2.name,"AUC: ",round(auc_bm2,5), " KS: ", round(score_table2['KS'].max(),3))
        
        score_table2_t=score_table2[['pct_cum_target','actual_rate']]
        score_table2_t.columns=[score2.name+'_pct_cum_target',score2.name+'_actual_rate']
        table_total=pd.concat([table_total,score_table2_t],axis=1)
        ax1.plot(table_total['pct_cum_acct'],table_total[score2.name+'_actual_rate'],'y',label=score2.name)
        
        pct_cum_acct=table_total['pct_cum_acct']
        pct_cum_acct.at[-1]=0
        pct_cum_target=table_total[score2.name+'_pct_cum_target']
        pct_cum_target.at[-1]=0
        ax2.plot(pct_cum_acct.sort_index(),pct_cum_target.sort_index(),label=score2.name)
    
    
    s_pct_cum_acct=score_table['pct_cum_acct']
    s_pct_cum_acct.at[-1]=0
    
    s_pct_cum_target=score_table['pct_cum_target']
    s_pct_cum_target.at[-1]=0
    
    s_pct_cum_nontarget=score_table['pct_cum_nontarget']
    s_pct_cum_nontarget.at[-1]=0
    
    
    #ax0.plot(s_pct_cum_acct.sort_index(),s_pct_cum_target.sort_index(),'--r',label="Target")
    #ax0.plot(s_pct_cum_acct.sort_index(),s_pct_cum_nontarget.sort_index(),'--b',label="Non-Target")
    
    
    ax2.plot(s_pct_cum_acct.sort_index(),s_pct_cum_acct.sort_index(),':',label='baseline')
    
    print()
    print("Gains Table--Model:")
    score_table_print=score_table.copy()
    
    score_table_print['pct_cum_acct']=score_table_print['pct_cum_acct'].map('{:,.1%}'.format)
    score_table_print['pct_cum_target']=score_table_print['pct_cum_target'].map('{:,.1%}'.format)
    score_table_print['pct_cum_nontarget']=score_table_print['pct_cum_nontarget'].map('{:,.1%}'.format)
    score_table_print['actual_rate']=score_table_print['actual_rate'].map('{:,.1%}'.format)
    
    print(score_table_print)
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
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    
    ax2.legend()
    ax2.set_xlabel('% Total Population')
    ax2.set_ylabel('% Cum Target')
    ax2.set_title('Gains Table')
    ax2.set_ylim([0,1])
    ax2.set_xlim([0,1])
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    plt.show()
    print()
    

    
    
    
    
    
    
    
    
   
    
    


    
    
   
    
    
    
    
    
    
    
    