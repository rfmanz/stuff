import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import math


def partial_dependency_woe(model, df, feature,model_features,woe_dict=None, num_cut=20, sample_size=None):
    """
    Calculate partial dependency of a feature given a model.
    """
    if sample_size:
        d = df[model_features].sample(sample_size).copy()
    else:
        d = df[model_features].copy()
    
    
    ### check if woe_dict is available
    if woe_dict is None:
        cutoff1=[0]
    else:
        cutoff1= woe_dict[feature]['cutoff']
    
    cutoff=cutoff1.copy()
            
    ### if woe cut is less than num_cut, then cut the values based on quantiles
    ### get the list of cutoff values for each attributes
    if len(cutoff1)<num_cut:
        quant_list=list(np.linspace(0.001,0.999,num_cut))
        cutoff=[df[feature].quantile(x) for x in quant_list]
        cutoff=list(set(cutoff))
        cutoff.insert(len(cutoff),float("inf"))
        cutoff.insert(0,float("-inf"))    
    
    if (len(cutoff1)>=len(cutoff)/4)&(len(cutoff1)>=20):
        cutoff=cutoff1
    
    cutoff.sort()
    
    lable=list(range(1,len(cutoff)))
    d['attr_label']=pd.cut(d[feature],cutoff,right=True,labels=lable)
    
    df_cut=pd.DataFrame()
    
    df_cut['min']=d[d[feature].isnull()==False].groupby('attr_label')[feature].min()
    df_cut['max']=d[d[feature].isnull()==False].groupby('attr_label')[feature].max()
    df_cut['grid']=d[d[feature].isnull()==False].groupby('attr_label')[feature].median()
    df_cut=df_cut.reset_index()
    df_cut=df_cut[df_cut['grid'].isnull()==False]
    
    
    df_cut.at[10000,'min']=np.nan
    df_cut.at[10000,'max']=np.nan
    df_cut.at[10000,'grid']=np.nan
    
    grid=df_cut['grid'].to_list()
    
    preds={}
    
    
    for x in grid:
        d[feature] = x
        y = np.average(model.predict(d[model_features]))
        if np.isnan(x):
            df_cut.loc[df_cut['grid'].isnull(),'pred']=y
        else:
            df_cut.loc[df_cut['grid']==x,'pred']=y 
            
    return df_cut



def pdp_dict(model, df, attr_list=None, woe_dict=None, num_cut=20,sample_size=None,print_ind=False):
    pdp_dict={}
   
    model_features=model.feature_name()
    
    pred_mean=np.mean(model.predict(df[model_features]))
    
    #print("processing pdp:") 
    fig = plt.figure(figsize=(30, 160))
    i=0
    
    attr_importance_df1=pd.DataFrame()
    attr_importance_df1['feature']=model.feature_name()
    attr_importance_df1['importance']=model.feature_importance(importance_type='gain',iteration=model.best_iteration)
    attr_importance_df1.sort_values(by="importance",ascending=False,inplace=True)
    
    if attr_list is None:
        attr_list=attr_importance_df1['feature'].to_list()
    
    for attr in  attr_list:
        
        df_11=partial_dependency_woe(model, df, attr, model_features, woe_dict,num_cut,sample_size=sample_size)
        nn=len(df_11)
        df_11['impact']=df_11['pred']-np.repeat(pred_mean,nn)
        pdp_dict[attr]=df_11
        i=i+1
        
        if print_ind==True:
            #if len(df_11)<num_cut:
            #    ax = fig.add_subplot(20, 3, i)
            #    x_list=list(range(len(pdp_dict[attr]['grid'])-1))
            #    ax.plot(x_list,pdp_dict[attr]['impact'][:-1],'C2')
            #    ax.set_xticks(x_list)
            #    ax.set_xticklabels(np.round(pdp_dict[attr]['grid'][:-1],3),rotation=45,fontsize=10)
                #ax.set_xticklabels(rotation=45)
            #    #ax.xaxis.set_major_locator(ticker.FixedLocator([list(pdp_dict['g201a']['grid'][0:3])+list(pdp_dict['g201a']['grid'][-3:])]))
            #    ax.title.set_text(attr) 
            #    ax.scatter(0 ,pdp_dict[attr]['impact'][-1],  color = 'red', marker= 'o')
            
            df_plot=df_11[df_11['grid'].isnull()==False]
            df_plot['attr_num']=pd.to_numeric(df_plot['attr_label'])
            ax = fig.add_subplot(20, 3, i)
            ax.plot(df_plot['attr_num'].values, df_plot['impact'].values,'C2')
            ax.axhline(y=0,linestyle='--', color='g')
            ax.get_xaxis().set_ticks([])
            x_min=df_11['min'].min()
            x_max=df_11['max'].max()
            xlable='min:{0}                                                              max:{1}'.format(x_min,x_max)
            ax.set_xlabel(xlable)
            ax.set_ylabel('impact')
            ax.title.set_text(attr)
            ax.scatter(1 ,df_11['impact'][10000],  color = 'red', marker= 'X',s=40)
                             
    return pdp_dict
        

    
def partial_dependency_ice(model, df, feature,woe_dict=None, num_cut=20,sample_size=None):
    """
    Calculate partial dependency of a feature given a model.
    """
    
    model_features=model.feature_name()
    
    if sample_size:
        d = df[model_features].sample(sample_size).copy()
    else:
        d = df[model_features].copy()
    
    
    ### check if woe_dict is available
    if woe_dict is None:
        cutoff1=[0]
    else:
        cutoff1= woe_dict[feature]['cutoff']
    
    cutoff=cutoff1.copy()
            
    ### if woe cut is less than num_cut, then cut the values based on quantiles
    ### get the list of cutoff values for each attributes
    if len(cutoff1)<num_cut:
        quant_list=list(np.linspace(0.001,0.999,num_cut))
        cutoff=[df[feature].quantile(x) for x in quant_list]
        cutoff=list(set(cutoff))
        cutoff.insert(len(cutoff),float("inf"))
        cutoff.insert(0,float("-inf"))    
    
    if len(cutoff1)>=len(cutoff):
        cutoff=cutoff1
    
    cutoff.sort()
    
    lable=list(range(1,len(cutoff)))
    d['attr_label']=pd.cut(d[feature],cutoff,right=True,labels=lable)
    
    df_cut=pd.DataFrame()
    
    df_cut['min']=d[d[feature].isnull()==False].groupby('attr_label')[feature].min()
    df_cut['max']=d[d[feature].isnull()==False].groupby('attr_label')[feature].max()
    df_cut['grid']=d[d[feature].isnull()==False].groupby('attr_label')[feature].median()
    df_cut=df_cut.reset_index()
    df_cut=df_cut[df_cut['grid'].isnull()==False]
    
    
    df_cut.at[10000,'min']=np.nan
    df_cut.at[10000,'max']=np.nan
    df_cut.at[10000,'grid']=np.nan
    
    grid=df_cut['grid'].to_list()
      
    for x in grid:
        d[feature] = x
        d['model']=model.predict(d[model_features])
        y = d['model'].mean()
                                 
        if np.isnan(x):
            df_cut.loc[df_cut['grid'].isnull(),'pred']=y
        else:
            df_cut.loc[df_cut['grid']==x,'pred']=y 
                                 
        pred_quantile={}
        
        quant_list=[]
        for quant in [0.1,0.25,0.5,0.75,0.9]:
            i='p_'+str(int(quant*100))
            quant_list.append(i)
            pred_quantile[i]=d['model'].quantile(quant)                              
            if np.isnan(x):
                df_cut.loc[df_cut['grid'].isnull(),i]=pred_quantile[i]
            else:
                df_cut.loc[df_cut['grid']==x,i]=pred_quantile[i] 
    return df_cut,quant_list


def pdp_ice_plot(model, df, attr_list=None, woe_dict=None, num_cut=20,sample_size=None,print_ind=True,out_path='ice_pdp.pdf'):
    pdp_dict={}
    
    model_features=model.feature_name()
    
    pred_mean=np.mean(model.predict(df[model_features]))
    
    #print("processing pdp:") 
    fig = plt.figure(figsize=(30, 240))
    i=0
    
    attr_importance_df1=pd.DataFrame()
    attr_importance_df1['feature']=model.feature_name()
    attr_importance_df1['importance']=model.feature_importance(importance_type='gain',iteration=model.best_iteration)
    attr_importance_df1.sort_values(by="importance",ascending=False,inplace=True)
    
    if attr_list is None:
        attr_list=attr_importance_df1['feature'].to_list()
    
    
    for attr in  attr_list:
        
        df_11,q_list=partial_dependency_ice(model, df, attr, woe_dict,num_cut,sample_size)
        
        nn=len(df_11)
        df_11['impact']=df_11['pred']-np.repeat(pred_mean,nn)
        for aa in q_list:
            df_11[aa]=df_11[aa]-np.repeat(pred_mean,nn)
        
        i=i+1
        
        if print_ind==True:
            
            df_plot=df_11[df_11['grid'].isnull()==False]
            df_plot['attr_num']=pd.to_numeric(df_plot['attr_label'])
            ax = fig.add_subplot(18, 2, i)
            
            ax.plot(df_plot['attr_num'].values, df_plot['impact'].values,'C2',linewidth=4)
            
            for aa in q_list:
                ax.plot(df_plot['attr_num'].values ,df_plot[aa].values, linestyle='dotted',linewidth=3)
            
            ax.axhline(y=0,linestyle='--', color='g')
            ax.get_xaxis().set_ticks([])
            x_min=df_11['min'].min()
            x_max=df_11['max'].max()
            xlable='min:{0}                                                              max:{1}'.format(x_min,x_max)
            ax.set_xlabel(xlable,size=20)
            ax.set_ylabel('impact',size=20)
            ax.set_title(attr,fontsize=20)
            ax.tick_params(labelsize=20)
            
            ax.scatter(1 ,df_11['impact'][10000],  color = 'red', marker= 'X',s=80)
            
        plt.savefig(out_path)
        
    
            
        
    

             

