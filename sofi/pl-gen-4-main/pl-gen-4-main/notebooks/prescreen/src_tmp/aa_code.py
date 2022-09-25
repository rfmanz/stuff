import pandas as pd
import numpy as np


def adverse_action_code(input_df,pdp,model,aa_df):
           
    inq_list=['IQT9420']
    
    i=0
    feature_list=model.feature_name()
    impact_df=pd.DataFrame()
    grid={}
    
    for attr in feature_list:
        if input_df[attr].isna().sum()>=1:
            pos=10000
        else:
            a=np.array(input_df[attr].values)-np.array(pdp[attr][0:-1]['min'].to_list())
            b=np.ma.MaskedArray(a,a<0)
            pos=np.ma.argmin(b)
            
            
        impact_df.loc[i,'attr']=attr
        impact_df.loc[i,'pos']=pos
        impact_df.loc[i,'value']=input_df[attr].values
        impact_df.loc[i,'impact']=pdp[attr]['impact'][pos]
        i=i+1
    
    impact_df.sort_values(by='impact',ascending=False,inplace=True)
    
    #impact_df['inq_flag']=0
    #impact_df.loc[impact_df['attr'].isin(inq_list),'inq_flag']=1
    #rank_list=list(range(len(feature_list)))
    
    #impact_df['rank'] = rank_list
    #impact_df.reset_index(drop=True,inplace=True)
    
    #impact_df.loc[(impact_df['rank']<=3)&(impact_df['inq_flag']==1),'inq_top4']=1
    #impact_df.loc[(impact_df['inq_flag']==1)&(impact_df['impact']>0),'inq_impact']=1
    
    #if impact_df['inq_top4'].sum()>=1:
    #    top_df=impact_df[impact_df['rank']<=3]
        
    #elif impact_df['inq_impact'].sum()>=1:
    #    top_df=impact_df[(impact_df['rank']<=3)|(impact_df['inq_impact']==1)][0:5]
    #else:
    #    top_df=impact_df[impact_df['rank']<=3]
  
    ##top_df=top_df[top_df['impact']>0]
    
    impact_df.loc[impact_df['attr'].isin(inq_list),'inq_flag']=1
    
    
    impact_df['inq_impact']=0
    impact_df.loc[(impact_df['inq_flag']==1)&(impact_df['impact']>0.0085),'inq_impact']=1
    
    pos_impact=impact_df[impact_df.impact>0]
    pos_impact_aa=pos_impact.merge(aa_df,how='left',on='attr')
    
    pos_impact_aa['aa']=np.where(pos_impact_aa['value'].isnull(),pos_impact_aa['AA_code_missing'],pos_impact_aa['AA_code_valid'])
    
    pos_impact_aa.sort_values(by=['inq_impact','impact'],ascending=False,inplace=True)
    pos_impact_aa.reset_index(drop=True,inplace=True)
    
    if impact_df['inq_impact'].max()==1:
        n=5
    else:
        n=4
        
    aa_output=pos_impact_aa.drop_duplicates(['aa'],keep='first')
    aa_output= aa_output[0:n]
    aa_output.sort_values(by=['impact'],ascending=False,inplace=True)
    #pos_impact_aa.drop_duplicates(['aa'],keep='first',inplace=True)
    
    return pos_impact_aa,aa_output
    
 
    
    
