# Libraries
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy.stats import norm
import source.engine.utils_clv as clv
from pathlib import Path
from datetime import datetime
import json

def can_transform(X):
    """
    Returns a pandas dataframe with transformed variables

    Arguments:
    X -- A pandas dataframe containing the variables used to build the cancelation curve

    Example:
    >>> curve = CurveCan()
    >>> df_transformed = curve.transform(X)

    Which returns: a pandas dataframe
    """
    return X

def can_predict(X,param_Can):
    """
    Returns a pandas dataframe with one additional column called "curve" (Cancelation Curve)

    Arguments:
    X -- a pandas dataframe cointaining transformed variables used to build the cancelation curve

    Example:
    >>> curve = CurveCan()
    >>> df_transformed = curve.predict(X)
    >>> df_curves = curve.predict(df_transformed)

    Which returns:
            curve
    0    [[0.0, 0.0, 0.03, 0.07, 0.09, 0.2, 0.4]]
    1    [[0.0, 0.0, 0.03, 0.04, 0.09, 0.1, 0.2]]
    2    [[0.0, 0.0, 0.05, 0.6, 0.08, 0.09, 0.1]]
    """
    X = pd.merge(X,param_Can,left_on=["cancel","inst_cancel"],right_on=["Producto","Plazo"],how="left")
    #can = X.loc[:,"Mes_0":"Mes_55"]
    can = X.loc[:,"Mes_1":"Mes_55"]
    can = pd.concat([can,pd.DataFrame(np.transpose(np.tile(can["Mes_55"],240-55).reshape(240-55,can.shape[0])))],axis=1)
    X["curve"] = np.concatenate(can.values,axis = 0).reshape(can.shape[0],can.shape[1]).tolist()
    return X["curve"].to_frame()

def pd_transform(X):
    """
    Returns a pandas dataframe with transformed columns added to the dataframe

    Arguments:
    X -- a pandas dataframe containing the variables used to build the pd curve.

    Example:
    >>> curve = CurvePD()
    >>> X = curve.transform(X)

    Which returns: a pandas dataframe
    """
    return X

def pd_predict(X, param_pd1, param_pd2):
    """
    Returns a pandas dataframe with one column of name "curve"

    Arguments:
    X -- a pandas dataframe containing the variables used to build the pd curve.

    Example:
    >>> curve = CurvePD()
    >>> df = curve.predict(X)

    Which returns:

            curve
    0    [[0.0, 0.0, 0.03, 0.07, 0.09, 0.2, 0.4]]
    1    [[0.0, 0.0, 0.03, 0.04, 0.09, 0.1, 0.2]]
    2    [[0.0, 0.0, 0.05, 0.6, 0.08, 0.09, 0.1]]

    """
    X = X.copy()
    columnas = X.columns
    X = pd.merge(X,param_pd1,left_on=["inst_pd1","seg_pd1","pd"],right_on=["Plazo","Segmento De Riesgo","Producto"],how="left")
    X_pd1 = X[X.notnull().Mes_1].drop(columns=["Producto"])
    X_pd2 = pd.merge(X[X.isnull().Mes_1].loc[:,columnas],param_pd2,left_on=["inst_pd2","seg_pd2"],right_on=["Plazo","Segmento De Riesgo"],how = "left")
    X_pd = pd.concat([X_pd1,X_pd2],axis = 0,ignore_index=True)
    X_pd.reset_index(drop = True)
    X_pd["orden2"] = X_pd["orden"]
    X_pd = X_pd.set_index("orden")
    X_pd = X_pd.sort_index(ascending=True)
    #pdf = X_pd.loc[:,"Mes_0":"Mes_55"]
    pdf = X_pd.loc[:,"Mes_1":"Mes_55"]
    #pdf.reindex(np.arange(0,pdf.shape[0]))
    pdf = pd.concat([pdf,pd.DataFrame(np.transpose(np.tile(pdf["Mes_55"],240-55).reshape(240-55,pdf.shape[0])))],axis=1)
    #pdf = pdf.join(pd.DataFrame(np.transpose(np.tile(pdf["Mes_55"],240-55).reshape(240-55,pdf.shape[0]))))
    X["curve"] = np.concatenate(pdf.values,axis = 0).reshape(pdf.shape[0],pdf.shape[1]).tolist()
    return X["curve"].to_frame()

def pre_transform(X, param_pre):
    """
    Returns a pandas dataframe with transformed variables

    Arguments:
    X -- A pandas dataframe containing the variables used to build the prepayment curve

    Example:
    >>> curve = CurvePre()
    >>> df_transformed = curve.transform(X)

    Which returns: a pandas dataframe
    """
    X = X.copy()
    return X

def pre_predict(X, param_pre):
    """
    Returns a pandas dataframe with one additional column called "curve" (Prepayment Curve)

    Arguments:
    X -- a pandas dataframe cointaining transformed variables used to build the prepayment curve

    Example:
    >>> curve = CurvePre()
    >>> df_transformed = curve.predict(X)
    >>> df_curves = curve.predict(df_transformed)

    Which returns:
            curve
    0    [[0.0, 0.0, 0.03, 0.07, 0.09, 0.2, 0.4]]
    1    [[0.0, 0.0, 0.03, 0.04, 0.09, 0.1, 0.2]]
    2    [[0.0, 0.0, 0.05, 0.6, 0.08, 0.09, 0.1]]
    """
    X = X.copy()
    
    # Getting coefficients in the same order as the df of transformed variables
    #merge entre lo que transform y param, toda la curva
    X = pd.merge(X,param_pre,left_on=["prepaid","inst_pre","seg_pre"],right_on=["Producto","Plazo","Segmento De Riesgo"],how="left")
    #pre = X.loc[:,"Mes_0":"Mes_55"]
    pre = X.loc[:,"Mes_1":"Mes_55"]
    pre = pd.concat([pre,pd.DataFrame(np.transpose(np.tile(pre["Mes_55"],240-55).reshape(240-55,pre.shape[0])))],axis=1)
    X["curve"] = np.concatenate(pre.values,axis = 0).reshape(pre.shape[0],pre.shape[1]).tolist()
    return X["curve"].to_frame()

def lgd_transform(X, param_lgd):
    """
    Returns a pandas dataframe with transformed variables

    Arguments:
    X -- A pandas dataframe containing the variables used to build the LGD curve

    Example:
    >>> curve = CurvePre()
    >>> df_transformed = curve.transform(X)

    Which returns: a pandas dataframe
    """
    return X

def lgd_predict(X, param_lgd):
    """
    Returns a pandas dataframe with one additional column called "curve" (LGD Curve)

    Arguments:
    X -- a pandas dataframe cointaining transformed variables used to build the LGD curve

    Example:
    >>> curve = CurveLGD()
    >>> df_transformed = curve.predict(X)
    >>> df_curves = curve.predict(df_transformed)

    Which returns:
            curve
    0    [[0.0, 0.0, 0.03, 0.07, 0.09, 0.2, 0.4]]
    1    [[0.0, 0.0, 0.03, 0.04, 0.09, 0.1, 0.2]]
    2    [[0.0, 0.0, 0.05, 0.6, 0.08, 0.09, 0.1]]
    """
    X = pd.merge(X,param_lgd,left_on=["inst_pre"],right_on=["Plazo"],how="left")
    #lgd = X.loc[:,"Mes_0":"Mes_55"]
    lgd = X.loc[:,"Mes_1":"Mes_55"]
    lgd = pd.concat([lgd,pd.DataFrame(np.transpose(np.tile(lgd["Mes_55"],240-55).reshape(240-55,lgd.shape[0])))],axis=1)
    X["curve"] = np.concatenate(lgd.values,axis = 0).reshape(lgd.shape[0],lgd.shape[1]).tolist()
    return X["curve"].to_frame()

def cap_transform(X, param_capital):
    """
    Returns a pandas dataframe with transformed variables

    Arguments:
    X -- A pandas dataframe containing the variables used to build the CAP curve

    Example:
    >>> curve = CurveCAP()
    >>> df_transformed = curve.transform(X)

    Which returns: a pandas dataframe
    """
    return X

def cap_predict(X, param_capital):
    """
    Returns a pandas dataframe with one additional column called "curve" (CAP Curve)

    Arguments:
    X -- a pandas dataframe cointaining transformed variables used to build the CAP curve

    Example:
    >>> curve = CurveCAP()
    >>> df_transformed = curve.predict(X)
    >>> df_curves = curve.predict(df_transformed)

    Which returns:
            curve
    0    [[0.0, 0.0, 0.03, 0.07, 0.09, 0.2, 0.4]]
    1    [[0.0, 0.0, 0.03, 0.04, 0.09, 0.1, 0.2]]
    2    [[0.0, 0.0, 0.05, 0.6, 0.08, 0.09, 0.1]]
    """
    X = X.copy()
    X = pd.merge(X,param_capital,left_on=["capital","seg_cap"],right_on=["Producto","SegmentoRiesgo"],how="left")
    cap = X.loc[:,"Req_Patrimonio_NIIF"]
    #cap_vector = pd.DataFrame(np.transpose(np.tile(cap,241).reshape(241,cap.shape[0])))
    cap_vector = pd.DataFrame(np.transpose(np.tile(cap,240).reshape(240,cap.shape[0])))
    #X["curve"] = np.concatenate(cap_vector.values,axis=0).reshape(cap.shape[0],241).tolist()
    X["curve"] = np.concatenate(cap_vector.values,axis=0).reshape(cap.shape[0],240).tolist()
    return X["curve"].to_frame()

def costos_transform(X):
    X = X.copy()
    #rango de monto para costo de venta
    X["Rango"] = [0]*X.shape[0]
    X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=0 ,X.MO_DESEMBOLSO_SOLES<5000),"Rango"] = 1
    X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=5000 ,X.MO_DESEMBOLSO_SOLES<10000),"Rango"] = 2
    X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=10000 ,X.MO_DESEMBOLSO_SOLES<20000),"Rango"] = 3
    X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=20000 ,X.MO_DESEMBOLSO_SOLES<40000),"Rango"] = 4
    X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=40000 ,X.MO_DESEMBOLSO_SOLES<60000),"Rango"] = 5
    X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=60000 ,X.MO_DESEMBOLSO_SOLES<80000),"Rango"] = 6
    X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=80000 ,X.MO_DESEMBOLSO_SOLES<100000),"Rango"] = 7
    X.loc[X.MO_DESEMBOLSO_SOLES>=100000,"Rango"] = 8
    return X

def costos_predict(X, param_t0,param_t1T):
    costo = pd.merge(X,param_t0,left_on=["Rango","CANAL","TIPO_CLIENTE"],right_on=["Rango","Canal","TIPO_CLIENTE"],how="left")
    X["costo_t0"] = costo["CostoVenta"]
    #costo servir
    X["costo_t1T"]=param_t1T.iat[0,0]
    return X["costo_t0"].to_frame(),X["costo_t1T"].to_frame()

def constant_installment(principal,interest_rate,contratual_maturity):
    # calcula la cuota a pagar (constante)
    return(principal*((interest_rate*(1+interest_rate)**contratual_maturity)/((1+interest_rate)**contratual_maturity-1)))

def payment_schedule(X):
    # Calcula el cronograma
    Y = X.copy()
    Y["TEM"] = np.power(1+X["TEA"],1/12)-1
    Y["const_inst"] = constant_installment(Y["MO_DESEMBOLSO_SOLES"],Y["TEM"],Y["PLAZO_MESES"])
    pay_sche = []
    for ind,fila in Y.iterrows():
        pay_sche.append(pd.DataFrame({"num_of_inst" : np.arange(1,fila["PLAZO_MESES"]+1),
                                      "NU_PRESTAMO" : np.repeat(fila["NU_PRESTAMO"],fila["PLAZO_MESES"])},
                                      columns=["num_of_inst","NU_PRESTAMO"]))
    pay_sche = pd.concat(pay_sche)
    pay_sche = pd.merge(pay_sche,Y,left_on=["NU_PRESTAMO"],right_on=["NU_PRESTAMO"],how="left")
    pay_sche["capital_balance"] = pay_sche["MO_DESEMBOLSO_SOLES"]*(np.power(1+pay_sche["TEM"],pay_sche["PLAZO_MESES"])-np.power(1+pay_sche["TEM"],pay_sche["num_of_inst"]-1))/((np.power(1+pay_sche["TEM"],pay_sche["PLAZO_MESES"])-1))
    pay_sche["interest"] = pay_sche["capital_balance"]*pay_sche["TEM"]
    pay_sche["amortization"] = pay_sche["MO_DESEMBOLSO_SOLES"]*pay_sche["TEM"]*(np.power(1+pay_sche["TEM"],pay_sche["num_of_inst"]-1))/(np.power(1+pay_sche["TEM"],pay_sche["PLAZO_MESES"])-1)
    return(pay_sche)
    
def schedule_inst_dur(pay_sche):
    pay_sche["p_capital_balance"] = pay_sche["capital_balance"]
    pay_sche["p_interest"] = pay_sche["interest"]
    pay_sche["p_amortization"] = pay_sche["amortization"]
    pay_sche["prepayment"] = (pay_sche["p_capital_balance"]- pay_sche["p_amortization"])*pay_sche["curve_prepayment"]
    for prestamo in pay_sche["NU_PRESTAMO"].drop_duplicates():
        pay_sche[pay_sche.NU_PRESTAMO.eq(prestamo)].num_of_inst
        for ind,fila in pay_sche[pay_sche.NU_PRESTAMO.eq(prestamo)].iterrows():
            if fila.num_of_inst>1:
                pay_sche.at[ind,"p_capital_balance"] = pay_sche.loc[ind-1,"p_capital_balance"] - pay_sche.loc[ind-1,"p_amortization"] - pay_sche.loc[ind-1,"prepayment"] if pay_sche.loc[ind-1,"p_capital_balance"] - pay_sche.loc[ind-1,"p_amortization"] - pay_sche.loc[ind-1,"prepayment"]>0 else 0
                pay_sche.at[ind,"p_interest"] = pay_sche.loc[ind,"p_capital_balance"] * pay_sche.loc[ind,"TEM"]
                pay_sche.at[ind,"p_amortization"] = pay_sche.loc[ind,"const_inst"]-pay_sche.loc[ind,"p_interest"] if pay_sche.loc[ind,"const_inst"]-pay_sche.loc[ind,"p_interest"] < pay_sche.loc[ind,"p_capital_balance"] else pay_sche.loc[ind,"p_capital_balance"]
                pay_sche.at[ind,"prepayment"] = pay_sche.loc[ind,"curve_prepayment"]*(pay_sche.loc[ind,"p_capital_balance"]-pay_sche.loc[ind,"p_amortization"]) if pay_sche.loc[ind,"curve_prepayment"]*(pay_sche.loc[ind,"p_capital_balance"]-pay_sche.loc[ind,"p_amortization"])>0 else 0
    pay_sche["numerador"] = pay_sche["num_of_inst"]*(pay_sche["p_interest"] + pay_sche["p_amortization"] + pay_sche["prepayment"])/np.power(1+pay_sche["TEM"],pay_sche["num_of_inst"])
    pay_sche["denominador"] = (pay_sche["p_interest"] + pay_sche["p_amortization"] + pay_sche["prepayment"])/np.power(1+pay_sche["TEM"],pay_sche["num_of_inst"])
    durations = pay_sche[["NU_PRESTAMO","numerador","denominador"]].groupby(["NU_PRESTAMO"]).sum()
    durations["duration_i"] = (durations["numerador"]/durations["denominador"])/12
    durations.reset_index(level=0,inplace=True)
    return(durations[["NU_PRESTAMO","duration_i"]])

def schedule_const_inst(pay_sche):
    pay_sche["p_capital_balance"] = pay_sche["capital_balance"]
    pay_sche["p_interest"] = pay_sche["interest"]
    pay_sche["p_amortization"] = pay_sche["amortization"]
    pay_sche["prepayment"] = (pay_sche["p_capital_balance"]- pay_sche["p_amortization"])*pay_sche["curve_prepayment"]
    for prestamo in pay_sche["NU_PRESTAMO"].drop_duplicates():
        pay_sche[pay_sche.NU_PRESTAMO.eq(prestamo)].num_of_inst
        for ind,fila in pay_sche[pay_sche.NU_PRESTAMO.eq(prestamo)].iterrows():
            if fila.num_of_inst>1:
                pay_sche.at[ind,"p_capital_balance"] = pay_sche.loc[ind-1,"p_capital_balance"] - pay_sche.loc[ind-1,"p_amortization"] - pay_sche.loc[ind-1,"prepayment"] if pay_sche.loc[ind-1,"p_capital_balance"] - pay_sche.loc[ind-1,"p_amortization"] - pay_sche.loc[ind-1,"prepayment"]>0 else 0
                pay_sche.at[ind,"p_interest"] = pay_sche.loc[ind,"p_capital_balance"] * pay_sche.loc[ind,"TEM"]
                pay_sche.at[ind,"p_amortization"] = constant_installment(pay_sche.loc[ind,"p_capital_balance"],pay_sche.loc[ind,"TEM"],pay_sche.loc[ind,"PLAZO_MESES"]-pay_sche.loc[ind,"num_of_inst"]+1)-pay_sche.loc[ind,"p_interest"]
                pay_sche.at[ind,"prepayment"] = pay_sche.loc[ind,"curve_prepayment"]*(pay_sche.loc[ind,"p_capital_balance"]-pay_sche.loc[ind,"p_amortization"]) if pay_sche.loc[ind,"curve_prepayment"]*(pay_sche.loc[ind,"p_capital_balance"]-pay_sche.loc[ind,"p_amortization"])>0 else 0
    pay_sche["numerador"] = pay_sche["num_of_inst"]*(pay_sche["p_interest"] + pay_sche["p_amortization"] + pay_sche["prepayment"])/np.power(1+pay_sche["TEM"],pay_sche["num_of_inst"])
    pay_sche["denominador"] = (pay_sche["p_interest"] + pay_sche["p_amortization"] + pay_sche["prepayment"])/np.power(1+pay_sche["TEM"],pay_sche["num_of_inst"])
    durations = pay_sche[["NU_PRESTAMO","numerador","denominador"]].groupby(["NU_PRESTAMO"]).sum()
    durations["duration_c"] = (durations["numerador"]/durations["denominador"])/12
    durations.reset_index(level=0,inplace=True)
    return(durations[["NU_PRESTAMO","duration_c"]])

def duration_model(X,dur_prep):
    Y = X.copy()
    red_const_inst=0.5
    red_inst_dur=0.5
    pay_sche = payment_schedule(Y)
    pay_sche["key"] = 1
    dur_prep["key"] = 1
    cruce = pd.merge(pay_sche[["NU_PRESTAMO","PLAZO_MESES","key"]].drop_duplicates(),dur_prep,on=["key"])
    cruce = cruce[cruce["PLAZO_MESES"]>=cruce["maduracion"]]
    del pay_sche["key"]
    del dur_prep["key"]
    del cruce["key"]
    del cruce["PLAZO_MESES"]
    cruce = pd.merge(pay_sche,cruce,left_on=["NU_PRESTAMO","num_of_inst"],right_on=["NU_PRESTAMO","maduracion"],how="left")
    del cruce["maduracion"]
    #add the durations
    Y = pd.merge(Y,schedule_inst_dur(cruce),left_on=["NU_PRESTAMO"],right_on=["NU_PRESTAMO"],how="left")
    Y = pd.merge(Y,schedule_const_inst(cruce),left_on=["NU_PRESTAMO"],right_on=["NU_PRESTAMO"],how="left")
    Y["duration"]=(round((red_const_inst * Y["duration_c"] + red_inst_dur*Y["duration_i"])*12*30))
    Y.duration = Y.duration.astype(int)
    del Y["duration_c"]
    del Y["duration_i"]
    return Y

def tt_desc_transform(X,pre_dur_curve,param_td,param_remcap):
    df = duration_model(X,pre_dur_curve)
    #df.loc[:,"td"] = param_td.values[0][0]
    df.loc[:,"td"] = df.loc[:,"TIR_OBJ"]
    df.loc[:,"remcap"] = param_remcap.values[0][0]
    return df

def tt_desc_predict(X,tt_list):
    return pd.merge(X,tt_list,left_on=["PERIODO","duration"],right_on=["PERIODO","Plazo"],how="left")[["TT"]]

def inof_transform(X):
    #generamos el rango de plazo para el inof
    X["rango_plazo"] = 0
    X.loc[np.logical_and(X.PLAZO_MESES>=0,X.PLAZO_MESES<=6),"rango_plazo"] = 1
    X.loc[np.logical_and(X.PLAZO_MESES>=7,X.PLAZO_MESES<=12),"rango_plazo"] = 2
    X.loc[np.logical_and(X.PLAZO_MESES>=13,X.PLAZO_MESES<=18),"rango_plazo"] = 3
    X.loc[np.logical_and(X.PLAZO_MESES>=19,X.PLAZO_MESES<=24),"rango_plazo"] = 4
    X.loc[np.logical_and(X.PLAZO_MESES>=25,X.PLAZO_MESES<=30),"rango_plazo"] = 5
    X.loc[np.logical_and(X.PLAZO_MESES>=31,X.PLAZO_MESES<=36),"rango_plazo"] = 6
    X.loc[np.logical_and(X.PLAZO_MESES>=37,X.PLAZO_MESES<=42),"rango_plazo"] = 7
    X.loc[np.logical_and(X.PLAZO_MESES>=43,X.PLAZO_MESES<=48),"rango_plazo"] = 8
    X.loc[np.logical_and(X.PLAZO_MESES>=49,X.PLAZO_MESES<=54),"rango_plazo"] = 9
    X.loc[np.logical_and(X.PLAZO_MESES>=55,X.PLAZO_MESES<=60),"rango_plazo"] = 10
    X.loc[np.logical_and(X.PLAZO_MESES>=61,X.PLAZO_MESES<=66),"rango_plazo"] = 11
    X.loc[np.logical_and(X.PLAZO_MESES>=67,X.PLAZO_MESES<=72),"rango_plazo"] = 12
    X.loc[np.logical_and(X.PLAZO_MESES>=73,X.PLAZO_MESES<=78),"rango_plazo"] = 13
    X.loc[np.logical_and(X.PLAZO_MESES>=79,X.PLAZO_MESES<=84),"rango_plazo"] = 14
    X.loc[np.logical_and(X.PLAZO_MESES>=85,X.PLAZO_MESES<=90),"rango_plazo"] = 15
    X.loc[np.logical_and(X.PLAZO_MESES>=91,X.PLAZO_MESES<=96),"rango_plazo"] = 16
    X.loc[np.logical_and(X.PLAZO_MESES>=97,X.PLAZO_MESES<=120),"rango_plazo"] = 17
    X.loc[np.logical_and(X.PLAZO_MESES>=121,X.PLAZO_MESES<=144),"rango_plazo"] = 18
    X.loc[np.logical_and(X.PLAZO_MESES>=145,X.PLAZO_MESES<=168),"rango_plazo"] = 19
    X.loc[np.logical_and(X.PLAZO_MESES>=169,X.PLAZO_MESES<=180),"rango_plazo"] = 20
    return X

def inof_predict(X,inof):
    X1 = pd.merge(X,inof,left_on=["rango_plazo"],right_on=["bucket"],how="left")
    X1["inof"] = X1["MO_DESEMBOLSO_SOLES"]*(1-(1/(1+X1["prima"])))*X1["inof"]*X1["factor_ajuste"]
    return X1["inof"].to_frame()

def log(cod_modelo,autor,matricula,obs,funcion):

    exDict = {
        'CODIGO_MODELO': cod_modelo,
        'Autor': autor, 
        'Matricula': matricula, 
        'Funcion': funcion, 
        'Fecha':str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S')), 
        'Cantidad':obs.shape[0]
        }
        
    with open( Path(".").resolve() / 'logs_clv.txt', 'a') as file:
        file.write(json.dumps(exDict)+'\n')
