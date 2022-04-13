# Script Curvas CLV

import pandas as pd
import numpy as np
import source.engine.utils as ut
from pathlib import Path

# 1. Cancelaciones
'''p1 = Path(".").resolve() / "data" / "can_thetaTt.csv"
p2 = Path(".").resolve() / "data" / "can_thetaT.csv"
p5 = Path(".").resolve() / "data" / "varios_PDcalibrada.csv"
p6 = Path(".").resolve() / "data" / "varios_Group_PD.csv"
p7 = Path(".").resolve() / "data" / "varios_Enalta.csv"
param_Tt = pd.read_csv(str(p1))
param_T = pd.read_csv(str(p2))
param_PDCal = pd.read_csv(str(p5))
param_GroupPD = pd.read_csv(str(p6))
param_Enalta = pd.read_csv(str(p7))

hyperparams = {
    "param_Tt": param_Tt,
    "param_T": param_T,
    "param_PDCal": param_PDCal,
    "param_GroupPD": param_GroupPD,
    "param_Enalta": param_Enalta}'''

class CurveCan():
    """ Creating Cancelation Probability Curves """

    def __init__(self, hyperparams):
        for key in hyperparams:
            setattr(self, key, hyperparams[key])

    def transform(self,X):
        return ut.can_transform(X)

    def predict(self, X):
        X_predict = ut.can_predict(X, self.param_Can)
        return X_predict

# 2. PD Curve
'''p1 = Path(".").resolve() / "data" / "pd_thetaTt.csv"
p2 = Path(".").resolve() / "data" / "pd_thetaT.csv"
p3 = Path(".").resolve() / "data" / "pd_scalars.csv"
p4 = Path(".").resolve() / "data" / "pd_idgroup.csv"
param_Tt = pd.read_csv(str(p1))
param_T = pd.read_csv(str(p2))
param_scalar = pd.read_csv(str(p3))
param_id_group = pd.read_csv(str(p4))

hyperparams = {
    "param_Tt": param_Tt,
    "param_T": param_T,
    "param_scalar": param_scalar,
    "param_id_group": param_id_group}'''

class CurvePD():
    """ Creating Default Probability Temporal Curves """

    def __init__(self, hyperparams):
        for key in hyperparams:
            setattr(self, key, hyperparams[key])

    def transform(self, X):
        X_transformed = ut.pd_transform(X)
        return X_transformed

    def predict(self, X):
        X_predict = ut.pd_predict(X, self.param_pd1, self.param_pd2)
        return X_predict

# 3. Prepayment Curve
'''p1 = Path(".").resolve() / "data" / "pre_Logit.csv"
p2 = Path(".").resolve() / "data" / "pre_MCO.csv"
p3 = Path(".").resolve() / "data" / "pre_Primera.csv"
p5 = Path(".").resolve() / "data" / "varios_PDcalibrada.csv"
p6 = Path(".").resolve() / "data" / "varios_Group_PD.csv"
p7 = Path(".").resolve() / "data" / "varios_Enalta.csv"
param_Logit = pd.read_csv(str(p1))
param_MCO = pd.read_csv(str(p2))
param_Primera = pd.read_csv(str(p3))
param_PDCal = pd.read_csv(str(p5))
param_GroupPD = pd.read_csv(str(p6))
param_Enalta = pd.read_csv(str(p7))

hyperparams = {
    "param_Logit": param_Logit,
    "param_MCO": param_MCO,
    "param_Primera": param_Primera,
    "param_PDCal": param_PDCal,
    "param_GroupPD": param_GroupPD,
    "param_Enalta": param_Enalta}'''

# Prepayment Curve
class CurvePre():
    """ Creating the Prepayment Curve """

    def __init__(self, hyperparams):
        for key in hyperparams:
            setattr(self, key, hyperparams[key])

    def transform(self, X):
        X_transformed = ut.pre_transform(X, self.param_pre)
        return X_transformed

    def predict(self, X):
        X_predict = ut.pre_predict(X,self.param_pre)
        return X_predict


# 4. LGD Curve 
'''lgd1 = Path(".").resolve() / "data" / "lgd_ThetaT.csv"
lgd2 = Path(".").resolve() / "data" / "lgd_Limites_Inputs.csv"
lgd3 = Path(".").resolve() / "data" / "lgd_Enalta.csv"
param_T = pd.read_csv(str(lgd1))
limit_T = pd.read_csv(str(lgd2))
lgd_Enalta = pd.read_csv(str(lgd3))

hyperparams = {
    "param_T": param_T,
    "limit_T": limit_T,
    "lgd_Enalta": lgd_Enalta}'''

# LGD Curve
class CurveLGD():
    """ Creating LGD Curves """

    def __init__(self, hyperparams):
        for key in hyperparams:
            setattr(self, key, hyperparams[key])

    def transform(self, X):
        X_transformed = ut.lgd_transform(X, self.param_lgd)
        return X_transformed

    def predict(self, X):
        X_predict = ut.lgd_predict(X, self.param_lgd)
        return X_predict


# 7. CAP Curve
'''p1 = Path(".").resolve() / "data" / "ecap_scalars.csv"
p2 = Path(".").resolve() / "data" / "ecap_PDTTC.csv"
param_scalar = pd.read_csv(str(p1))
param_PDTTC = pd.read_csv(str(p2))

hyperparams = {"param_scalar": param_scalar, "param_PDTTC": param_PDTTC}'''

class CurveCAP():
    """ Class for creating CAP Curve """

    def __init__(self, hyperparams):
        for key in hyperparams:
            setattr(self, key, hyperparams[key])

    def transform(self, X):
        # X_transformed = ut.cap_transform(X, self.param_PDTTC, self.)
        X_transformed = ut.cap_transform(X, self.param_capital)
        return X_transformed

    def predict(self, X):
        # X_predict = ut.cap_predict(X, self.param_scalar, self.)
        X_predict = ut.cap_predict(X,self.param_capital)
        return X_predict


# Curve Costos
'''p1 = Path(".").resolve() / "data" / "costos_canalT0.csv"
p2 = Path(".").resolve() / "data" / "costos_montoT0.csv"
p3 = Path(".").resolve() / "data" / "costos_otrosT0.csv"
p4 = Path(".").resolve() / "data" / "costos_t.csv"
param_canalT0 = pd.read_csv(str(p1))
param_montoT0 = pd.read_csv(str(p2))
param_otrosT0 = pd.read_csv(str(p3))
param_costost = pd.read_csv(str(p4))

hyperparams = {"param_canalT0": param_canalT0, "param_montoT0": param_montoT0,
               "param_otrosT0": param_otrosT0, "param_costost": param_costost}
'''

class CurveCostos():
    def __init__(self, hyperparams):
        for key in hyperparams:
            setattr(self, key, hyperparams[key])

    def transform(self, X):
        X = ut.costos_transform(X)
        return X
    def predict(self, X):
        X = ut.costos_predict(X, self.param_t0,self.param_t1T)
        return X

class CurvesInof():
    def __init__(self,hyperparams):
        for key in hyperparams:
            setattr(self, key, hyperparams[key])
    def transform(self, X):
        return(ut.inof_transform(X))
    
    def predict(self, X):
        return(ut.inof_predict(X, self.param_inof))

# Curve TT&Descuentos
'''p1 = Path(".").resolve() / "data" / "tt_descuentos_tasadescuento.csv" 
p2 = Path(".").resolve() / "data" / "tt_descuentos_tt.csv"
p3 = Path(".").resolve() / "data" / "tt_descuentos_remcap.csv"
p4 = Path(".").resolve() / "data" / "tt_descuentos_tasadescuento_leads.csv"
param_tasadescuento = pd.read_csv(str(p1))
param_tt = pd.read_csv(str(p2))
param_remcap = pd.read_csv(str(p3))
param_tasadescuento_leads = pd.read_csv(str(p4))
 
hyperparams = {"param_tasadescuento": param_tasadescuento,
               "param_tasadescuento_leads":param_tasadescuento_leads,
               "param_remcap": param_remcap,
               "param_tt": param_tt}
'''

class CurvesTT_Desc():
    def __init__(self, hyperparams):
        for key in hyperparams:
            setattr(self, key, hyperparams[key])

    def transform(self, X):
        return(ut.tt_desc_transform(X, self.param_prep_dur,self.param_td,self.param_remcap))
    
    def predict(self, X):
        return(ut.tt_desc_predict(X, self.param_tt))