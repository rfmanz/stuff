import numpy as np
import pandas as pd
from pathlib import Path
import source.engine.utils_clv as clv
import source.engine.utils as ut
import source.engine.curves_all as curves


class SetPath:
    def __init__(self, path="data"):
        self.path = path
        self.hyperparams = {}
        self.set_path_can()
        self.set_path_pd()
        self.set_path_pre()
        self.set_path_lgd()
        self.set_path_ecap()
        self.set_path_cost()
        self.set_path_tt()
        self.set_path_inof()
        self.set_path_mapProducts()
        self.set_path_mapSegRisk()
        self.set_path_mapInst()

    def set_path_can(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_Cancelaciones.csv"
        param_Can = pd.read_csv(
            str(p1),
            sep="|",
            float_precision="round_trip",
            dtype=np.float64,
            converters={"Producto": str},
        )
        hyperparams = {"param_Can": param_Can}
        self.hyperparams["CAN"] = hyperparams

    def set_path_pd(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_PD1.csv"
        p2 = Path(".").resolve() / self.path / "MiBanco_PD2.csv"
        param_pd1 = pd.read_csv(
            str(p1),
            sep="|",
            float_precision="round_trip",
            dtype=np.float64,
            converters={"Segmento De Riesgo": str, "Producto": str},
        )
        param_pd2 = pd.read_csv(
            str(p2),
            sep="|",
            float_precision="round_trip",
            dtype=np.float64,
            converters={"Segmento De Riesgo": str},
        )
        hyperparams = {"param_pd1": param_pd1, "param_pd2": param_pd2}
        self.hyperparams["PD"] = hyperparams

    def set_path_pre(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_Prepagos.csv"
        param_pre = pd.read_csv(
            str(p1),
            sep="|",
            float_precision="round_trip",
            dtype=np.float64,
            converters={"Segmento De Riesgo": str, "Producto": str},
        )
        hyperparams = {"param_pre": param_pre}
        self.hyperparams["PRE"] = hyperparams

    def set_path_lgd(self):
        lgd1 = Path(".").resolve() / self.path / "MiBanco_LGD.csv"
        param_lgd = pd.read_csv(str(lgd1), sep="|", float_precision="round_trip")
        hyperparams = {"param_lgd": param_lgd}
        self.hyperparams["LGD"] = hyperparams

    def set_path_ecap(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_Capital.csv"
        param_capital = pd.read_csv(str(p1), sep="|", float_precision="round_trip")
        hyperparams = {"param_capital": param_capital}
        self.hyperparams["CAP"] = hyperparams

    def set_path_cost(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_Costos.csv"
        p2 = Path(".").resolve() / self.path / "MiBanco_CostoServir.csv"
        param_costo_t0 = pd.read_csv(str(p1), sep="|")
        param_costo_t1T = pd.read_csv(str(p2), sep="|")
        hyperparams = {"param_t0": param_costo_t0, "param_t1T": param_costo_t1T}
        self.hyperparams["COST"] = hyperparams

    def set_path_tt(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_TT.csv"
        p2 = Path(".").resolve() / self.path / "MiBanco_DuracionPrepagos.csv"
        p3 = Path(".").resolve() / self.path / "MiBanco_TD.csv"
        p4 = Path(".").resolve() / self.path / "MiBanco_RemCap.csv"
        param_tt = pd.read_csv(str(p1), sep="|")
        param_prep_dur = pd.read_csv(str(p2), sep="|")
        param_td = pd.read_csv(str(p3))
        param_remcap = pd.read_csv(str(p4))
        hyperparams = {
            "param_tt": param_tt,
            "param_prep_dur": param_prep_dur,
            "param_td": param_td,
            "param_remcap": param_remcap,
        }
        self.hyperparams["TT_DSCTO"] = hyperparams

    def set_path_inof(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_Inof.csv"
        param_inof = pd.read_csv(str(p1), sep="|")
        hyperparams = {"param_inof": param_inof}
        self.hyperparams["INOF"] = hyperparams

    def set_path_mapProducts(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_MapProducts.csv"
        param_mapProd = pd.read_csv(str(p1), sep="|")
        hyperparams = {"param_mapProd": param_mapProd}
        self.hyperparams["MAPS_PRODUCTS"] = hyperparams

    def set_path_mapSegRisk(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_MapSegmentoRiesgo.csv"
        param_mapSegRisk = pd.read_csv(str(p1), sep="|")
        hyperparams = {"param_mapSegRisk": param_mapSegRisk}
        self.hyperparams["MAPS_SEG_RISK"] = hyperparams

    def set_path_mapInst(self):
        p1 = Path(".").resolve() / self.path / "MiBanco_MapPlazos.csv"
        param_mapInst = pd.read_csv(str(p1), sep="|")
        hyperparams = {"param_mapInst": param_mapInst}
        self.hyperparams["MAPS_INST"] = hyperparams

    def get_hyperparam(self):
        return self.hyperparams


class CLVEngine:
    def __init__(self, hyperparams=SetPath().get_hyperparam()):
        self.hyperparams = hyperparams
        self.m, self.param, self.ingresos, self.score = None, None, None, None
        self.curves = {
            "PD": None,
            "PRE": None,
            "CAN": None,
            "LGD": None,
            "CAP": None,
            "COST": None,
            "TT_DSCTO": None,
            "INOF": None,
        }
        self.X_tr = {
            "PD": None,
            "PRE": None,
            "CAN": None,
            "LGD": None,
            "CAP": None,
            "COST": None,
            "INOF": None,
            "TT_DSCTO": None,
            "descuentos": None,
            "desembolsos": None,
            "r": None,
            "T": None,
        }
        self.curves_instance = {
            "PD": curves.CurvePD(self.hyperparams["PD"]),  # limpiar sef
            "PRE": curves.CurvePre(self.hyperparams["PRE"]),
            "CAN": curves.CurveCan(self.hyperparams["CAN"]),
            "LGD": curves.CurveLGD(self.hyperparams["LGD"]),
            "CAP": curves.CurveCAP(self.hyperparams["CAP"]),
            "COST": curves.CurveCostos(self.hyperparams["COST"]),
            "TT_DSCTO": curves.CurvesTT_Desc(self.hyperparams["TT_DSCTO"]),
            "INOF": curves.CurvesInof(self.hyperparams["INOF"]),
        }

    def uniformize_input(self, X):
        X = X.copy()
        products = self.hyperparams["MAPS_PRODUCTS"]["param_mapProd"]
        segRisk = self.hyperparams["MAPS_SEG_RISK"]["param_mapSegRisk"]
        segInst = self.hyperparams["MAPS_INST"]["param_mapInst"]
        X = pd.merge(X, products, left_on="GRUPO0", right_on="real_name", how="left")
        X = pd.merge(X, segRisk, left_on="SEG_RIE", right_on="seg_risk", how="left")
        X = pd.merge(
            X, segInst, left_on="PLAZO_MESES", right_on="installment", how="left"
        )
        X["orden"] = np.arange(0, X.shape[0])
        return X

    def transform(self, X):
        """Uniformize the input data"""
        X = self.uniformize_input(X)
        """ Apply the transform methods of every curve
        """
        for key in self.curves_instance:
            self.X_tr[key] = self.curves_instance[key].transform(X)
        # Getting variables for NPV function
        (
            self.X_tr["descuentos"],
            self.X_tr["desembolsos"],
            self.X_tr["r"],
            self.X_tr["T"],
        ) = clv.get_parameters_transform(X, self.X_tr["TT_DSCTO"])
        self.m = X.shape[0]

        return self.X_tr

    def predict(self):
        """Apply the predict methods of every curve"""
        for key in self.curves:
            # if key=="CAP": # LGD 21
            #     self.X_tr[key].loc[:, "LGD_TTC"] = self.curves_instance["LGD"].lgd21
            self.curves[key] = self.curves_instance[key].predict(self.X_tr[key])
        return self.curves

    def get_parameters_pv(self):
        """This function is called by other functions in the class
        Returns: self.param, a parameter for any posterior VAN function
        """
        self.param = clv.get_param(self.X_tr, self.curves)

    def get_pv(self):
        """Returns a np.array containing the Net Present Value"""
        if self.param == None:
            self.get_parameters_pv()
        self.pv = clv.van_rmin(
            self.X_tr["descuentos"],
            self.X_tr["desembolsos"],
            self.X_tr["r"],
            self.X_tr["T"],
            self.param,
        )

        return self.pv.reshape(
            self.m,
        )

    def get_rmin(self):
        """Returns a np.array containing the T_min"""
        if self.param == None:
            self.get_parameters_pv()
        # def funct(x): return clv.van_rmin(
        def funct(x):
            return clv.van_cronograma(
                self.X_tr["descuentos"],
                self.X_tr["desembolsos"],
                x,
                self.X_tr["T"],
                self.param,
            ).reshape(
                self.m,
            )

        def compute():
            r0 = np.ones((1, self.m)) * 0.16 / 12
            tmin = clv.pbroyden(funct, r0)
            return tmin.reshape(
                self.m,
            )

        self.rmin = compute()
        return self.rmin

    def get_rmin_leads(self):
        """Returns a np.array containing the T_min"""
        if self.param == None:
            self.get_parameters_pv()

        def funct(x):
            return clv.van_rmin(
                self.X_tr["descuentos_leads"],
                self.X_tr["desembolsos"],
                x,
                self.X_tr["T"],
                self.param,
            ).reshape(
                self.m,
            )

        def compute():
            r0 = np.ones((1, self.m)) * 0.16 / 12
            tmin = clv.pbroyden(funct, r0)
            return tmin.reshape(
                self.m,
            )

        self.rmin = compute()
        return self.rmin

    def get_rmin_decomp(self, r):
        """
        Return a pandas dataframe containgin VAN Total, VAN Capital, VAN TT, VAN Perdida, VAN Costos, VAN IR,
        Saldo Promedio & Capital Promedio

        Arguments:
        r: Tmin
        """
        if self.param == None:
            self.get_parameters_pv()
        df_decomp = clv.get_rmin_van_decomp(
            self.X_tr["descuentos"],
            self.X_tr["desembolsos"],
            r,
            self.X_tr["T"],
            self.param,
        )

        return df_decomp

    def get_irr(self, r=[None]):
        """Returns a np.array containing the Internal Interest Rate
        Parameters:
        >>> r: Interest Rate of the loan
        """
        if self.param == None:
            self.get_parameters_pv()
        if r[0] == None:
            r = self.X_tr["r"]

        def funct(x):
            return clv.van_cronograma(
                x, self.X_tr["desembolsos"], r, self.X_tr["T"], self.param
            ).reshape(
                self.m,
            )

        def compute():
            d0 = np.ones((1, self.m)) * 0.30 / 12
            tirs = clv.pbroyden(funct, d0)
            return tirs.reshape(
                self.m,
            )

        # def compute():
        #    from scipy import optimize
        #    d0 = np.ones((1,self.m))*0.30/12
        #    tirs = optimize.broyden1(funct, d0,maxiter=30)
        #    return tirs.reshape(self.m,)

        self.irr = compute()
        return self.irr

    def get_schedule(self, i):
        """Returns a pandas dataframe with the schedule for the i-th observation
        Arguments:
        >>> i: Observation to report
        """
        if self.param == None:
            self.get_parameters_pv()
        schedules = clv.get_schedule(
            self.X_tr["descuentos"],
            self.X_tr["desembolsos"],
            self.X_tr["r"],
            self.X_tr["T"],
            self.param,
            i,
        )

        return schedules

    def get_decomposition_leads(self, tasas_propuestas, rmins_mensual):
        """Returns: pandas dataframe with profitability indicators,
        >>> CODCLAVECIC, TIR, Tasa Propuesta, Rmin, Rmin decomposition, Average Capital, Average Balance, Desembolsos, TIR ponderada, Rmin ponderada,
            Rango Score, Rango Ingresos, Flag PDH, Flag Risky
        """
        if self.param == None:
            self.get_parameters_pv()

        tasas_mensual = (
            np.power((1 + tasas_propuestas), 1 / 12) - 1
        )  # Mensualizando la tasa propuesta
        rmins = np.power((1 + rmins_mensual), 12) - 1  # Anualizando la rmin mensual
        tirs_mensual = self.get_irr(r=tasas_mensual)  # TIR mensual
        tirs = np.power((1 + tirs_mensual), 12) - 1  # Se anualiza la TIR
        rmin_decomp_df = self.get_rmin_decomp(
            r=rmins_mensual
        )  # Recibe una rmin mensual

        flag_pdh = self.flag_pdh
        flag_risky = (rmins > 0.349).astype(int)

        df = pd.DataFrame()
        df.loc[:, "CODCLAVECIC"] = self.cliente
        df.loc[:, "TIR"] = tirs  # TIR en base a tasa propuesta
        df.loc[:, "Tasa Minima"] = rmins
        df.loc[
            :, "Tasa Propuesta"
        ] = tasas_propuestas  # Tasas Propuestas luego de aplicar reglas de negocio
        df = pd.concat(
            [df, rmin_decomp_df], axis=1
        )  # Descomposicion de la Tmin + Saldo y Capital Promedio
        df.loc[:, "Desembolsos"] = self.X_tr["desembolsos"]
        df.loc[:, "Tasas Ponderadas"] = (
            df.loc[:, "Desembolsos"].values * tasas_propuestas
        )  # Tmin Ponderada por Desembolsos
        df.loc[:, "TIR Ponderada"] = (
            df.loc[:, "Capital Promedio"].values * tirs
        )  # TIR Ponderada por Capital Promedio
        df.loc[:, "Rango Ingresos"] = ut.transf_label_rango_ingreso(self.ingresos)
        df.loc[:, "Rango Score"] = ut.transf_label_rango_score(self.score_post_covid)
        df.loc[:, "Flag PDH"] = flag_pdh
        df.loc[:, "Tmin>34.9%"] = flag_risky  # Flag muy riesgoso (Cap en el tope)

        return df

    def get_profitability_indicators(self):
        """Returns: pandas DataFrame with profitability indicators
        >>> VAN, TIR, Rmin, Rmin decomposition, Average Capital, Average Balance
        """
        if self.param == None:
            self.get_parameters_pv()

        pvs = self.get_pv()
        tirs = self.get_irr()  # TIR mensual
        rmins = self.get_rmin()  # Tmin mensual
        rmin_decomp_df = self.get_rmin_decomp(rmins)  # Recibe una rmin mensual

        tirs = np.power((1 + tirs), 12) - 1  # Se anualiza la TIR
        rmins = np.power((1 + rmins), 12) - 1  # Se anualiza la Tmin

        pvs_df = pd.DataFrame(pvs, columns=["VAN"])  # VAN
        tirs_df = pd.DataFrame(tirs, columns=["TIR"])  # TIR
        rmin_df = pd.DataFrame(rmins, columns=["RMIN"])  # Tmin
        df = pd.concat(
            [pvs_df, tirs_df, rmin_df, rmin_decomp_df], axis=1
        )  # Concatenando los DataFrames

        return df

    def get_components(self, dict_comp, r_calc):
        if self.param == None:
            self.get_parameters_pv()

        if r_calc == "base":
            r = self.X_tr["r"]
        elif r_calc == "tmin":
            r = self.rmin

        df = clv.get_components(
            self.X_tr["descuentos"],
            self.X_tr["desembolsos"],
            r,
            self.X_tr["T"],
            self.param,
            dict_comp,
        )  # .reshape(self.m,)
        return df

    def get_components_validation(self, dict_comp, r_calc):
        if self.param == None:
            self.get_parameters_pv()

        if r_calc == "base":
            r = self.X_tr["r"]
        elif r_calc == "tmin":
            r = self.rmin

        df = clv.get_components_validation(
            self.X_tr["descuentos"],
            self.X_tr["desembolsos"],
            r,
            self.X_tr["T"],
            self.param,
            dict_comp,
        )
        return df

    def get_components_validation2(self, dict_comp, r_calc):
        if self.param == None:
            self.get_parameters_pv()

        if r_calc == "base":
            r = self.X_tr["r"]
        elif r_calc == "tmin":
            r = self.rmin

        df = clv.get_components_validation2(
            self.X_tr["descuentos"],
            self.X_tr["desembolsos"],
            r,
            self.X_tr["T"],
            self.param,
            dict_comp,
        )
        return df

    def get_report(self, dict_com):
        df1 = self.get_components(dict_com, "base")
        tir_mensual = self.get_irr()
        tir_anual = (1 + tir_mensual) ** 12 - 1
        df1["tir"] = tir_anual
        df1["periodo"] = self.X_tr["CAN"]["PERIODO"].tolist()
        df1["codigo_cliente"] = self.X_tr["CAN"]["CO_CLIENTE"].tolist()
        df1["numero_prestamo"] = self.X_tr["CAN"]["NU_PRESTAMO"].tolist()
        self.get_rmin()
        df2 = self.get_components(dict_com, "tmin")
        tir_mensual = self.get_irr(self.rmin)
        tir_anual = (1 + tir_mensual) ** 12 - 1
        df2["tir"] = tir_anual
        df2["periodo"] = self.X_tr["CAN"]["PERIODO"].tolist()
        df2["codigo_cliente"] = self.X_tr["CAN"]["CO_CLIENTE"].tolist()
        df2["numero_prestamo"] = self.X_tr["CAN"]["NU_PRESTAMO"].tolist()
        return df1, df2

    def get_report_validation(self, dict_com):
        df1 = self.get_components_validation(dict_com, "base")
        tir_mensual = self.get_irr()
        tir_anual = (1 + tir_mensual) ** 12 - 1
        df1["tir"] = tir_anual
        df1["periodo"] = self.X_tr["CAN"]["PERIODO"].tolist()
        df1["codigo_cliente"] = self.X_tr["CAN"]["CO_CLIENTE"].tolist()
        df1["numero_prestamo"] = self.X_tr["CAN"]["NU_PRESTAMO"].tolist()
        return df1

    def get_report_validation2(self, dict_com):
        df1 = self.get_components_validation2(dict_com, "base")
        tir_mensual = self.get_irr()
        tir_anual = (1 + tir_mensual) ** 12 - 1
        df1["tir"] = tir_anual
        df1["periodo"] = self.X_tr["CAN"]["PERIODO"].tolist()
        df1["codigo_cliente"] = self.X_tr["CAN"]["CO_CLIENTE"].tolist()
        df1["numero_prestamo"] = self.X_tr["CAN"]["NU_PRESTAMO"].tolist()
        return df1
