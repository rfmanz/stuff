# Script para obtener la Tasa Interna de Retorno (TIR) y Tea Mínima (TMIN)
import os

import pandas as pd
import numpy as np
import source.engine.CLV_Engine as pricing
import source.engine.utils as ut
from pathlib import Path
import time

import csv

# 0.5 validacion input
# -----------------------------------------------------------------------------------------------------------
# required_columns = ['PERIODO', 'CO_CLIENTE', 'NU_PRESTAMO', 'TIPO_CLIENTE', 'GRUPO0', 'CO_MONEDA', 'FE_DESEMBOLSO', 'MO_DESEMBOLSO_SOLES', 'SEG_RIE', 'CANAL', 'PLAZO_MESES', 'TEA']
required_columns = [
    "PERIODO",
    "CO_CLIENTE",
    "NU_PRESTAMO",
    "TIPO_CLIENTE",
    "GRUPO0",
    "CO_MONEDA",
    "FE_DESEMBOLSO",
    "MO_DESEMBOLSO_SOLES",
    "SEG_RIE",
    "CANAL",
    "PLAZO_MESES",
    "TEA",
    "TIR_OBJ",
]

# Detect delimiter
home = "D:/CLV_Original/data/input/"


######################################

data_filename = "CDD Express v2.csv"
######################################

full_filename = Path(".").resolve() / "data" / "input" / data_filename

full_path = home + data_filename

# Delimiter
sample_bytes = 64
dialect = csv.Sniffer().sniff(open(full_filename).read(sample_bytes), delimiters="|,;")
print(dialect.delimiter)

# Count rows
reader = csv.reader(open(full_filename))
lines = len(list(reader))
print(lines)

# Detect headers
d_reader = csv.DictReader(open(full_path), dialect=dialect)
headers = d_reader.fieldnames
headers
set(headers) == set(required_columns)

###############################################################################################################
# 1. PARAMETROS (definidos por usuario)
# -----------------------------------------------------------------------------------------------------------
codigo_modelo = "MOD-MIB-0002766"
nombre_usuario = "RFA"
matricula_usuario = "89894001"
# data_filename = full_path
n_rows = lines
chunk_size = 5000
# n_rows = 500
# chunk_size = 500
output_filename = "get_components_" + data_filename


dict_comp = [
    "saldo",
    "tea",
    "tt",
    "saldo",
    "capital",
    "ingreso_financiero",
    "costo_financiero",
    "perdida_esperada",
    "rem_capital",
    "desgravamen",
    "costos",
    "imp_renta",
    "flujo_neto",
    "van",
    "tir",
    "roe",
]

t = time.time()


#########################################################################################################
# 2. DATA
# -----------------------------------------------------------------------------------------------------------

inputs = pd.read_csv(
    full_filename, nrows=n_rows, sep=dialect.delimiter, encoding="ISO-8859-1"
)
# inputs_it = pd.read_csv(full_filename, nrows=n_rows, chunksize=chunk_size, iterator=True,sep=dialect.delimiter)
# ut.log(cod_modelo = codigo_modelo,autor=nombre_usuario,matricula=matricula_usuario,obs=inputs,funcion="get_components")

#########################################################################################

# 2.5 VALIDACIONES
# -----------------------------------------------------------------------------------------------------------
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

inputs2 = inputs[inputs["NU_PRESTAMO"].notna()]
if (
    is_string_dtype(inputs2.MO_DESEMBOLSO_SOLES)
    and inputs2.MO_DESEMBOLSO_SOLES.str.find("%").all()
):
    inputs2.MO_DESEMBOLSO_SOLES = inputs2.MO_DESEMBOLSO_SOLES.str.replace(",", "")
    inputs2.MO_DESEMBOLSO_SOLES = inputs2.MO_DESEMBOLSO_SOLES.astype(int)
if not (
    is_numeric_dtype(inputs2.PERIODO) and inputs2.PERIODO.astype(str).str.len() == 6
).all():
    print("ERROR EN PERIODO")
inputs2 = inputs2.drop(columns=inputs2.columns.difference(required_columns))
inputs2.CO_CLIENTE.fillna(inputs2.index.to_series() + 1, inplace=True)
inputs2.CO_CLIENTE = inputs2.CO_CLIENTE.astype(int)
if is_string_dtype(inputs2.TEA) and inputs2.TEA.str.find("%").all():
    inputs2.TEA = (inputs2.TEA.str[:-1].astype(float)) / 100
inputs2.FE_DESEMBOLSO.fillna(
    pd.to_datetime(inputs.PERIODO, format="%Y%m"), inplace=True
)
# pd.to_datetime(inputs.PERIODO, format='%Y%m').dt.strftime('%Y-%m-%d')
if not inputs2.TIPO_CLIENTE.isin(["RECURRENTE", "NUEVO"]).all():
    print("ERROR EN TIPO_CLIENTE")
if not inputs2.GRUPO0.isin(
    [
        "Activo Fijo",
        "Capital de Trabajo",
        "RURAL",
        "Hipotecario",
        "Construccion de Vivienda",
        "Lineas",
        "Consumo",
        "Compra Deuda",
    ]
).all():
    print("ERROR EN GRUPO0")
if not (inputs2.CO_MONEDA == 1).all():
    print("ERROR EN CO_MONEDA")
if not (inputs2.SEG_RIE).isin(["RMB", "RMB", "RB", "RM", "RA", "RMA"]).all():
    print("ERROR EN SEG_RIE")
if not (inputs2.CANAL).isin(["ADN", "VGA", "CALL"]).all():
    print("ERROR EN CANAL")

# if is_string_dtype(inputs2.NU_PRESTAMO) and inputs2.NU_PRESTAMO.str.find('%').all():
#     inputs2.NU_PRESTAMO = inputs2.NU_PRESTAMO.str.replace('%',"")
#     inputs2.NU_PRESTAMO = inputs2.NU_PRESTAMO.astype(float).astype(int)

inputs2.PERIODO = inputs2.PERIODO.astype(int)
inputs2 = inputs2.iloc[:500]

# -----------------------------------------------------------------------------------------


def peek(df, rows=3):
    concat1 = pd.concat([df.dtypes, df.iloc[:3, :].T], axis=1).reset_index()
    concat1.columns = [""] * len(concat1.columns)
    return concat1


def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        # if verbose and best_type is not None and best_type != str(col_type):
        # print(f"Column {col} converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(
            f"Memory usage decreased from"
            f" {start_mem:.2f}MB to {end_mem:.2f}MB"
            f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)"
        )


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB."""
    return df.memory_usage(*args, **kwargs).sum()


# ----------------------------------------------------------------------------------------------

inputs2 = inputs
inputs2.to_csv("D:/CLV_Original/data/input/" + data_filename + "_clean.csv")
inputs_it = pd.read_csv(
    "D:/CLV_Original/data/input/" + data_filename + "_clean.csv",
    nrows=n_rows,
    chunksize=chunk_size,
    iterator=True,
)
inputs = inputs2
# for inputs in inputs_it:
#     print(inputs)
reduce_memory_usage(inputs)


# -----------------------------------------------------------------------------------------------------------
df1_list = []
df2_list = []
inputs_orig = inputs
n_chunk = 0
for inputs in inputs_it:
    n_chunk = n_chunk + 1
    print("---------CHUNK " + str(n_chunk) + "--------")
    clv = pricing.CLVEngine()
    # inputs_tratado = ut.tratamiento_data_leads(inputs)
    clv.transform(inputs.reset_index(drop=True))
    clv.predict()
    # df = clv.get_components(dict_comp,r)
    df1, df2 = clv.get_report(dict_comp)
    df1_list.append(df1)
    df2_list.append(df2)
df1_total = pd.concat(df1_list, ignore_index=True)
df2_total = pd.concat(df2_list, ignore_index=True)
# AGREGAMOS EL CODIGO DEL MODELO AL OUTPUT
df1_total["CODIGO_MODELO"] = codigo_modelo
df2_total["CODIGO_MODELO"] = codigo_modelo

# EXPORTANDO RESULTADOS
output_filename1 = "tir_" + output_filename
output_filename2 = "tmin_" + output_filename
full_filename_output1 = Path(".").resolve() / "data" / "output" / output_filename1
full_filename_output2 = Path(".").resolve() / "data" / "output" / output_filename2
df1_total.to_csv(full_filename_output1, index=False)
df2_total.to_csv(full_filename_output2, index=False)

elapsed = time.time() - t
print(elapsed)

# BACKUP INPUTS AND OUTPUTS
# path_input  = "\\\\domibco.com.pe\\fs\\PRI_FINANZAS\\6_G_CTRL_FINANCIERO\\Control Financiero\\24. CLV\\CLV1.0-backup\\input\\"
# path_output = "\\\\domibco.com.pe\\fs\\PRI_FINANZAS\\6_G_CTRL_FINANCIERO\\Control Financiero\\24. CLV\\CLV1.0-backup\\output\\"

# inputs_orig.to_csv(path_input + data_filename,sep = "|")
# df1_total.to_csv(path_output + output_filename1,sep = "|")
# df2_total.to_csv(path_output + output_filename2,sep = "|")
# print("Backup realizado...")


# def costos_transform(X):
#     X = X.copy()
#     #rango de monto para costo de venta
#     X["Rango"] = [0]*X.shape[0]
#     X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=0 ,X.MO_DESEMBOLSO_SOLES<5000),"Rango"] = 1
#     X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=5000 ,X.MO_DESEMBOLSO_SOLES<10000),"Rango"] = 2
#     X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=10000 ,X.MO_DESEMBOLSO_SOLES<20000),"Rango"] = 3
#     X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=20000 ,X.MO_DESEMBOLSO_SOLES<40000),"Rango"] = 4
#     X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=40000 ,X.MO_DESEMBOLSO_SOLES<60000),"Rango"] = 5
#     X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=60000 ,X.MO_DESEMBOLSO_SOLES<80000),"Rango"] = 6
#     X.loc[np.logical_and(X.MO_DESEMBOLSO_SOLES>=80000 ,X.MO_DESEMBOLSO_SOLES<100000),"Rango"] = 7
#     X.loc[X.MO_DESEMBOLSO_SOLES>=100000,"Rango"] = 8
#     return X


# costos_transform(inputs)

# import subprocess
# import shlex
# command = "python test2.py --option1 -dir D:/CLV/test2.py"
# args = shlex.split(command)
# my_subprocess = subprocess.Popen(args)

# import sys
# print(sys.version)
