# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
from pathlib import Path
# Functions
import warnings
warnings.filterwarnings("ignore")
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def Bfactor2(rate, T, time=None, Tmax=60):
    rate = rate.reshape(1, -1)
    # Input validation
    _, m = rate.shape
    _, m1 = T.shape
    assert (m == m1), "rate and T should be of the same shape"
    T_m = np.max(T)
    if Tmax < T_m:
        Tmax = T_m
    # Building inputs
    # t = np.repeat(np.arange(1,Tmax+1).reshape(-1,1),m,axis=1)
    t = np.arange(1, Tmax + 1).reshape(-1, 1)
    # r = np.repeat(rate,Tmax,axis=0)
    r = rate
    # T = np.repeat(T,Tmax,axis=0)
    # Vectorized computation
    arr = (np.power(1 + r, T) - np.power(1 + r, t - 1)) / \
        (np.power(1 + r, T) - 1)
    return np.maximum(arr, 0)

def Afactor1(rate, T, time=None, Tmax=241):
    t = np.arange(1, Tmax + 1).reshape(-1, 1)
    Den = np.power(1 + rate, T-t+1) - 1
    Af = np.divide(rate , Den, out=np.ones_like(Den),where=(Den!=0))
    Af[(Af==np.inf)|(np.isnan(Af))]=0
    return np.maximum(Af, 0)

def Bf1(t, r, T):
    return ((1 + r)**T - (1 + r)**(t - 1)) / ((1 + r)**T - 1)

def Bfactor1(r, T, Tmax=241):
    t = np.arange(1, Tmax + 1).reshape(-1, 1)  # shape (T,1)
    Bf = np.maximum(np.array([Bf1(time, rate, Tm)
                              for time in t for rate in r for Tm in T]), 0)
    return Bf

def shift(arr, num, fill_value=0):
    """
    Shifts a numpy array of shape (T,m) num times filling fill_values to the missing slots

    Parameters
    ----------
    arr : array_like of shape = (T,m)
        a numpy array containing T observations for m samples
    num : scalar indicating the number of lags (if positive) or leads (if negative) to shift the numpy array verticaly

    fill_value: scalar indicating shift numpy array {1}

    Returns
    -------
    arr : array_like of shape = (T,m)
        a numpy array containing shifted T observations for m samples

    Examples
    -------
    np.random.seed(42)
    test_array = np.random.randn(3,5)
    test_array
    shift(test_array,2)

    Returns

    array( [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337]])


    shift(test_array,1)

    Returns
    array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337],
       [-0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004]])

    shift(test_array,-1)

    Returns
    array([[-0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004],
       [-0.46341769, -0.46572975,  0.24196227, -1.91328024, -1.72491783],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def cif_to_cond_prob(cif1, cif2, s1=1, s2=1):
    """
    Converts two CIFs (Cumulative Incidence Functions) to conditional probabilities of death.
    T is the number of periods, m is the number of samples.

    Parameters
    ----------
    cif1 : array_like of shape = (T,m)
        a numpy array containing event 1 CIF curve in each column.
    cif2 : array_like of shape = (T,m)
        a numpy array containing event 1 CIF curve in each column.
    s1: scalar indicating shift of cif1 curves {1}
    s2: scalar indicating shift of cif1 curves {1}

    Returns
    -------
    (conditional_prob_1, conditional_prob_2)
        2-tuple containing one numpy array in each element.
        conditional_prob_1 contains pd curves for the first CIF
        conditional_prob_2 contains pd curves for the first CIF

    Example
    -------
    cif_PD = np.cumsum(np.random.uniform(0,high=1,size=(10,5)),axis=0)/10
    cif_CAN = np.cumsum(np.random.uniform(0,high=1,size=(10,5)),axis=0)/10
    PD , CAN = cif_to_cond_prob(cif_PD,cif_CAN,s1=1,s2=3)
    """
    cif1_shifted = shift(cif1, s1)
    # cif2_shifted = shift(cif2, s2)
    # den = (1-cif1_shifted-cif2_shifted)   # Descomentar esta linea si el clv
    # se corrige l modelo estandar
    # Comentar esta linea si el clv se corrige al modelo estandar
    den = (1 - cif1_shifted - cif2)
    # cond_prob2 = (cif2-cif2_shifted)/den  # Descomentar esta linea si el clv
    # se corrige l modelo estandar
    # cond_prob2 = 0  # Comentar esta linea si el clv se corrige al modelo estandar
    cond_prob1 = np.clip((cif1 - cif1_shifted) / den, 0, 1)
    b = np.isnan(cond_prob1)
    cond_prob1[b]=0
    return (cond_prob1, 0)

def compute_S(PD, CAN, PRE, Bf):
    """
    Computes Survival curve and laggs it one period S(t-1)

    Parameters
    ----------
    PD : array_like of shape = (T,m)
        a numpy array containing probability of default (conditional to survival in t-1). T periods in each rows (1...T) and m samples in each column (1 ...m).
    CAN : array_like of shape = (T,m)
        a numpy array containing probability of full prepayment (conditional to survival in t-1). T periods in each rows (1...T) and m samples in each column (1 ...m).
    PRE : array_like of shape = (T,m)
        a numpy array containing probability of prepayment (conditional to survival in t-1). T periods in each rows (1...T) and m samples in each column (1 ...m).
    Bf: array_like of shape = (T,m)
        a numpy array containing Bf factors. T periods in each rows (1...T) and m samples in each column (1 ...m). Row 1 contains only 1"s


    Returns
    -------
    S_(t-1)
       numpy array in each element (T rows and m columns)

    Example
    -------


    """
    # Pseudo survival function
    Stilde = np.cumprod((1 - PD) * (1 - CAN), axis=0)
    Stilde_m1 = shift(Stilde, 1, 1)

    # Prepayment survival function
    PRE_m1 = shift(PRE, 1, 0)
    Den = Stilde_m1 * Bf
    C = np.divide(PRE_m1 , Den,out=np.zeros_like(Den),where=(Den!=0))

    Cstar = np.cumsum(C, axis=0)
    PRE_bar = np.maximum(np.minimum(1 - Cstar, 1), 0)
    # Survival Function
    S = Stilde_m1 * PRE_bar
    return S

def pbroyden(
        func,
        x0=0.01,
        eps=0.000001,
        tol=0.00001,
        max_iter=30,
        verbose=True):
    """
    Pseudo Broyden Method: Algorithm to find roots for a multivariate function with multivariate outputs.

    Parameters
    ----------
    func : function (Rn to Rn) (If used to compute Economic Profit (at present value) usually inputs a vector of m samples of rates for initial values and outputs a m-vector of Economic profits)

    x0 : array_like of shape (m,) that inputs initial guess for rates
    eps : step to vary arguments to optimize func
    tol : tolerance number for diference between 0 and the actual solution value output
    max_iter: maximun of iterations to use
    verbose: If set to true prints # of iterations

    Returns
    -------
    arr : array_like of shape = (m,) containing optimun solution (IRR if used to compute roots of Economic Profit)

    Examples
    -------

    1) Simulation of data
    np.random.seed(1)
    # Datos del producto
    m = 1_000 # Number of samples
    desembolsos = np.random.randint(10000,11000,(1,m))  # shape (1,m)
    T = np.random.choice([12, 24, 36, 48, 60 ],m).reshape(1,-1) # shape (1,m)
    Tmax = np.max(T)
    descuentos = 0.19/12*np.ones((1,m)).reshape(1,-1)
    # Curvas del CLENTE para este producto
    cif_PD = np.cumsum(np.random.uniform(
        0,high=1,size=(Tmax,m)),axis=0)/Tmax/3  # shape (T, m)
    cif_CAN = np.cumsum(np.random.uniform(
        0,high=1,size=(Tmax,m)),axis=0)/Tmax/3 # shape (T, m)
    PRE = np.cumsum(np.random.uniform(0,high=1,size=(Tmax,m)),
                    axis=0)/Tmax/30 # shape (T, m)
    PD , CAN = cif_to_cond_prob(cif_PD,cif_CAN,s1=1,s2=1)

    2) defining function to optimize

    3) Solving for roots with Pseudo Broyden"s Algorithm
    r0 = np.random.uniform(0.2/12,.30/12,(1,m)).reshape(1,-1)   # shape (1,m)
    param = {"PD":PD,"CAN":CAN,"PRE":PRE}
    funct = lambda x: van(descuentos,desembolsos,x,T,param).reshape(m,)
    def test2():
        tmin = pbroyden(funct, r0.reshape(m,))
        return tmin.reshape(m,)

    """
    iter_counter = 0
    error = 10
    xm = x0
    xp = xm + eps
    fem = func(xm)
    fep = func(xp)
    m = fem.shape[0]
    delta_error = 10
    solved = False
    solved_operations = 0
    counter_solved = 0
    mask_solved = np.array([False]*m).reshape(m,)
    mask_reshape_solved = np.array([False]*m).reshape(1,m)
    while ~( (error < tol) | (iter_counter > max_iter)| (delta_error<0.00001)|((counter_solved==5)&(solved_operations>0.90*m))):
    # while ~( (error < tol) | (iter_counter > max_iter)| (delta_error<0.0001)):
        
        swap = np.abs(fem)>np.abs(fep)
        xmtemp = xm[0,swap]
        xm[0,swap] = xp[0,swap] 
        xp[0,swap] = xmtemp

        femtemp = fem[swap]
        fem[swap] = fep[swap] 
        fep[swap] = femtemp

        iter_counter += 1
        Den = (xp[~mask_reshape_solved]-xm[~mask_reshape_solved])
        fprime = np.divide((fep[~mask_solved] - fem[~mask_solved]), Den, out=np.ones_like(Den), where=(Den!=0))
        xp[~mask_reshape_solved] = xm[~mask_reshape_solved]
        fep[~mask_solved] = fem[~mask_solved]
        
        xm[~mask_reshape_solved] = xm[~mask_reshape_solved] - fem[~mask_solved] / fprime
        fem = func(xm) # xm[~mask_reshape_solved] + mask output function
        
        solved = (np.abs(fem)<tol)
        mask_solved[solved] = True
        mask_reshape_solved = mask_solved.reshape(1,m)

        error_prev = error
        error = np.nansum(np.abs(fem[~mask_solved]))
        delta_error = abs(error - error_prev)

        #q = [0,0.01,0.05,0.25,0.5,0.95,0.99,1]
        #distrib = lambda x:  np.nanquantile(x,q)
        #num_nan = np.count_nonzero(np.isnan(fem))
        solved_operations_prev = solved_operations
        solved_operations = np.count_nonzero(solved)
        if solved_operations==solved_operations_prev:
            counter_solved+=1 
        else:
            counter_solved=0
         
        # print('________________________________________________ ')
        # print('iteration:', iter_counter) 
        # print('q:', q) 
        # print('# nans:', num_nan)
        # print('# solved_operations:', solved_operations)
        # print('Error:', error, 'Delta:', delta_error)
        # print('Van distrib:', distrib(fem))
        # print('x distrib:', distrib(xm))
        # print('fprime distrib:', distrib(fprime))

    if (iter_counter > max_iter):
        print("Maximun of iterations reached")
    elif verbose:
        print("Solution Found in {} iterations!".format(iter_counter))

    return xm

def van_cronograma(descuentos, desembolsos, r, T, param):
    """
    Parameters:
    >>> r: tasa de la iteración previa
    >>> descuentos: TIR Objetivo
    >>> desembolsos: Desembolsos efectivos
    >>> T: Plazos
    >>> param: Diccionario de parametros del clv.get_parameters_pv()

    Returns:

    """
    cif_PD = param["cif_PD"]
    cif_CAN = param["cif_CAN"]
    cum_PRE = param["cum_PRE"]
    LGD = param["LGD"]
    rc = param["rc"]
    ECf = param["ECf"]
    RemCap = param["RemCap"]
    IR = param["IR"]

    r = r.reshape(1, -1)
    

    # Comentar si CLV es estándar
    PD, _ = cif_to_cond_prob(
        shift(cif_PD, -2), shift(cif_CAN, 1), 1, 1)
    #PD, _ = cif_to_cond_prob(
    #    shift(cif_PD, -1), shift(cif_CAN, 1), 1, 1)
    #PD = shift(PD,-1)
    CAN, _ = cif_to_cond_prob(shift(cif_CAN, 0), shift(
        cif_PD, -2), 1, 1)  # Comentar si CLV es estándar
    # PD, CAN = cif_to_cond_prob(cif_CAN,cif_PD,1,1) # Descomentar si CLV es estándar
    PRE = np.maximum(cum_PRE - shift(cum_PRE, 1),0)

    descuentos = descuentos.reshape(1, -1)

    # Bbar: Contractual Balance
    #Bf = Bfactor2(r, T, Tmax=241)
    Bf = Bfactor2(r, T, Tmax=240)
    Bbar = np.multiply(Bf, desembolsos)

    # Survival Curves
    Sm1 = compute_S(PD, CAN, PRE, Bf)

    # B: Behavioral Balance
    B = np.maximum(Sm1*Bbar, 0)

    # I: Interest Income from Lending
    B_m_D = B*(1-PD)
    I = B_m_D*r  # <-- Interest Income

    #P = np.minimum(np.multiply(PRE, B[0, :]),B_m_D)
    ###1
    D = PD*B  # <-- Default
    # D = PD*shift(B,-1)
    #A = np.multiply(Afactor1(r, T, Tmax=241), B_m_D)
    A = np.multiply(Afactor1(r, T, Tmax=240), B_m_D)
    ###1
    P = np.minimum(np.multiply(PRE, B[0, :]),B_m_D)*np.logical_not(B_m_D-A==0)

    C = CAN*(B_m_D - A) # <-- Cancelations

    # EF: Interest Outcome from Cost of Funds
    D_ef = shift(D, 2)
    B_ef_aux = B[0, :]-np.cumsum(D_ef+A+C+P, axis=0)
    B_ef = np.maximum(np.concatenate([B[0, :].reshape(1, -1), B_ef_aux[0:-1, :]], axis=0), 0)
    #agregado
    B_ef[B_ef<0.05]=0
    
    COF = -B_ef*rc  # <-- Cost Of Funds (Interest Outcome)
    EL = -LGD*D_ef  # <-- Expected Loss
    EC = B_ef*ECf  # <-- Economic Capital
    RemEC = EC*RemCap  # <-- Remuneration to Economic Capital
    # Economic Capital Flow (Variation)
    EC_flow = -np.diff(EC, axis=0, append=0)
    EC_0 = -EC[0, :].reshape(1, -1)

    # C: Costs
    costos = param["COSTOS_VIDA"].values.reshape(1,-1)
    CH = np.maximum((1-cif_PD-cif_CAN), 0)  # Cuentas Hábiles

    #costos_total = costos*CH  # Costos Mantenimiento cuentas
    costos_total = -costos*CH*(B_m_D>0)  # Costos Mantenimiento cuentas
   
    
    
    # Costos Totales periodo 1-T / Comentar si se incorporan costos indirectos
    #B1 = shift(B,-1,0)
    # costos_total = -(costos_total)*(B_m_D>0)
    costos_T0 = -param["COSTOS_T0"]  # Costos en el periodo t=0

    # Ingresos No Financieros (INOF)
    # inof_t0 = param["INOF_T0"]
    inof_t0 = np.transpose(param["INOF_T0"])
    #Amort_c = Bbar-shift(Bbar, -1)  # Contractual Amortization (TxM)
    #I_c = Bbar*r  # Contractual Interest
    # Cuota = Amort_c[0, :] + I_c[0, :]  # Cuota (1xM)

    # Taxes (Impuesto a la Renta)
    tax_0 = -(costos_T0 + inof_t0)*IR
    
    #taxes = -(I+COF+EL+RemEC+costos_total)*IR
    taxes = -(I+COF+EL+RemEC+costos_total)*IR
    # Computing Cash Flow
    #CF = np.maximum(I + COF + EL + RemEC + EC_flow  + costos_total + taxes,0)
    CF = I + COF + EL + RemEC + EC_flow  + costos_total + taxes

    # Computing Net Present Value
    Tmax = CF.shape[0]
    descuento_matrix = np.array([np.power(1/(1+descuento), t)
                                 for descuento in descuentos for t in np.arange(1, Tmax+1).reshape(-1, 1)])
    # Discounted Cash Flows for simple sum
    CF_discounted = np.multiply(descuento_matrix, CF)
    CF_0 = tax_0 + EC_0 + costos_T0 + inof_t0  # Cash Flow in t=0
    pv = np.sum(CF_discounted, axis=0).reshape(1, -1) + CF_0  # Net Present Value

    return pv

def get_components(descuentos, desembolsos, r, T, param,dict_comp):
    """
    Parameters:
    >>> r: tasa de la iteración previa
    >>> descuentos: TIR Objetivo
    >>> desembolsos: Desembolsos efectivos
    >>> T: Plazos
    >>> param: Diccionario de parametros del clv.get_parameters_pv()

    Returns:

    """
    cif_PD = param["cif_PD"]
    cif_CAN = param["cif_CAN"]
    cum_PRE = param["cum_PRE"]
    LGD = param["LGD"]
    rc = param["rc"]
    ECf = param["ECf"]
    RemCap = param["RemCap"]
    IR = param["IR"]

    r = r.reshape(1, -1)
    

    # Comentar si CLV es estándar
    PD, _ = cif_to_cond_prob(
        shift(cif_PD, -2), shift(cif_CAN, 1), 1, 1)
    #PD, _ = cif_to_cond_prob(
    #    shift(cif_PD, -1), shift(cif_CAN, 1), 1, 1)
    #PD = shift(PD,-1)
    CAN, _ = cif_to_cond_prob(shift(cif_CAN, 0), shift(
        cif_PD, -2), 1, 1)  # Comentar si CLV es estándar
    # PD, CAN = cif_to_cond_prob(cif_CAN,cif_PD,1,1) # Descomentar si CLV es estándar
    PRE = np.maximum(cum_PRE - shift(cum_PRE, 1),0)

    descuentos = descuentos.reshape(1, -1)

    # Bbar: Contractual Balance
    #Bf = Bfactor2(r, T, Tmax=241)
    Bf = Bfactor2(r, T, Tmax=240)
    Bbar = np.multiply(Bf, desembolsos)

    # Survival Curves
    Sm1 = compute_S(PD, CAN, PRE, Bf)

    # B: Behavioral Balance
    B = np.maximum(Sm1*Bbar, 0)

    # I: Interest Income from Lending
    B_m_D = B*(1-PD)
    I = B_m_D*r  # <-- Interest Income

    #P = np.minimum(np.multiply(PRE, B[0, :]),B_m_D)
    ###1
    D = PD*B  # <-- Default
    # D = PD*shift(B,-1)
    #A = np.multiply(Afactor1(r, T, Tmax=241), B_m_D)
    A = np.multiply(Afactor1(r, T, Tmax=240), B_m_D)
    ###1
    P = np.minimum(np.multiply(PRE, B[0, :]),B_m_D)*np.logical_not(B_m_D-A==0)

    C = CAN*(B_m_D - A) # <-- Cancelations

    # EF: Interest Outcome from Cost of Funds
    D_ef = shift(D, 2)
    B_ef_aux = B[0, :]-np.cumsum(D_ef+A+C+P, axis=0)
    B_ef = np.maximum(np.concatenate([B[0, :].reshape(1, -1), B_ef_aux[0:-1, :]], axis=0), 0)
    #agregado
    B_ef[B_ef<0.05]=0
    
    COF = -B_ef*rc  # <-- Cost Of Funds (Interest Outcome)
    EL = -LGD*D_ef  # <-- Expected Loss
    EC = B_ef*ECf  # <-- Economic Capital
    RemEC = EC*RemCap  # <-- Remuneration to Economic Capital
    # Economic Capital Flow (Variation)
    EC_flow = -np.diff(EC, axis=0, append=0)
    EC_0 = -EC[0, :].reshape(1, -1)

    # C: Costs
    costos = param["COSTOS_VIDA"].values.reshape(1,-1)
    CH = np.maximum((1-cif_PD-cif_CAN), 0)  # Cuentas Hábiles

    #costos_total = costos*CH  # Costos Mantenimiento cuentas
    costos_total = -costos*CH*(B_m_D>0)  # Costos Mantenimiento cuentas
   
    
    
    # Costos Totales periodo 1-T / Comentar si se incorporan costos indirectos
    #B1 = shift(B,-1,0)
    # costos_total = -(costos_total)*(B_m_D>0)
    costos_T0 = -param["COSTOS_T0"]  # Costos en el periodo t=0

    # Ingresos No Financieros (INOF)
    # inof_t0 = param["INOF_T0"]
    inof_t0 = np.transpose(param["INOF_T0"])
    #Amort_c = Bbar-shift(Bbar, -1)  # Contractual Amortization (TxM)
    #I_c = Bbar*r  # Contractual Interest
    # Cuota = Amort_c[0, :] + I_c[0, :]  # Cuota (1xM)

    # Taxes (Impuesto a la Renta)
    tax_0 = -(costos_T0 + inof_t0)*IR
    
    #taxes = -(I+COF+EL+RemEC+costos_total)*IR
    taxes = -(I+COF+EL+RemEC+costos_total)*IR
    # Computing Cash Flow
    #CF = np.maximum(I + COF + EL + RemEC + EC_flow  + costos_total + taxes,0)
    CF = I + COF + EL + RemEC + EC_flow  + costos_total + taxes

    # Computing Net Present Value
    Tmax = CF.shape[0]
    descuento_matrix = np.array([np.power(1/(1+descuento), t)
                                 for descuento in descuentos for t in np.arange(1, Tmax+1).reshape(-1, 1)])
    # Discounted Cash Flows for simple sum
    CF_discounted = np.multiply(descuento_matrix, CF)
    CF_0 = tax_0 + EC_0 + costos_T0 + inof_t0  # Cash Flow in t=0
    
    df = pd.DataFrame(index = [np.arange(0,CF.shape[1])])
    
    if "tea" in dict_comp:
        df["tea"] = ((1+r)**12-1).reshape(-1,)
    if "tt" in dict_comp:
        df["tt"] = ((1+rc)**12-1).reshape(-1,)
    if "saldo" in dict_comp:
        df["saldo"] = np.sum(B,axis=0).reshape(-1,)
    if "capital" in dict_comp:
        df["capital"] = np.sum(EC,axis=0).reshape(-1,)
    if "ingreso_financiero" in dict_comp:
        df["ingreso_financiero"] = np.sum(I,axis=0).reshape(-1,)
    if "costo_financiero" in dict_comp:
        df["costo_financiero"] = np.sum(COF,axis=0).reshape(-1,)
    if "perdida_esperada" in dict_comp:
        df["perdida_esperada"] = np.sum(EL,axis=0).reshape(-1,)
    if "rem_capital" in dict_comp:
        df["rem_capital"] = np.sum(RemEC,axis=0).reshape(-1,)
    if "desgravamen" in dict_comp:
        df["desgravamen"] = np.sum(inof_t0,axis=0).reshape(-1,)
    if "costos" in dict_comp:
        df["costos"] = np.sum(costos_total + costos_T0,axis=0).reshape(-1,) 
    if "imp_renta" in dict_comp:
        df["imp_renta"] = np.sum(taxes + tax_0,axis=0).reshape(-1,)
    if "flujo_neto" in dict_comp:
        df["flujo_neto"] = (np.sum(CF,axis=0).reshape(1,-1) + CF_0).reshape(-1,)
    if "van" in dict_comp:
        pv = np.sum(CF_discounted, axis=0).reshape(1, -1) + CF_0
        df["van"] = pv.reshape(-1,)
    if "roe" in dict_comp:
        df["roe"] = (1+(np.sum(CF,axis=0).reshape(1,-1) + CF_0).reshape(-1,) / np.sum(EC,axis=0).reshape(-1,))**12-1
    return df

def get_schedule(descuentos, desembolsos, r, T, param, i):
    """
    Arguments:
    r:
    i:

    descuentos: TIR Objetivo
    desembolsos: Desembolsos efectivos
    T:
    param:

    Returns:
    Schedule for the i-th observation

    """
    cif_PD = param["cif_PD"]
    can = param["can"]
    pre = param["pre"]
    XC = param["XC"]
    XP = param["XP"]
    LGD = param["LGD"]
    rc = param["rc"]
    ECf = param["ECf"]
    RemCap = param["RemCap"]
    IR = param["IR"]

    r = r.reshape(1, -1)
    r_year = np.power(1 + r.reshape(-1, 1), 12) - 1

    # Update r --> Dependencia de Cancelaciones y Prepagos con la TEA
    cif_CAN = can.update_r(XC, r_year)
    cum_PRE = pre.update_r(XP, r_year)

    # Comentar si CLV es estándar
    PD, _ = cif_to_cond_prob(
        shift(cif_PD, -2), shift(cif_CAN, 1), 1, 1)
    CAN, _ = cif_to_cond_prob(shift(cif_CAN, 0), shift(
        cif_PD, -2), 1, 1)  # Comentar si CLV es estándar
    # PD, CAN = cif_to_cond_prob(cif_CAN,cif_PD,1,1) # Descomentar si CLV es estándar
    PRE = np.maximum(cum_PRE - shift(cum_PRE, 1),0)

    descuentos = descuentos.reshape(1, -1)

    # Bbar: Contractual Balance
    Bf = Bfactor2(r, T, Tmax=60)
    Bbar = np.multiply(Bf, desembolsos)

    # Survival Curves
    Sm1 = compute_S(PD, CAN, PRE, Bf)

    # B: Behavioral Balance
    B = np.maximum(Sm1*Bbar, 0)

    # I: Interest Income from Lending
    B_m_D = B*(1-PD)
    I = B_m_D*r  # <-- Interest Income

    P = np.minimum(np.multiply(PRE, B[0, :]),B_m_D)
    D = PD*B  # <-- Default
    # A = np.maximum((B - shift(B, -1)-D-P-CAN*B_m_D )/(1-CAN), 0)  # <-- Amortizations
    A = np.multiply(Afactor1(r, T, Tmax=60), B_m_D)
    C = CAN*(B_m_D -A) # <-- Cancelations

    # EF: Interest Outcome from Cost of Funds
    D_ef = shift(D, 2)
    B_ef_aux = B[0, :]-np.cumsum(D_ef+A+C+P, axis=0)
    B_ef = np.maximum(np.concatenate(
        [B[0, :].reshape(1, -1), B_ef_aux[0:-1, :]], axis=0), 0)
    
    COF = -B_ef*rc  # <-- Cost Of Funds (Interest Outcome)
    EL = -LGD*D_ef  # <-- Expected Loss
    EC = B_ef*ECf  # <-- Economic Capital
    RemEC = EC*RemCap  # <-- Remuneration to Economic Capital
    # Economic Capital Flow (Variation)
    EC_flow = -np.diff(EC, axis=0, append=0)
    EC_0 = -EC[0, :].reshape(1, -1)

    # EC_prom = (EC) --> Añadir cap promedio para ponderar la TIR
    # Para TIR PROM excluir los errores

    # C: Costs
    costos = param["COSTOS_VIDA"]
    CH = np.maximum((1-cif_PD-cif_CAN), 0)  # Cuentas Hábiles
    rr1 = param["RR1"]  # Curva Roll Rate (1-30)
    rr2 = param["RR2"]  # Curva Roll Rate (31-60)

    costos_rr1 = CH*costos[0]*rr1  # Costos cobranzas 1-30
    costos_rr2 = CH*costos[1]*rr2  # Costos cobranzas 31-60
    costos_BmD = B_m_D*costos[2]  # Costos Mantenimiento SBS
    # Costos: adminitrativos, procesos de operaciones, transaccionales, publicidad, otros
    costos_CH = CH * costos[3]
    # costos_indirectos = -param["costos_indirectos"] # Costos Indirectos: No se consideran en el CLV CEF, de ser necesario, crearlos acá

    # costos_total = -(costos_rr1 + costos_rr2 + costos_BmD + costos_CH + costos_indirectos) # Descomentar si se incorporan costos indirectos
    # Costos Totales periodo 1-T / Comentar si se incorporan costos indirectos
    B1 = shift(B,-1,0)
    costos_total = -(costos_rr1 + costos_rr2 + costos_BmD + costos_CH)*(B_m_D>0)
    costos_T0 = -param["COSTOS_T0"]  # Costos en el periodo t=0

    # Ingresos No Financieros (INOF)
    inof_tr, inof_min, inof_max = param["INOF_TR"], param["INOF_MIN"], param["INOF_MAX"]
    pa = param["PA"]
    Amort_c = Bbar-shift(Bbar, -1)  # Contractual Amortization (TxM)
    I_c = Bbar*r  # Contractual Interest
    Cuota = Amort_c[0, :] + I_c[0, :]  # Cuota (1xM)

    portes = CH*inof_tr[0]
    seguro = B_m_D*inof_tr[1]
    pago_atrasado = np.clip(inof_tr[2]*Cuota, inof_min, inof_max)*CH*pa

    inof = (portes + seguro + pago_atrasado)*(B_m_D>0)

    # Taxes (Impuesto a la Renta)
    tax_0 = -(costos_T0)*IR
    taxes = -(I+COF+EL+RemEC+inof+costos_total)*IR

    # Computing Cash Flow
    CF = I + COF + EL + RemEC + EC_flow + inof + costos_total + taxes

    # Computing Net Present Value
    Tmax = CF.shape[0]
    descuento_matrix = np.array([np.power(1/(1+descuento), t)
                                 for descuento in descuentos for t in np.arange(1, Tmax+1).reshape(-1, 1)])

    schedule = pd.DataFrame()
    schedule_flow = pd.DataFrame()
    # Discounted Cash Flows for simple sum
    pv_list = [("VAN TOTAL", "Flujo Neto", "Flujo Neto 0", CF[:,i], costos_T0[0,i]+tax_0[0,i]+EC_0[0,i]), 
            ("VAN CAPITAL", "Capital Requerido", "Capital Requerido 0", EC_flow[:,i], EC_0[0,i]), 
            ("VAN RemCap", "Remuneración de Capital", "Remuneración de Capital 0", RemEC[:,i], 0), 
            ("VAN Intereses", "Intereses Financieros", "Intereses Financieros 0", I[:,i], 0), 
            ("VAN INOF", "Ingresos No Financieros Totales", "Ingresos No Financieros 0", inof[:,i], 0), 
            ("VAN TT", "Intereses Egresos Financieros", "Intereses Egresos Financieros 0", COF[:,i], 0), 
            ("VAN PERDIDA", "Pérdida de Capital", "Pérdida de Capital 0", EL[:,i], 0), 
            ("VAN COSTOS", "Costos Totales", "Costo 0", costos_total[:,i], costos_T0[0,i]), 
             ("VAN IR", "IR", "IR 0", taxes[:,i], tax_0[0,i]),
            ("B", "B", "B", B[:,i], 0),
            ("B_ef", "B_ef", "B_ef", B_ef[:,i], 0),
            ("D", "D", "D", D[:,i], 0),
            ("D_ef", "D_ef", "D_ef", D_ef[:,i], 0),
            ("C", "C", "C", C[:,i], 0),
            ("P", "P", "P", P[:,i], 0),
            ("A", "A", "A", A[:,i], 0),
            ("I", "I", "I", I[:,i], 0),
              ]
    for title1, title2, title3, t_1_T, t_0 in pv_list:
        CF_discounted = np.multiply(descuento_matrix[:,i], t_1_T)
        schedule.loc[:, title1] = [np.sum(CF_discounted, axis=0).reshape(1,-1) + t_0] # PV + T0
        schedule_flow.loc[:,title2] = t_1_T # CF
        if t_0!=0:
            schedule.loc[:, title3] = t_0 # T0
    schedule.to_csv("Schedule - "+str(i)+".csv")

    return schedule, schedule_flow

def get_rmin_van_decomp(descuentos, desembolsos, r, T, param):
    """
    Arguments:
    r: Tasa mínima calculada previamente --> self.get_rmin()

    descuentos: TIR Objetivo
    desembolsos: Desembolsos efectivos
    T:
    param:

    Returns:
    Dataframe with VAN Total, VAN Capital, VAN TT, VAN Perdida, VAN Costos, VAN IR, Saldo Promedio & Capital Promedio 

    """

    cif_PD = param["cif_PD"]
    can = param["can"]
    pre = param["pre"]
    XC = param["XC"]
    XP = param["XP"]
    LGD = param["LGD"]
    rc = param["rc"]
    ECf = param["ECf"]
    RemCap = param["RemCap"]
    IR = param["IR"]

    r = r.reshape(1, -1)
    r_year = np.power(1 + r.reshape(-1, 1), 12) - 1

    # Update r --> Dependencia de Cancelaciones y Prepagos con la TEA
    cif_CAN = can.update_r(XC, r_year)
    cum_PRE = pre.update_r(XP, r_year)

    # Comentar si CLV es estándar
    PD, _ = cif_to_cond_prob(
        shift(cif_PD, -2), shift(cif_CAN, 1), 1, 1)
    CAN, _ = cif_to_cond_prob(shift(cif_CAN, 0), shift(
        cif_PD, -2), 1, 1)  # Comentar si CLV es estándar
    # PD, CAN = cif_to_cond_prob(cif_CAN,cif_PD,1,1) # Descomentar si CLV es estándar
    PRE = np.maximum(cum_PRE - shift(cum_PRE, 1),0)

    descuentos = descuentos.reshape(1, -1)

    # Bbar: Contractual Balance
    Bf = Bfactor2(r, T, Tmax=60)
    Bbar = np.multiply(Bf, desembolsos)

    # Survival Curves
    Sm1 = compute_S(PD, CAN, PRE, Bf)

    # B: Behavioral Balance
    B = np.maximum(Sm1*Bbar, 0)

    # I: Interest Income from Lending
    B_m_D = B*(1-PD)
    I = B_m_D*r  # <-- Interest Income

    P = np.minimum(np.multiply(PRE, B[0, :]),B_m_D)
    D = PD*B  # <-- Default
    # A = np.maximum((B - shift(B, -1)-D-P-CAN*B_m_D )/(1-CAN), 0)  # <-- Amortizations
    A = np.multiply(Afactor1(r, T, Tmax=60), B_m_D)
    C = CAN*(B_m_D -A) # <-- Cancelations

    # EF: Interest Outcome from Cost of Funds
    D_ef = shift(D, 2)
    B_ef_aux = B[0, :]-np.cumsum(D_ef+A+C+P, axis=0)
    B_ef = np.maximum(np.concatenate(
        [B[0, :].reshape(1, -1), B_ef_aux[0:-1, :]], axis=0), 0)
    
    COF = -B_ef*rc  # <-- Cost Of Funds (Interest Outcome)
    EL = -LGD*D_ef  # <-- Expected Loss
    EC = B_ef*ECf  # <-- Economic Capital
    RemEC = EC*RemCap  # <-- Remuneration to Economic Capital
    # Economic Capital Flow (Variation)
    EC_flow = -np.diff(EC, axis=0, append=0)
    EC_0 = -EC[0, :].reshape(1, -1)

    # C: Costs
    costos = param["COSTOS_VIDA"]
    CH = np.maximum((1-cif_PD-cif_CAN), 0)  # Cuentas Hábiles
    rr1 = param["RR1"]  # Curva Roll Rate (1-30)
    rr2 = param["RR2"]  # Curva Roll Rate (31-60)

    costos_rr1 = CH*costos[0]*rr1  # Costos cobranzas 1-30
    costos_rr2 = CH*costos[1]*rr2  # Costos cobranzas 31-60
    costos_BmD = B_m_D*costos[2]  # Costos Mantenimiento SBS
    # Costos: adminitrativos, procesos de operaciones, transaccionales, publicidad, otros
    costos_CH = CH * costos[3]
    # costos_indirectos = -param["costos_indirectos"] # Costos Indirectos: No se consideran en el CLV CEF, de ser necesario, crearlos acá

    # costos_total = -(costos_rr1 + costos_rr2 + costos_BmD + costos_CH + costos_indirectos) # Descomentar si se incorporan costos indirectos
    # Costos Totales periodo 1-T / Comentar si se incorporan costos indirectos
    B1 = shift(B,-1,0)
    costos_total = -(costos_rr1 + costos_rr2 + costos_BmD + costos_CH)*(B_m_D>0)
    costos_T0 = -param["COSTOS_T0"]  # Costos en el periodo t=0

    # Ingresos No Financieros (INOF)
    inof_tr, inof_min, inof_max = param["INOF_TR"], param["INOF_MIN"], param["INOF_MAX"]
    pa = param["PA"]
    Amort_c = Bbar-shift(Bbar, -1)  # Contractual Amortization (TxM)
    I_c = Bbar*r  # Contractual Interest
    Cuota = Amort_c[0, :] + I_c[0, :]  # Cuota (1xM)

    portes = CH*inof_tr[0]
    seguro = B_m_D*inof_tr[1]
    pago_atrasado = np.clip(inof_tr[2]*Cuota, inof_min, inof_max)*CH*pa

    inof = (portes + seguro + pago_atrasado)*(B_m_D>0)

    # Taxes (Impuesto a la Renta)
    tax_0 = -(costos_T0)*IR
    taxes = -(I+COF+EL+RemEC+inof+costos_total)*IR

    # Computing Cash Flow
    CF = I + COF + EL + RemEC + EC_flow + inof + costos_total + taxes

    # Computing Net Present Value
    Tmax = CF.shape[0]
    m = CF.shape[1]
    descuento_matrix = np.array([np.power(1/(1+descuento), t)
                                 for descuento in descuentos for t in np.arange(1, Tmax+1).reshape(-1, 1)])

    schedule = pd.DataFrame()
   # Discounted Cash Flows for simple sum
    # pv_list = [("VAN TOTAL", CF, costos_T0+tax_0+EC_0), ("VAN CAPITAL", EC_flow, EC_0), ("VAN TT", COF, 0), ("VAN PERDIDA", EL, 0), 
    #         ("VAN COSTOS", costos_total, costos_T0), ("VAN IR", taxes, tax_0)]
    # for title, t_1_T, t_0 in pv_list:
    #     CF_discounted = np.multiply(descuento_matrix, t_1_T)
    #     schedule[title] = (np.sum(CF_discounted, axis=0).reshape(1,-1) + t_0).reshape(m,) # PV + T0
    pv_list = [("VAN TOTAL", CF, costos_T0+tax_0+EC_0), ("VAN CAPITAL", EC_flow, EC_0), ("VAN TT", COF, 0), ("VAN PERDIDA", EL, 0), 
            ("VAN COSTOS", costos_total, costos_T0), ("VAN IR", taxes, tax_0), ("VAN INTERESES", I, 0), ("VAN INOF", inof, 0), ("VAN REMCAP", RemEC, 0)]
    for title, t_1_T, t_0 in pv_list:
        CF_discounted = np.multiply(descuento_matrix, t_1_T)
        schedule[title] = (np.sum(CF_discounted, axis=0).reshape(1,-1) + t_0).reshape(m,) # PV + T0

    # Saldo y Cap. Requerido Total:
    saldo_promedio = np.sum(B, axis=0) / T # 1 x m
    EC_promedio = np.sum(EC, axis=0) / T # 1 x m
    schedule["Saldo Promedio"] = saldo_promedio.reshape(m,) # m,
    schedule["Capital Promedio"] = EC_promedio.reshape(m,) # m,

    return schedule

def get_parameters_transform(X, tt_dscto):
    m = X.shape[0]
    dscto_anual = tt_dscto["td"].values
    descuentos = ((1+dscto_anual)**(1/12)-1)*np.ones((1, m)).reshape(1, -1)
    # dscto_anual_leads = tt_dscto["Tasa_dscto_leads"].values
    # descuentos_leads = ((1+dscto_anual_leads)**(1/12)-1)*np.ones((1, m)).reshape(1, -1)
    desembolsos = X.loc[:, "MO_DESEMBOLSO_SOLES"].values
    r = X.loc[:, "TEA"].values
    r = np.power(1+r, 1/12)-1
    r = r.reshape(1, -1)
    T = X.loc[:, "PLAZO_MESES"].values
    T = T.reshape(1, -1)

    return descuentos, desembolsos, r, T

def get_param(X_transformed, X_curves):
    
    RemCap_year = X_transformed["TT_DSCTO"]["remcap"].values
    RemCap = (1+RemCap_year)**(1/12)-1

    rc = X_curves["TT_DSCTO"]["TT"].values
    rc = np.power(1+rc,1/12)-1
    rc = rc.reshape(1,-1)

    cif_PD = X_curves["PD"] # Matrix of cumulative Pds (shape T,m)
    cif_PD = np.concatenate(cif_PD.values.tolist()).T
    cif_CAN = X_curves["CAN"] # Matrix of cumulative CANs (shape T,m)
    cif_CAN = np.concatenate(cif_CAN.values.tolist()).T
    cum_PRE = X_curves["PRE"] # Matrix of cumulative PREs (shape T,m)
    cum_PRE = np.concatenate(cum_PRE.values.tolist()).T
    LGD = X_curves["LGD"]
    LGD = np.concatenate(LGD.values.tolist()).T
    ECf = X_curves["CAP"]
    ECf = np.concatenate(ECf.values.tolist()).T
    PD , _ = cif_to_cond_prob(shift(cif_PD,-2),shift(cif_CAN,1),1,1) # comentar si clv se corrige a modelo estandar
    CAN , _ = cif_to_cond_prob(shift(cif_CAN,0),shift(cif_PD,-2),1,1) # comentar si clv se corrige a modelo estandar
    # PD , CAN = cif_to_cond_prob(cif_CAN,cif_PD,1,1) # descomentar si clv se corrige al modelo estandar. todo se calculara en un solo paso
    PRE = cum_PRE - shift(cum_PRE,1)
    COSTOS_T0, COSTOS_VIDA = X_curves["COST"]
    COSTOS_T0 = np.array(np.sum(COSTOS_T0, axis=1)).reshape(1, -1)
    # COSTOS_VIDA = np.array(COSTOS_VIDA).T
    # INOF_T0 = X_transformed["INOF"]
    INOF_T0 = np.array(X_curves["INOF"])
    # INOF_T0 = np.array(INOF_T0).T

    param = {"PD":PD,"cif_PD" :cif_PD, "CAN":CAN, "cif_CAN":cif_CAN, "PRE":PRE, "cum_PRE":cum_PRE,"COSTOS_VIDA":COSTOS_VIDA, "COSTOS_T0":COSTOS_T0,
            "rc":rc,"LGD":LGD,"ECf":ECf, "RemCap": RemCap,"INOF_T0":INOF_T0, "IR":0.295}

    return param
