import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
from fpdf import FPDF
import tempfile
import datetime
import logging
import sys

# ============================================================================
# 0. LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BCNModel")

# ============================================================================
# 1. CONFIGURACION VISUAL
# ============================================================================
st.set_page_config(
    page_title="Barcelona Strategic Model v33.0",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
h1 { color: #0d47a1; font-family: 'Helvetica', sans-serif; font-weight: 800; }
.stButton>button {
    width: 100%; background-color: #0d47a1; color: white;
    font-weight: 600; border-radius: 6px; height: 48px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2); border: none;
}
.stButton>button:hover { background-color: #1565c0; }
</style>
""", unsafe_allow_html=True)
warnings.filterwarnings('ignore')

# ============================================================================
# 2. DATOS — AMBOS SETS
# ============================================================================

# Orden canónico de distritos (mismo en ambos sets)
distritos = [
    'Ciutat Vella', 'Eixample', 'Gracia', 'Horta Guinardo', 'Les Corts',
    'Nou Barris', 'Sant Andreu', 'Sant Marti', 'Sants-Montjuic', 'Sarria-Sant Gervasi'
]
n_distritos = len(distritos)

# Densidad licencias turísticas — invariante al set de datos
concentracion_vt_real = np.array([0.150, 0.460, 0.105, 0.030, 0.035,
                                   0.004, 0.015, 0.120, 0.110, 0.050])

# Parámetros de simulación — invariantes al set de datos
sensibilidad_distrital = np.array([0.95, 0.85, 0.80, 0.35, 0.60,
                                    0.30, 0.40, 0.70, 0.60, 0.65])
rho_vta_alq = 0.45

# Índice macro agregado Barcelona EUR/m2 (INE/IPV) — para reconstrucción CAPM
datos_agregados_bcn = {
    2007: 4071, 2008: 3737, 2009: 3728, 2010: 3752, 2011: 3398, 2012: 3076,
    2013: 2992, 2014: 3061, 2015: 3297, 2016: 3649, 2017: 4154, 2018: 4232,
    2019: 4135, 2020: 4018, 2021: 3910, 2022: 4063, 2023: 4167, 2024: 4700,
    2025: 5042, 2026: 5200
}
anos_macro   = np.array(list(datos_agregados_bcn.keys()))
precios_macro = np.array(list(datos_agregados_bcn.values()), dtype=float)

# ---------------------------------------------------------------------------
# SET A — PORTALES INMOBILIARIOS (Idealista / Incasol)
# Venta EUR/m2: columnas 2019..2026
# ---------------------------------------------------------------------------
DATASET_A = {
    'nombre':     'Datos Portales Inmobiliarios',
    'fuente':     'Idealista / Incasol (Generalitat de Catalunya)',
    'fecha_dato': 'Q1 2026',
    'ano_fin':    2026,
    'anos_venta': np.arange(2019, 2027),
    'anos_alq':   np.arange(2019, 2027),
    # Orden distritos: Ciutat Vella, Eixample, Gracia, Horta Guinardo, Les Corts,
    #                  Nou Barris, Sant Andreu, Sant Marti, Sants-Montjuic, Sarria-Sant Gervasi
    'venta': np.array([
        [4200, 4100, 4250, 4450, 4620, 4740, 4817, 4811],
        [5500, 5350, 5550, 5850, 6050, 6180, 6299, 6363],
        [4600, 4480, 4650, 4900, 5100, 5220, 5301, 5404],
        [3350, 3280, 3400, 3600, 3750, 3820, 3862, 3905],
        [5350, 5200, 5400, 5700, 5950, 6080, 6135, 6355],
        [2450, 2400, 2490, 2630, 2730, 2780, 2804, 2946],
        [3300, 3220, 3340, 3530, 3670, 3740, 3774, 3730],
        [4180, 4080, 4230, 4460, 4640, 4730, 4784, 4899],
        [3780, 3690, 3830, 4040, 4200, 4280, 4338, 4477],
        [5900, 5750, 5970, 6290, 6540, 6670, 6738, 6896],
    ], dtype=float),
    'alquiler': np.array([
        [22.5, 20.8, 21.5, 23.2, 24.8, 25.8, 26.4, 25.7],
        [22.8, 21.2, 22.0, 23.8, 25.2, 26.1, 26.7, 26.0],
        [20.8, 19.5, 20.2, 21.8, 23.1, 24.0, 24.4, 23.7],
        [15.6, 14.8, 15.3, 16.5, 17.5, 18.1, 18.4, 19.1],
        [18.7, 17.5, 18.1, 19.6, 20.8, 21.6, 22.0, 21.9],
        [13.4, 12.8, 13.2, 14.3, 15.1, 15.6, 15.8, 16.3],
        [15.8, 15.0, 15.5, 16.8, 17.8, 18.3, 18.6, 18.5],
        [20.0, 18.8, 19.5, 21.0, 22.3, 23.2, 23.5, 23.6],
        [17.7, 16.7, 17.3, 18.7, 19.8, 20.5, 20.8, 20.8],
        [19.8, 18.5, 19.2, 20.7, 22.0, 22.8, 23.2, 23.6],
    ], dtype=float),
}

# ---------------------------------------------------------------------------
# SET B — AYUNTAMIENTO DE BARCELONA
# Orden distritos igual al canónico (reordenado desde fuente):
#   Fuente: Ciutat Vella, Eixample, Sants-Montjuïc, Les Corts, Sarrià-Sant Gervasi,
#           Gràcia, Horta-Guinardó, Nou Barris, Sant Andreu, Sant Martí
# Reordenamos a: Ciutat Vella, Eixample, Gracia, Horta Guinardo, Les Corts,
#                Nou Barris, Sant Andreu, Sant Marti, Sants-Montjuic, Sarria-Sant Gervasi
#
# Venta EUR/m2: 2012-2025 (fuente: Portal de Dades Ajuntament BCN)
# Separador miles = punto → valores ya en float
# ---------------------------------------------------------------------------
_venta_ayto_src = np.array([
    # CV      EIX     SANTS   LCORTS  SARRIA  GRACIA  HORTA   NOUB    SANDREU SMARTI
    [2355,   2767,   1965,   2974,   3339,   2590,   2215,   1668,   2151,   2432],  # 2012
    [2266,   2619,   1739,   2857,   3105,   2369,   1770,   1536,   1756,   2122],  # 2013
    [2387,   2683,   1843,   2981,   3108,   2374,   1776,   1398,   1823,   2169],  # 2014
    [2730,   2983,   2186,   3076,   3509,   2575,   2101,   2029,   2100,   2555],  # 2015
    [2989,   3381,   2354,   3284,   3750,   2953,   2038,   1621,   1994,   2458],  # 2016
    [3457,   3869,   2815,   3880,   4330,   3445,   2464,   1918,   2419,   3001],  # 2017
    [3833,   4111,   3137,   4204,   4755,   3824,   2690,   2174,   2626,   3069],  # 2018
    [3592,   4045,   3153,   3980,   4497,   3766,   2648,   2191,   2717,   3029],  # 2019
    [3584,   3977,   3092,   4161,   4388,   3695,   2657,   2098,   2671,   3095],  # 2020
    [3515,   4034,   3112,   4001,   4563,   3717,   2811,   2073,   2770,   3095],  # 2021
    [3752,   4391,   3220,   4152,   4811,   3990,   2897,   2495,   2905,   3490],  # 2022
    [3749,   4582,   3305,   4260,   5054,   4059,   2833,   2373,   2810,   3377],  # 2023
    [3845,   4869,   3446,   4666,   5200,   4320,   3213,   2531,   3124,   3693],  # 2024
    [4144,   5325,   3869,   5100,   5629,   4814,   3464,   3005,   3505,   4138],  # 2025
], dtype=float)

# Reordenar columnas de fuente a orden canónico:
# Fuente idx: CV=0, EIX=1, SANTS=2, LCORTS=3, SARRIA=4, GRACIA=5, HORTA=6, NOUB=7, SANDREU=8, SMARTI=9
# Canónico:   CV=0, EIX=1, GRACIA=2, HORTA=3, LCORTS=4, NOUB=5, SANDREU=6, SMARTI=7, SANTS=8, SARRIA=9
_col_map_ayto = [0, 1, 5, 6, 3, 7, 8, 9, 2, 4]
_venta_ayto_canon = _venta_ayto_src[:, _col_map_ayto].T  # shape (10, 14)

# Alquiler EUR/m2/mes: 2000-2025 (fuente: Portal de Dades Ajuntament BCN)
# Separador decimales = coma → ya convertido a float
_alq_ayto_src = np.array([
    # CV    EIX   SANTS  LCORTS SARRIA GRACIA HORTA  NOUB  SANDREU SMARTI
    [5.5,  5.8,  6.1,   7.6,   7.4,   6.3,  5.8,  5.4,  5.5,   5.7],  # 2000
    [6.4,  6.8,  6.7,   8.6,   8.2,   7.0,  6.4,  6.2,  6.1,   6.5],  # 2001
    [7.3,  7.5,  7.5,   8.9,   9.1,   8.1,  7.1,  7.2,  6.7,   7.5],  # 2002
    [7.6,  7.9,  7.8,   9.3,   9.4,   8.4,  7.7,  7.3,  7.3,   7.7],  # 2003
    [8.6,  8.5,  8.7,  10.0,  10.0,   9.3,  8.3,  7.9,  7.8,   8.2],  # 2004
    [9.8,  9.5,  9.5,  10.8,  10.8,  10.2,  9.2,  8.9,  9.0,   9.2],  # 2005
    [10.7, 10.3, 10.5, 11.9,  11.6,  11.3, 10.2,  9.9,  9.6,  10.3],  # 2006
    [11.6, 11.3, 11.3, 13.0,  12.7,  12.5, 11.1, 10.8, 10.7,  11.2],  # 2007
    [12.6, 11.9, 12.2, 13.4,  13.3,  13.0, 12.0, 11.5, 11.5,  11.8],  # 2008
    [12.4, 11.6, 11.9, 12.9,  12.9,  12.8, 11.4, 11.2, 11.1,  11.6],  # 2009
    [12.1, 11.3, 11.5, 12.3,  12.8,  12.6, 10.8, 10.7, 10.7,  11.2],  # 2010
    [12.0, 11.2, 11.3, 12.1,  12.8,  12.0, 10.6, 10.3, 10.4,  10.9],  # 2011
    [11.7, 10.8, 10.8, 11.6,  12.3,  11.4, 10.1,  9.5,  9.8,  10.5],  # 2012
    [11.4, 10.2, 10.1, 11.2,  11.7,  11.0,  9.3,  8.7,  9.2,   9.9],  # 2013
    [11.4, 10.3,  9.9, 11.0,  11.9,  10.8,  9.0,  8.4,  8.9,   9.7],  # 2014
    [12.4, 11.2, 10.7, 12.0,  13.0,  11.8,  9.9,  8.9,  9.6,  10.6],  # 2015
    [13.9, 12.4, 11.7, 13.2,  14.3,  12.9, 10.7,  9.6, 10.4,  11.7],  # 2016
    [15.5, 13.4, 12.9, 14.2,  15.4,  14.3, 11.9, 10.6, 11.4,  12.9],  # 2017
    [14.5, 13.7, 13.0, 14.5,  15.4,  14.0, 11.9, 11.2, 11.7,  13.2],  # 2018
    [14.8, 14.3, 13.5, 15.0,  15.7,  14.6, 12.7, 11.7, 12.2,  13.9],  # 2019
    [14.3, 14.2, 13.6, 14.7,  15.6,  14.4, 12.8, 11.7, 12.3,  13.8],  # 2020
    [13.2, 13.3, 13.0, 13.8,  14.9,  13.8, 12.2, 11.3, 11.7,  13.3],  # 2021
    [15.8, 14.9, 14.1, 15.2,  16.2,  15.3, 13.1, 12.1, 12.5,  14.5],  # 2022
    [16.7, 16.5, 16.3, 16.8,  18.2,  16.9, 14.5, 13.3, 13.8,  16.2],  # 2023
    [16.6, 16.6, 15.6, 17.4,  18.4,  17.5, 14.5, 13.1, 13.7,  16.2],  # 2024
    [17.8, 16.5, 16.3, 17.2,  18.9,  17.3, 14.6, 13.1, 14.4,  16.2],  # 2025
], dtype=float)

_alq_ayto_canon = _alq_ayto_src[:, _col_map_ayto].T  # shape (10, 26)

DATASET_B = {
    'nombre':     'Datos Oficiales Ayuntamiento de Barcelona',
    'fuente':     'Portal de Dades Ajuntament de Barcelona',
    'fecha_dato': '2025',
    'ano_fin':    2025,
    'anos_venta': np.arange(2012, 2026),
    'anos_alq':   np.arange(2000, 2026),
    'venta':      _venta_ayto_canon,
    'alquiler':   _alq_ayto_canon,
}

DATASETS = {
    'Datos Portales Inmobiliarios':             DATASET_A,
    'Datos Oficiales Ayuntamiento de Barcelona': DATASET_B,
}

# ============================================================================
# 3. INDICADORES MACRO Y CADENA DE MARKOV NO HOMOGENEA
# ============================================================================
MACRO_DEFAULTS = {
    "Crecimiento": {
        "euribor": 1.5, "ipc": 2.5, "paro": 9.0,
    },
    "Estanflacion": {
        "euribor": 3.8, "ipc": 5.5, "paro": 13.0,
    },
    "Recesion (Estructural)": {
        "euribor": 2.5, "ipc": 1.0, "paro": 20.0,
    },
}


def calcular_probabilidades_regimen(euribor, ipc, paro):
    euribor_norm = np.clip((euribor - 0.0) / 5.0, 0.0, 1.0)
    ipc_norm     = np.clip((ipc     - 0.0) / 8.0, 0.0, 1.0)
    paro_norm    = np.clip((paro   - 5.0) / 25.0, 0.0, 1.0)
    score_boom   = (1.0 - euribor_norm) * 0.4 + \
                   (1.0 - paro_norm)    * 0.4 + \
                   np.clip(ipc_norm * (1 - ipc_norm) * 4, 0, 1) * 0.2
    score_crisis = euribor_norm * 0.45 + paro_norm * 0.45 + \
                   np.clip((ipc_norm - 0.5), 0, 0.5) * 0.1
    raw  = np.array([0.5, score_boom, score_crisis])
    exps = np.exp(raw * 2.5)
    return exps / exps.sum()


def construir_mtm_macro(euribor, ipc, paro):
    p_norm, p_boom, p_crisis = calcular_probabilidades_regimen(euribor, ipc, paro)
    PERSIST = 0.55
    mtm = np.array([
        [PERSIST + (1 - PERSIST) * p_norm,
         (1 - PERSIST) * p_boom,
         (1 - PERSIST) * p_crisis],
        [0.35 * (1 - p_boom),
         PERSIST + (1 - PERSIST) * p_boom,
         0.05 + 0.10 * p_crisis],
        [0.12 + 0.20 * p_norm,
         0.02,
         PERSIST + (1 - PERSIST) * p_crisis],
    ])
    for r in range(3):
        s = mtm[r].sum()
        if s > 0:
            mtm[r] /= s
    return mtm


def validar_mtm(mtm, nombre):
    row_sums = mtm.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"MTM '{nombre}' invalida: sumas={row_sums}")
    return True


# ============================================================================
# 4. MOTOR ESTRUCTURAL — calibración adaptativa al set de datos
# ============================================================================

@st.cache_data
def inicializar_modelo(dataset_key: str):
    """
    Calibra el modelo estructural completo para el set de datos indicado.

    LOGICA ADAPTATIVA DE ESTIMACION:
    ---------------------------------
    Si el set tiene n_retornos_venta >= 10 (suficiente para VAR con p=10):
        → VAR(1) multivariado completo con regularización Ridge (lambda=0.5)
          para garantizar radio espectral < 1 sin truncar información.
    Si n_retornos_venta < 10:
        → AR(1) univariado diagonal por distrito, truncado a [-0.35, +0.35].
          Estadísticamente más honesto con n<<p.

    En ambos casos el motor de simulación es idéntico: A_var * r_prev.
    La diferencia es solo en cómo se estima A_var.
    """
    ds = DATASETS[dataset_key]
    pv = ds['venta']      # (10, T_v)
    pa = ds['alquiler']   # (10, T_a)

    # --- RETORNOS ---
    ret_v = (pv[:, 1:] / pv[:, :-1]) - 1   # (10, T_v-1)
    ret_a = (pa[:, 1:] / pa[:, :-1]) - 1   # (10, T_a-1)

    # --- DRIFT HISTORICO (media de retornos observados) ---
    drift_venta   = ret_v.mean(axis=1)
    drift_alquiler = ret_a.mean(axis=1)

    # --- BETAS CAPM ---
    ano_fin = ds['ano_fin']
    anos_v  = ds['anos_venta']
    # Alinear índice macro al rango de venta
    mask_m  = (anos_macro >= anos_v[0]) & (anos_macro <= anos_v[-1])
    pm_sub  = precios_macro[mask_m]
    if len(pm_sub) >= 2:
        ret_mkt = (pm_sub[1:] / pm_sub[:-1]) - 1
        n_align = min(ret_v.shape[1], len(ret_mkt))
        betas = []
        for i in range(n_distritos):
            rv = ret_v[i, -n_align:]
            rm = ret_mkt[-n_align:]
            if np.var(rm) > 1e-10:
                b = np.cov(rv, rm)[0, 1] / np.var(rm)
            else:
                b = 1.0
            betas.append(np.clip(b, 0.5, 1.5))
        betas = np.array(betas)
    else:
        betas = np.ones(n_distritos)

    # --- ESTIMACION A_VAR ADAPTATIVA ---
    n_ret = ret_v.shape[1]  # número de retornos anuales de venta
    logger.info("[%s] n_retornos_venta=%d", dataset_key, n_ret)

    if n_ret >= 10:
        # VAR(1) multivariado con regularizacion Ridge
        # r(t) = A * r(t-1) + eps
        # A_ridge = (X'X + lambda*I)^{-1} X'Y
        Y = ret_v[:, 1:].T    # (T-2, 10)
        X = ret_v[:, :-1].T   # (T-2, 10)
        lam = 0.5  # regularizacion: encoge coefs hacia 0, garantiza estabilidad
        try:
            XtX  = X.T @ X + lam * np.eye(n_distritos)
            A_var = np.linalg.solve(XtX, X.T @ Y)  # (10, 10)
            # Verificar radio espectral y reescalar si necesario
            rho_spec = np.abs(np.linalg.eigvals(A_var)).max()
            if rho_spec > 0.85:
                A_var = A_var * (0.85 / rho_spec)
            metodo_var = f"VAR(1) Ridge (lambda={lam}, n={n_ret})"
            rho_final  = np.abs(np.linalg.eigvals(A_var)).max()
        except Exception:
            A_var = np.diag(np.zeros(n_distritos))
            metodo_var = "VAR(1) fallback=0"
            rho_final  = 0.0
    else:
        # AR(1) univariado diagonal — honesto con n<<p
        ar1 = np.zeros(n_distritos)
        for i in range(n_distritos):
            r = ret_v[i]
            if len(r) >= 3:
                cv = np.cov(r[1:], r[:-1])[0, 1]
                vv = np.var(r[:-1])
                ar1[i] = np.clip(cv / vv if vv > 1e-10 else 0.0, -0.35, 0.35)
        A_var = np.diag(ar1)
        metodo_var = f"AR(1) diagonal (n={n_ret})"
        rho_final  = np.abs(ar1).max()

    logger.info("[%s] Metodo: %s | Radio espectral final: %.4f",
                dataset_key, metodo_var, rho_final)

    # --- RECONSTRUCCION HISTORICA para Cholesky ---
    # Usamos el rango largo del índice macro para reconstruir 2007-inicio
    idx_ini = np.where(anos_macro == anos_v[0])[0]
    if len(idx_ini) == 0:
        idx_ini = 0
    else:
        idx_ini = idx_ini[0]

    reconstruidos = np.zeros((n_distritos, len(anos_macro)))
    # Rellenar años conocidos
    for ti, yr in enumerate(anos_v):
        m_idx = np.where(anos_macro == yr)[0]
        if len(m_idx):
            reconstruidos[:, m_idx[0]] = pv[:, ti]
    # Rellenar hacia atrás
    np.random.seed(999)
    for t in range(idx_ini - 1, -1, -1):
        rmt = (precios_macro[t + 1] / precios_macro[t]) - 1
        for d in range(n_distritos):
            ruido = np.random.normal(0, 0.003)
            r_d   = betas[d] * rmt + ruido
            if reconstruidos[d, t + 1] > 0:
                reconstruidos[d, t] = reconstruidos[d, t + 1] / (1 + r_d)
            else:
                reconstruidos[d, t] = pv[d, 0] / (1 + r_d) ** (idx_ini - t)

    # --- CHOLESKY ---
    ret_full = np.zeros_like(reconstruidos)
    for t in range(1, len(anos_macro)):
        denom = reconstruidos[:, t - 1]
        denom = np.where(denom > 0, denom, 1.0)
        ret_full[:, t] = (reconstruidos[:, t] / denom) - 1
    ret_full = ret_full[:, 1:]
    corr = np.corrcoef(ret_full)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals  = np.maximum(eigvals, 1e-8)
        corr_r   = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_diag   = np.sqrt(np.diag(corr_r))
        corr_r   = corr_r / np.outer(d_diag, d_diag)
        try:
            L = np.linalg.cholesky(corr_r)
        except Exception:
            L = np.eye(n_distritos)
        corr = corr_r

    # --- VOLATILIDADES CALIBRADAS ---
    vol_cal = np.array([ret_v[i].std() for i in range(n_distritos)])
    vol_cal = np.clip(vol_cal, 0.015, 0.10)

    return {
        'reconstruidos': reconstruidos,
        'L':             L,
        'betas':         betas,
        'corr':          corr,
        'drift_venta':   drift_venta,
        'drift_alquiler': drift_alquiler,
        'A_var':         A_var,
        'vol_cal':       vol_cal,
        'metodo_var':    metodo_var,
        'rho_espectral': rho_final,
        'n_ret_venta':   n_ret,
    }


# ============================================================================
# 5. MOTOR DE SIMULACION MS-VAR (IDENTICO PARA AMBOS SETS)
# ============================================================================

def _params_hash(params: dict) -> tuple:
    return (
        params['dataset_key'],
        params['regimen'],
        round(params['euribor'], 4),
        round(params['ipc'],     4),
        round(params['paro'],    4),
        params['shock_vt'],
        params['shock_ano'],
        params.get('n_sim', 1000),
    )


@st.cache_data
def simular_cached(params_hash: tuple, _modelo: dict):
    """
    Motor Monte Carlo MS-VAR v33. Acepta cualquier set de datos a través de _modelo.
    El guion bajo en _modelo evita que st.cache_data intente hashear el dict de numpy.
    """
    (dataset_key, regimen, euribor, ipc, paro,
     shock_vt, shock_ano, n_sim) = params_hash

    ds = DATASETS[dataset_key]
    pv_base  = ds['venta'][:, -1].copy()   # precios base = último año del set
    pa_base  = ds['alquiler'][:, -1].copy()
    ano_base = ds['ano_fin']
    anos_proy_loc = np.arange(ano_base + 1, 2033)

    mtm = construir_mtm_macro(euribor, ipc, paro)
    validar_mtm(mtm, f"MTM {regimen}")

    # Ajuste macro al drift
    ipc_real_adj    = np.clip((ipc  - 2.0) / 100.0, -0.03, 0.03)
    euribor_adj_vta = -np.clip((euribor - 2.5) / 100.0, -0.02, 0.04)
    euribor_adj_alq =  np.clip((euribor - 2.5) / 100.0 * 0.4, -0.01, 0.02)
    paro_adj        = -np.clip((paro  - 10.0) / 100.0 * 0.5, -0.02, 0.03)

    lambda_var = 0.15
    A_var      = _modelo['A_var']
    L_base     = _modelo['L']
    corr_base  = _modelo['corr']
    vol_base   = _modelo['vol_cal']
    drift_v    = _modelo['drift_venta']
    drift_a    = _modelo['drift_alquiler']

    n_anos  = len(anos_proy_loc)
    np.random.seed(42)
    sim_vta = np.zeros((n_sim, n_distritos, n_anos))
    sim_alq = np.zeros((n_sim, n_distritos, n_anos))

    for sim in range(n_sim):
        p_vta  = pv_base.copy()
        p_alq  = pa_base.copy()
        estado = 0
        r_prev = drift_v.copy()

        for i, ano in enumerate(anos_proy_loc):
            # Transicion Markov
            rand, cum = np.random.random(), 0.0
            for j in range(3):
                cum += mtm[estado, j]
                if rand < cum:
                    estado = j
                    break

            # Drift endogeno
            var_comp = lambda_var * (A_var @ r_prev)
            dv = drift_v + var_comp + ipc_real_adj + euribor_adj_vta + paro_adj
            da = drift_a + var_comp * 0.7 + ipc_real_adj * 0.6 + euribor_adj_alq

            # Modulacion por estado
            if estado == 1:    # Boom
                tasa_v = dv * 1.35 + np.random.normal(0, 0.008, n_distritos)
                tasa_a = da * 1.25 + np.random.normal(0, 0.006, n_distritos)
            elif estado == 2:  # Crisis
                tasa_v = -np.abs(dv) * 0.6 - 0.015 + np.random.normal(0, 0.012, n_distritos)
                tasa_a = -np.abs(da) * 0.4 - 0.008 + np.random.normal(0, 0.008, n_distritos)
            else:              # Normal
                tasa_v = dv + np.random.normal(0, 0.008, n_distritos)
                tasa_a = da + np.random.normal(0, 0.006, n_distritos)

            tasa_v *= sensibilidad_distrital
            tasa_a *= sensibilidad_distrital * 0.8

            # DCC simplificado
            if estado == 2:
                vol_dyn = vol_base * 1.60
                cd = corr_base * 0.5 + np.ones_like(corr_base) * 0.5
                np.fill_diagonal(cd, 1.0)
                try:
                    ev, evec = np.linalg.eigh(cd)
                    ev = np.maximum(ev, 1e-8)
                    cd = evec @ np.diag(ev) @ evec.T
                    dd = np.sqrt(np.diag(cd))
                    cd = cd / np.outer(dd, dd)
                    L_dyn = np.linalg.cholesky(cd)
                except Exception:
                    L_dyn = L_base
            elif estado == 1:
                vol_dyn = vol_base * 0.85
                cd = corr_base * 0.8 + np.eye(n_distritos) * 0.2
                try:
                    L_dyn = np.linalg.cholesky(cd)
                except Exception:
                    L_dyn = L_base
            else:
                vol_dyn = vol_base
                L_dyn   = L_base

            eps    = L_dyn @ np.random.normal(0, 1, n_distritos)
            tasa_v += vol_dyn * eps
            tasa_a += vol_dyn * 0.75 * eps
            tasa_v  = np.maximum(tasa_v, -0.07)

            # Floor yield 2.5%
            yld    = (p_alq * 12) / p_vta
            resist = np.clip(yld / 0.025, 0.5, 1.0)
            tasa_v = np.where(tasa_v > 0, tasa_v * resist, tasa_v)

            # Feedback alquiler -> venta
            if i > 0:
                pa_prev  = sim_alq[sim, :, i - 1]
                pa_prev2 = pa_base if i == 1 else sim_alq[sim, :, i - 2]
                safe2    = np.where(pa_prev2 > 0, pa_prev2, 1.0)
                tasa_v  += rho_vta_alq * ((pa_prev / safe2) - 1)

            # Shock regulatorio VT
            shk_v = shk_a = 0.0
            if shock_vt and ano == shock_ano:
                factor = concentracion_vt_real / 0.46
                shk_v  = -0.10 * factor
                shk_a  = -0.06 * factor

            p_vta = p_vta * (1 + tasa_v) * (1 + shk_v)
            p_alq = p_alq * (1 + tasa_a) * (1 + shk_a)

            sim_vta[sim, :, i] = p_vta
            sim_alq[sim, :, i] = p_alq

            r_prev = np.clip(dv.copy(), -0.08, 0.08)

    return sim_vta, sim_alq, anos_proy_loc


def simular(params: dict, modelo: dict, n_sim: int = 1000):
    h = _params_hash({**params, 'n_sim': n_sim})
    return simular_cached(h, modelo)


# ============================================================================
# 6. TESTS UNITARIOS
# ============================================================================

def run_tests(modelo: dict, ds_key: str):
    ds   = DATASETS[ds_key]
    pv0  = ds['venta'][:, -1]
    pa0  = ds['alquiler'][:, -1]
    results = []

    ok = np.all(pv0 > 0) and np.all(pa0 > 0)
    results.append(("Precios base positivos", ok,
                    f"Venta min={pv0.min():.0f} | Alquiler min={pa0.min():.2f}"))

    yields = (pa0 * 12) / pv0
    ok = np.all((yields >= 0.02) & (yields <= 0.12))
    results.append(("Yields base en rango [2%-12%]", ok,
                    f"min={yields.min()*100:.1f}% max={yields.max()*100:.1f}%"))

    ok = np.all(np.abs(modelo['drift_venta']) < 0.30)
    results.append(("Drift historico plausible (<30%/a)", ok,
                    f"{(modelo['drift_venta']*100).round(2)}%"))

    rho = modelo['rho_espectral']
    ok  = rho <= 0.90
    results.append((f"Radio espectral A_var <= 0.90", ok,
                    f"Radio espectral: {rho:.4f} | Metodo: {modelo['metodo_var']}"))

    betas = modelo['betas']
    ok = np.all((betas >= 0.5) & (betas <= 1.5))
    results.append(("Betas CAPM en [0.5, 1.5]", ok,
                    f"min={betas.min():.3f} max={betas.max():.3f}"))

    L    = modelo['L']
    ok_l = np.allclose(L, np.tril(L), atol=1e-10) and np.all(np.diag(L) > 0)
    results.append(("Cholesky triangular inferior y diagonal positiva", ok_l,
                    f"Lower:{np.allclose(L, np.tril(L))} | PosDiag:{np.all(np.diag(L)>0)}"))

    recon   = L @ L.T
    max_err = np.abs(modelo['corr'] - recon).max()
    results.append(("Cholesky: L*L^T aprox Sigma", max_err < 1e-4,
                    f"Error max={max_err:.2e}"))

    for reg, (eu, ic, pa) in [
        ("Crecimiento",  (1.5, 2.5,  9.0)),
        ("Estanflacion", (3.8, 5.5, 13.0)),
        ("Recesion",     (2.5, 1.0, 20.0)),
    ]:
        mtm = construir_mtm_macro(eu, ic, pa)
        ok  = np.allclose(mtm.sum(axis=1), 1.0, atol=1e-6)
        results.append((f"MTM dinamica '{reg}' filas=1", ok,
                        f"Sumas: {mtm.sum(axis=1).round(6)}"))

    return results


# ============================================================================
# 7. GESTION DE ESTADO SESSION
# ============================================================================
if 'ds_key'       not in st.session_state:
    st.session_state.ds_key       = 'Datos Portales Inmobiliarios'
if 'regimen'      not in st.session_state:
    st.session_state.regimen      = 'Crecimiento'
if 'fig_distrito' not in st.session_state:
    st.session_state.fig_distrito = None
if 'idx_distrito' not in st.session_state:
    st.session_state.idx_distrito = 1


def reset_macro():
    reg = st.session_state.regimen_selector
    st.session_state.regimen = reg


# ============================================================================
# 8. SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("Configuracion del Modelo")

    # --- Set de datos ---
    st.subheader("Fuente de Datos")
    ds_key = st.radio(
        "Set de datos:",
        list(DATASETS.keys()),
        index=list(DATASETS.keys()).index(st.session_state.ds_key),
        key="ds_selector",
    )
    st.session_state.ds_key = ds_key
    ds_activo = DATASETS[ds_key]
    st.caption(f"Fuente: {ds_activo['fuente']} | Ultimo dato: {ds_activo['fecha_dato']}")
    st.caption(f"Venta: {ds_activo['anos_venta'][0]}-{ds_activo['anos_venta'][-1]} "
               f"| Alquiler: {ds_activo['anos_alq'][0]}-{ds_activo['anos_alq'][-1]}")

    st.markdown("---")

    # --- Régimen ---
    reg_sel = st.radio(
        "Regimen Economico:",
        ("Crecimiento", "Estanflacion", "Recesion (Estructural)"),
        key="regimen_selector",
        on_change=reset_macro,
    )
    st.markdown("---")

    # --- Indicadores macro ---
    st.subheader("Indicadores Macroeconomicos")
    defaults = MACRO_DEFAULTS.get(reg_sel, MACRO_DEFAULTS["Crecimiento"])
    euribor_val = st.slider("Euribor 12m (%)", 0.0, 6.0,
                             float(defaults['euribor']), 0.1)
    ipc_val     = st.slider("IPC anual (%)",   -1.0, 10.0,
                             float(defaults['ipc']),     0.1)
    paro_val    = st.slider("Tasa paro Barcelona (%)", 4.0, 30.0,
                             float(defaults['paro']),    0.5)

    probs = calcular_probabilidades_regimen(euribor_val, ipc_val, paro_val)
    st.markdown("**Probabilidades de regimen (endogenas):**")
    cp1, cp2, cp3 = st.columns(3)
    cp1.metric("Normal", f"{probs[0]*100:.0f}%")
    cp2.metric("Boom",   f"{probs[1]*100:.0f}%")
    cp3.metric("Crisis", f"{probs[2]*100:.0f}%")

    st.markdown("---")

    # --- Shock regulatorio ---
    st.subheader("Shock Ley Vivienda")
    shock_vt  = st.checkbox("Activar Shock Regulatorio VT", value=False)
    shock_ano = st.radio("Año del shock:", [2028, 2029],
                          index=1, horizontal=True,
                          disabled=not shock_vt)

    st.markdown("---")

    # --- Precisión ---
    st.subheader("Precision de Simulacion")
    n_sim_ui = st.select_slider("Simulaciones (n_sim):",
                                 options=[200, 500, 1000, 2000, 5000],
                                 value=1000)
    st.caption(f"Tiempo estimado: ~{n_sim_ui // 500 + 1}s")

    params_act = {
        'dataset_key': ds_key,
        'regimen':     reg_sel,
        'euribor':     euribor_val,
        'ipc':         ipc_val,
        'paro':        paro_val,
        'shock_vt':    shock_vt,
        'shock_ano':   shock_ano,
        'n_sim':       n_sim_ui,
    }

# ============================================================================
# 9. CALIBRACION DEL MODELO
# ============================================================================
with st.spinner("Calibrando modelo estructural..."):
    modelo_act = inicializar_modelo(ds_key)
    # Modelo del set alternativo (para comparativa en informe)
    ds_key_alt = [k for k in DATASETS.keys() if k != ds_key][0]
    modelo_alt = inicializar_modelo(ds_key_alt)

# ============================================================================
# 10. SIMULACIONES
# ============================================================================
with st.spinner(f"Simulando {n_sim_ui} trayectorias MS-VAR ({ds_key})..."):
    sv_act, sa_act, anos_proy_act = simular(params_act, modelo_act, n_sim_ui)

    # Escenarios de referencia (n_sim=200, dataset activo)
    def sim_ref(regimen_r, eu, ic, pa):
        p = {'dataset_key': ds_key, 'regimen': regimen_r,
             'euribor': eu, 'ipc': ic, 'paro': pa,
             'shock_vt': shock_vt, 'shock_ano': shock_ano}
        return simular(p, modelo_act, 200)

    sv_crec, sa_crec, _ = sim_ref("Crecimiento",  1.5, 2.5,  9.0)
    sv_est,  sa_est,  _ = sim_ref("Estanflacion", 3.8, 5.5, 13.0)
    sv_rec,  sa_rec,  _ = sim_ref("Recesion",     2.5, 1.0, 20.0)

    # Simulación del set alternativo con mismos params macro (para comparativa)
    params_alt = {**params_act, 'dataset_key': ds_key_alt}
    sv_alt, sa_alt, anos_proy_alt = simular(params_alt, modelo_alt, 500)

scenarios_data = {
    "Crecimiento":      (sv_crec, sa_crec, _),
    "Estanflacion":     (sv_est,  sa_est,  _),
    "Recesion":         (sv_rec,  sa_rec,  _),
    "Usuario (Activo)": (sv_act,  sa_act,  anos_proy_act),
}

# ============================================================================
# 11. CABECERA DASHBOARD
# ============================================================================
st.title("Barcelona Strategic Model v33.0")
st.markdown(
    f"**MS-VAR | Markov no homogeneo | DCC simplificado | "
    f"{ds_activo['fecha_dato']} | {ds_activo['fuente']} | "
    f"Metodo estimacion: {modelo_act['metodo_var']}**"
)
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Regimen",    reg_sel.replace(" (Estructural)",""))
c2.metric("Euribor",    f"{euribor_val:.1f}%")
c3.metric("IPC",        f"{ipc_val:.1f}%")
c4.metric("Paro BCN",   f"{paro_val:.1f}%")
c5.metric("Shock VT",   f"{shock_ano}" if shock_vt else "OFF")
c6.metric("n_sim",      str(n_sim_ui))
c7.metric("Datos",      "Portales" if "Portal" in ds_key else "Ayuntamiento")

# ============================================================================
# 12. TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Analisis Distrito", "Riesgo / Retorno",
    "Macro & Regimenes", "Tests del Modelo"
])

# ---  TAB 1 ---
with tab1:
    sel_dist = st.selectbox("Distrito:", distritos,
                             index=st.session_state.idx_distrito)
    idx = distritos.index(sel_dist)
    st.session_state.idx_distrito = idx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Venta
    rec = modelo_act['reconstruidos']
    ax1.plot(anos_macro, rec[idx], 'o-', color='#2c3e50',
             label='Historico/Rec.', linewidth=1.5, markersize=3)
    ax1.axvline(x=ds_activo['ano_fin'], color='navy',
                linestyle='--', linewidth=0.8, alpha=0.6,
                label=f"Dato real {ds_activo['ano_fin']}")

    p50  = np.percentile(sv_act[:, idx, :], 50, axis=0)
    p10  = np.percentile(sv_act[:, idx, :], 10, axis=0)
    p90  = np.percentile(sv_act[:, idx, :], 90, axis=0)
    last = ds_activo['venta'][idx, -1]
    anos_plot = np.concatenate([[ds_activo['ano_fin']], anos_proy_act])

    ax1.plot(anos_plot, np.concatenate([[last], p50]),
             '--', color='#e74c3c', linewidth=2, label='P50 activo')
    ax1.fill_between(anos_plot,
                     np.concatenate([[last], p10]),
                     np.concatenate([[last], p90]),
                     color='#e74c3c', alpha=0.15, label='IC P10-P90')
    ax1.plot(anos_plot,
             np.concatenate([[last], np.percentile(sv_crec[:,idx,:], 50, axis=0)]),
             ':', color='green', linewidth=1.2, label='Crecimiento')
    ax1.plot(anos_plot,
             np.concatenate([[last], np.percentile(sv_rec[:,idx,:], 50, axis=0)]),
             ':', color='black', linewidth=1.2, label='Recesion')
    ax1.set_title(f"Venta EUR/m2 - {sel_dist}", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(fontsize='x-small')
    ax1.set_xlabel("Ano")

    # Alquiler
    pa_hist = ds_activo['alquiler']
    anos_alq = ds_activo['anos_alq']
    ax2.plot(anos_alq, pa_hist[idx], 'o-', color='#2c3e50',
             linewidth=1.5, markersize=3)
    ax2.axvline(x=ds_activo['ano_fin'], color='navy',
                linestyle='--', linewidth=0.8, alpha=0.6)
    p50a = np.percentile(sa_act[:, idx, :], 50, axis=0)
    p10a = np.percentile(sa_act[:, idx, :], 10, axis=0)
    p90a = np.percentile(sa_act[:, idx, :], 90, axis=0)
    lasta = pa_hist[idx, -1]

    ax2.plot(anos_plot, np.concatenate([[lasta], p50a]),
             '--', color='#27ae60', linewidth=2, label='P50 activo')
    ax2.fill_between(anos_plot,
                     np.concatenate([[lasta], p10a]),
                     np.concatenate([[lasta], p90a]),
                     color='#27ae60', alpha=0.15, label='IC P10-P90')
    ax2.set_title(f"Alquiler EUR/m2/mes - {sel_dist}", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(fontsize='x-small')
    ax2.set_xlabel("Ano")

    plt.tight_layout()
    st.session_state.fig_distrito = fig
    st.pyplot(fig)

    v_fin = np.median(sv_act[:, idx, -1])
    a_fin = np.median(sa_act[:, idx, -1])
    yld   = (a_fin * 12) / v_fin * 100
    v_base = ds_activo['venta'][idx, -1]
    a_base = ds_activo['alquiler'][idx, -1]
    n_yr   = 2032 - ds_activo['ano_fin']

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Venta 2032",    f"{int(v_fin):,} EUR",
              f"{((v_fin/v_base)-1)*100:.1f}%")
    k2.metric("Alquiler 2032", f"{a_fin:.1f} EUR",
              f"{((a_fin/a_base)-1)*100:.1f}%")
    k3.metric("Yield 2032",    f"{yld:.1f}%")
    k4.metric("Drift hist. venta",
              f"{modelo_act['drift_venta'][idx]*100:.2f}%/a")
    k5.metric("Beta CAPM",     f"{modelo_act['betas'][idx]:.3f}")

    st.markdown(f"#### Precios Base {ds_activo['ano_fin']} — Todos los Distritos")
    pv0 = ds_activo['venta'][:, -1]
    pa0 = ds_activo['alquiler'][:, -1]
    df_p = pd.DataFrame({
        'Distrito':          distritos,
        f"Venta {ds_activo['ano_fin']} EUR/m2": pv0.astype(int),
        f"Alquiler {ds_activo['ano_fin']} EUR/m2": pa0,
        'Yield bruto':  [f"{(a*12/v*100):.1f}%" for v,a in zip(pv0,pa0)],
        'Drift hist.':  [f"{d*100:.2f}%" for d in modelo_act['drift_venta']],
        'Beta CAPM':    modelo_act['betas'].round(3),
    })
    st.dataframe(df_p.set_index('Distrito'), use_container_width=True)


# --- TAB 2 ---
with tab2:
    pv0 = ds_activo['venta'][:, -1]
    cagrs, vols = [], []
    n_yr = 2032 - ds_activo['ano_fin']
    for i in range(n_distritos):
        f = sv_act[:, i, -1]
        cagrs.append(((f.mean() / pv0[i]) ** (1/n_yr) - 1) * 100)
        vols.append(f.std() / f.mean() * 100)

    df_r = pd.DataFrame({
        'Distrito':         distritos,
        'Retorno CAGR (%)': cagrs,
        'Riesgo CV (%)':    vols,
    })
    fig_r = px.scatter(
        df_r, x='Riesgo CV (%)', y='Retorno CAGR (%)',
        text='Distrito', color='Retorno CAGR (%)',
        color_continuous_scale='RdYlGn', size=[20]*10,
        title=f"Riesgo/Retorno | {reg_sel} | "
              f"Euribor={euribor_val}% | Paro={paro_val}%"
    )
    mr = df_r['Riesgo CV (%)'].mean()
    mt = df_r['Retorno CAGR (%)'].mean()
    fig_r.add_vline(x=mr, line_dash="dash", line_color="red", line_width=1.5)
    fig_r.add_hline(y=mt, line_dash="dash", line_color="red", line_width=1.5)
    fig_r.add_annotation(x=df_r['Riesgo CV (%)'].min(), y=df_r['Retorno CAGR (%)'].max(),
                         text="ESTRELLAS (Buy)", showarrow=False,
                         font=dict(color="green", size=11))
    fig_r.add_annotation(x=df_r['Riesgo CV (%)'].max(), y=df_r['Retorno CAGR (%)'].min(),
                         text="INEFICIENTES (Sell)", showarrow=False,
                         font=dict(color="red", size=11))
    st.plotly_chart(fig_r, use_container_width=True)
    st.dataframe(df_r.style.format({'Retorno CAGR (%)':'{:.2f}%',
                                     'Riesgo CV (%)':'{:.2f}%'})
                 .background_gradient(cmap="Greens", subset=["Retorno CAGR (%)"]),
                 use_container_width=True)


# --- TAB 3 ---
with tab3:
    st.subheader("Motor Macro: Indicadores -> Probabilidades de Regimen")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.markdown("**MTM no homogenea activa**")
        mtm_act = construir_mtm_macro(euribor_val, ipc_val, paro_val)
        df_mtm  = pd.DataFrame(
            mtm_act.round(4),
            index=['Desde Normal', 'Desde Boom', 'Desde Crisis'],
            columns=['-> Normal', '-> Boom', '-> Crisis']
        )
        st.dataframe(df_mtm.style.background_gradient(cmap='Blues'),
                     use_container_width=True)
    with col_m2:
        eu_range = np.linspace(0.5, 5.5, 30)
        pb = [calcular_probabilidades_regimen(e, ipc_val, paro_val)[1]*100 for e in eu_range]
        pc = [calcular_probabilidades_regimen(e, ipc_val, paro_val)[2]*100 for e in eu_range]
        fig_m, ax_m = plt.subplots(figsize=(7, 3.5))
        ax_m.plot(eu_range, pb, color='green', linewidth=2, label='P(Boom)')
        ax_m.plot(eu_range, pc, color='red',   linewidth=2, label='P(Crisis)')
        ax_m.axvline(x=euribor_val, color='navy', linestyle='--',
                     linewidth=1.2, label=f'Euribor ({euribor_val}%)')
        ax_m.set_xlabel("Euribor 12m (%)"); ax_m.set_ylabel("Probabilidad (%)")
        ax_m.set_title("Sensibilidad regimen al Euribor")
        ax_m.legend(fontsize='small'); ax_m.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_m)

    st.markdown("---")
    st.subheader("Parametros calibrados por dataset activo")
    df_cal = pd.DataFrame({
        'Distrito':           distritos,
        'Drift venta %/a':    (modelo_act['drift_venta']*100).round(3),
        'Drift alquiler %/a': (modelo_act['drift_alquiler']*100).round(3),
        'Beta CAPM':          modelo_act['betas'].round(3),
        'Vol. calibrada %':   (modelo_act['vol_cal']*100).round(2),
    })
    st.dataframe(df_cal.set_index('Distrito'), use_container_width=True)
    st.caption(f"Metodo estimacion A_var: **{modelo_act['metodo_var']}** | "
               f"Radio espectral: {modelo_act['rho_espectral']:.4f}")

    st.markdown("---")
    st.subheader("Comparativa de drift entre sets de datos")
    df_comp_drift = pd.DataFrame({
        'Distrito': distritos,
        f"Drift Portales %/a":     (modelo_act['drift_venta']*100).round(3)
                                    if 'Portal' in ds_key
                                    else (modelo_alt['drift_venta']*100).round(3),
        f"Drift Ayuntamiento %/a": (modelo_alt['drift_venta']*100).round(3)
                                    if 'Portal' in ds_key
                                    else (modelo_act['drift_venta']*100).round(3),
    })
    st.dataframe(df_comp_drift.set_index('Distrito'), use_container_width=True)

    st.markdown("---")
    st.subheader("Matriz de correlacion historica")
    fig_corr, ax_c = plt.subplots(figsize=(8, 6))
    im = ax_c.imshow(modelo_act['corr'], cmap='RdYlGn', vmin=-1, vmax=1)
    ax_c.set_xticks(range(n_distritos))
    ax_c.set_xticklabels(distritos, rotation=45, ha='right', fontsize=7)
    ax_c.set_yticks(range(n_distritos))
    ax_c.set_yticklabels(distritos, fontsize=7)
    plt.colorbar(im, ax=ax_c)
    ax_c.set_title(f"Correlacion de Retornos — {ds_key}")
    for r in range(n_distritos):
        for c in range(n_distritos):
            ax_c.text(c, r, f"{modelo_act['corr'][r,c]:.2f}",
                      ha='center', va='center', fontsize=5)
    plt.tight_layout()
    st.pyplot(fig_corr)


# --- TAB 4 ---
with tab4:
    st.subheader(f"Tests del Motor — {ds_key}")
    test_results = run_tests(modelo_act, ds_key)
    n_pass = sum(1 for _, p, _ in test_results if p)
    total  = len(test_results)
    st.metric("Tests pasados", f"{n_pass}/{total}",
              delta="OK" if n_pass == total else "REVISAR")
    for tname, tpass, tmsg in test_results:
        with st.expander(f"[{'OK' if tpass else 'FAIL'}] {tname}"):
            st.code(tmsg)


# ============================================================================
# 13. UTILIDADES PDF
# ============================================================================

def clean_str(txt: str) -> str:
    repls = {
        '€':'EUR','á':'a','Á':'A','à':'a','â':'a','ä':'a',
        'é':'e','É':'E','è':'e','ê':'e','ë':'e',
        'í':'i','Í':'I','ì':'i','î':'i','ï':'i',
        'ó':'o','Ó':'O','ò':'o','ô':'o','ö':'o',
        'ú':'u','Ú':'U','ù':'u','û':'u','ü':'u',
        'ñ':'n','Ñ':'N','ç':'c','Ç':'C',
        '²':'2','–':'-','—':'-','−':'-',
        '\u2019':"'",'\u2018':"'",'\u201c':'"','\u201d':'"',
        '\u00ab':'"','\u00bb':'"','•':'-','·':'.',
        '≈':'~','≥':'>=','≤':'<=','×':'x','÷':'/',
        '\u00a0':' ','\u00ad':'','\u2026':'...','→':'->',
        'à':'a','ï':'i','ü':'u','ú':'u',
    }
    for k,v in repls.items():
        txt = txt.replace(k, v)
    return txt.encode('latin-1', errors='replace').decode('latin-1')


class ProfessionalPDF(FPDF):
    def header(self):
        self.set_font('Times','B',9)
        self.cell(0,8,'BCN STRATEGIC MODEL V33.0 | CONFIDENCIAL',0,1,'C')
        self.line(10,17,200,17)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times','I',8)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0,10,
                  f'Pagina {self.page_no()} | {ts} | BCN Model v33.0 | Confidencial',
                  0,0,'C')


def _encabezado_tabla(pdf, headers, widths):
    pdf.set_fill_color(44,62,80)
    pdf.set_text_color(255,255,255)
    pdf.set_font('Arial','B',8)
    for h,w in zip(headers,widths):
        pdf.cell(w,7,clean_str(h),1,0,'C',1)
    pdf.ln()
    pdf.set_text_color(0,0,0)
    pdf.set_font('Arial','',8)


def _bloque_trazabilidad(pdf, params, dist_name, n_sim, modelo):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    probs = calcular_probabilidades_regimen(
        params.get('euribor',2.0), params.get('ipc',2.5), params.get('paro',10.0))
    pdf.set_font('Arial','B',10)
    pdf.cell(0,7,clean_str("TRAZABILIDAD DEL ESCENARIO"),0,1)
    pdf.set_font('Courier','',8)
    for ln in [
        f"Timestamp              : {ts}",
        f"Modelo                 : Barcelona Strategic Model v33.0",
        f"Distrito               : {dist_name}",
        f"Set de datos activo    : {params.get('dataset_key','N/A')}",
        f"Fuente                 : {DATASETS[params['dataset_key']]['fuente']}",
        f"Ultimo dato disponible : {DATASETS[params['dataset_key']]['fecha_dato']}",
        f"Regimen economico      : {params.get('regimen','N/A')}",
        f"Euribor 12m            : {params.get('euribor',0):.2f}%",
        f"IPC anual              : {params.get('ipc',0):.2f}%",
        f"Tasa de paro           : {params.get('paro',0):.1f}%",
        f"Shock VT               : {'Activado (' + str(params.get('shock_ano',2029)) + ')' if params.get('shock_vt') else 'Desactivado'}",
        f"Simulaciones (n_sim)   : {n_sim}",
        f"Horizonte proyeccion   : {DATASETS[params['dataset_key']]['ano_fin']+1}-2032",
        f"Motor estimacion A_var : {modelo['metodo_var']}",
        f"Radio espectral A_var  : {modelo['rho_espectral']:.4f}",
        f"P(Normal/Boom/Crisis)  : {probs[0]*100:.1f}% / {probs[1]*100:.1f}% / {probs[2]*100:.1f}%",
    ]:
        pdf.cell(0,5,clean_str(ln),0,1)
    pdf.ln(3)


def _noshock_approx(sv, sa, dist_idx):
    conc = concentracion_vt_real[dist_idx]
    fv   = (conc / 0.46) * 0.14
    fa   = (conc / 0.46) * 0.10
    v    = np.median(sv[:, dist_idx, -1])
    a    = np.median(sa[:, dist_idx, -1])
    return v / max(1 - fv, 0.01), a / max(1 - fa, 0.01)


# ============================================================================
# 14. INFORME EJECUTIVO CON COMPARATIVA
# ============================================================================

def generate_exec_report(dist_idx, params, fig_chart, n_sim,
                          modelo_ppal, sv_ppal, sa_ppal, anos_ppal,
                          modelo_comp, sv_comp, sa_comp, anos_comp) -> bytes:

    ds_ppal = DATASETS[params['dataset_key']]
    ds_comp = DATASETS[ds_key_alt]
    probs   = calcular_probabilidades_regimen(
        params.get('euribor',2.0), params.get('ipc',2.5), params.get('paro',10.0))

    pdf = ProfessionalPDF()
    pdf.add_page()

    # Portada
    pdf.set_font('Arial','B',15)
    pdf.set_fill_color(44,62,80)
    pdf.set_text_color(255)
    pdf.cell(0,11,clean_str(
        f"INFORME EJECUTIVO: {distritos[dist_idx].upper()}"),0,1,'C',1)
    pdf.set_text_color(0)
    pdf.set_font('Arial','I',9)
    pdf.cell(0,6,clean_str(
        f"Barcelona Strategic Model v33.0 | Set activo: {params['dataset_key']} | "
        f"Motor: MS-VAR + Markov no homogeneo + DCC"),0,1,'C')
    pdf.ln(4)

    _bloque_trazabilidad(pdf, params, distritos[dist_idx], n_sim, modelo_ppal)

    # --- 1. Params macro ---
    pdf.set_font('Arial','B',11)
    pdf.cell(0,8,clean_str("1. PARAMETROS MACROECONOMICOS ACTIVOS"),0,1)
    pdf.set_font('Arial','',10)
    pdf.multi_cell(0,5,clean_str(
        f"Regimen: {params.get('regimen','N/A')} | "
        f"Euribor: {params.get('euribor',0):.2f}% | "
        f"IPC: {params.get('ipc',0):.2f}% | "
        f"Paro: {params.get('paro',0):.1f}%\n"
        f"Probabilidades endogenas: Normal={probs[0]*100:.1f}% | "
        f"Boom={probs[1]*100:.1f}% | Crisis={probs[2]*100:.1f}%\n"
        f"Drift historico venta ({distritos[dist_idx]}): "
        f"{modelo_ppal['drift_venta'][dist_idx]*100:.2f}%/a | "
        f"Beta CAPM: {modelo_ppal['betas'][dist_idx]:.3f} | "
        f"Vol. calibrada: {modelo_ppal['vol_cal'][dist_idx]*100:.2f}%\n"
        f"Metodo estimacion: {modelo_ppal['metodo_var']}"
    ))
    pdf.ln(3)

    # --- 2. Sensibilidad regulatoria ---
    pdf.set_font('Arial','B',11)
    pdf.cell(0,8,clean_str("2. SENSIBILIDAD REGULATORIA — IMPACTO SHOCK VT"),0,1)
    pdf.set_font('Arial','',9)
    shock_txt = (f"Activado en {params.get('shock_ano',2029)}"
                 if params.get('shock_vt') else "Desactivado")
    pdf.cell(0,5,clean_str(f"Estado del shock: {shock_txt}"),0,1)
    pdf.ln(1)

    hdrs = ["Escenario","Venta'32 (Shock)","Venta'32 (Sin Shock)","Alq'32 (Shock)","Alq'32 (Sin Shock)"]
    wdts = [35,38,38,38,38]
    _encabezado_tabla(pdf,hdrs,wdts)
    for sc_name, (sv_s, sa_s, _ap) in [
        ("Crecimiento", (sv_crec, sa_crec, None)),
        ("Estanflacion",(sv_est,  sa_est,  None)),
        ("Recesion",    (sv_rec,  sa_rec,  None)),
    ]:
        v32 = np.median(sv_s[:,dist_idx,-1])
        a32 = np.median(sa_s[:,dist_idx,-1])
        vns, ans = _noshock_approx(sv_s, sa_s, dist_idx)
        pdf.set_fill_color(245,245,245)
        for val,w in zip([clean_str(sc_name),
                          f"{int(v32):,}",f"{int(vns):,}",
                          f"{a32:.1f}",f"{ans:.1f}"], wdts):
            pdf.cell(w,6,val,1,0,'C',1)
        pdf.ln()
    pdf.ln(3)

    # --- 3. Gráfico ---
    pdf.set_font('Arial','B',11)
    pdf.cell(0,8,clean_str("3. PROYECCION GRAFICA (Horizonte a 2032)"),0,1)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig_chart.savefig(tmp.name, dpi=100, bbox_inches='tight')
        pdf.image(tmp.name, x=10, w=185)

    # --- 4. Tabla maestra ---
    pdf.add_page()
    pdf.set_font('Arial','B',11)
    pdf.cell(0,8,clean_str(
        f"4. TABLA MAESTRA — TODOS LOS DISTRITOS "
        f"({ds_ppal['ano_fin']} vs 2032) | {params['dataset_key']}"),0,1)
    pv0 = ds_ppal['venta'][:,-1]
    pa0 = ds_ppal['alquiler'][:,-1]
    hdrs2 = ["Distrito","Venta Base","Venta 2032","Var%",
             "Alq Base","Alq 2032","Yield'32","Beta","Drift%"]
    wdts2 = [40,22,22,16,20,20,18,14,15]
    _encabezado_tabla(pdf,hdrs2,wdts2)
    for i,d in enumerate(distritos):
        v32 = np.median(sv_ppal[:,i,-1])
        a32 = np.median(sa_ppal[:,i,-1])
        var = ((v32/pv0[i])-1)*100
        yld = (a32*12)/v32*100
        pdf.set_fill_color(240 if i%2==0 else 255)
        for val,w in zip([
            clean_str(d), f"{int(pv0[i]):,}", f"{int(v32):,}",
            f"{var:+.1f}%", f"{pa0[i]:.1f}", f"{a32:.1f}",
            f"{yld:.1f}%", f"{modelo_ppal['betas'][i]:.3f}",
            f"{modelo_ppal['drift_venta'][i]*100:.2f}%"
        ], wdts2):
            pdf.cell(w,6,val,1,0,'C',1)
        pdf.ln()

    # --- 5. Recomendación ---
    pdf.ln(4)
    pdf.set_font('Arial','B',11)
    pdf.cell(0,8,clean_str("5. RECOMENDACION AL INVERSOR"),0,1)
    pdf.set_font('Arial','',10)
    pdf.multi_cell(0,5,clean_str(
        f"Basado en {params['dataset_key']} | Motor v33.0 | "
        f"Datos base: {ds_ppal['fecha_dato']}\n\n"
        "- YIELD: Nou Barris y Sant Andreu presentan yields superiores al 5%.\n"
        "  Baja exposicion VT y demanda residencial estable.\n\n"
        "- PLUSVALIA: Eixample y Gracia lideran en Crecimiento por beta elevado.\n"
        "  En estanflacion o recesion este liderazgo se invierte.\n\n"
        "- RIESGO REGULATORIO: Shock VT asimetrico segun densidad.\n"
        "  Eixample (VT=0.46): hasta -10% venta. Nou Barris (VT=0.004): inmune.\n\n"
        f"- MACRO ACTUAL: Euribor={params.get('euribor',0):.1f}% comprime demanda de\n"
        "  compra y desplaza hacia alquiler. P(Crisis) elevada con paro alto.\n\n"
        "NOTA: Proyecciones sobre mediana (P50). Revisar IC P10-P90 para\n"
        "rango completo de incertidumbre."
    ))

    # =========================================================
    # --- 6. COMPARATIVA ENTRE SETS DE DATOS ---
    # =========================================================
    pdf.add_page()
    pdf.set_font('Arial','B',13)
    pdf.set_fill_color(44,62,80)
    pdf.set_text_color(255)
    pdf.cell(0,9,clean_str(
        "6. COMPARATIVA DE RESULTADOS: AMBOS SETS DE DATOS"),0,1,'C',1)
    pdf.set_text_color(0)
    pdf.ln(4)

    pdf.set_font('Arial','',9)
    pdf.multi_cell(0,5,clean_str(
        "La siguiente seccion presenta los resultados proyectados al 2032 usando "
        "identicos parametros macroeconomicos y el mismo motor de calculo, pero "
        "calibrado sobre cada set de datos de forma independiente. Las diferencias "
        "reflejan la distinta naturaleza de cada fuente: los portales inmobiliarios "
        "reportan precios de oferta (asking price, banda alta), mientras que los datos "
        "del Ayuntamiento reflejan estadisticas de transacciones reales. "
        "Mismos parametros macro: "
        f"Regimen={params.get('regimen','N/A')} | "
        f"Euribor={params.get('euribor',0):.1f}% | "
        f"IPC={params.get('ipc',0):.1f}% | "
        f"Paro={params.get('paro',0):.1f}%"
    ))
    pdf.ln(3)

    # Tabla comparativa por distrito
    pdf.set_font('Arial','B',10)
    pdf.cell(0,7,clean_str(
        f"Proyeccion Venta EUR/m2 a 2032 — "
        f"Portales (base {DATASET_A['ano_fin']}) vs "
        f"Ayuntamiento (base {DATASET_B['ano_fin']})"),0,1)

    hdrs_c = ["Distrito",
              f"Portales Base ({DATASET_A['ano_fin']})",
              "Portales 2032 (P50)",
              "Portales CAGR",
              f"Ayto Base ({DATASET_B['ano_fin']})",
              "Ayto 2032 (P50)",
              "Ayto CAGR"]
    wdts_c = [42, 26, 24, 20, 26, 24, 20]
    _encabezado_tabla(pdf, hdrs_c, wdts_c)

    pv_port = DATASET_A['venta'][:,-1]
    pv_ayto = DATASET_B['venta'][:,-1]
    n_yr_port = 2032 - DATASET_A['ano_fin']
    n_yr_ayto = 2032 - DATASET_B['ano_fin']

    # Identificar qué simulacion es de portales y cuál de ayuntamiento
    if 'Portal' in params['dataset_key']:
        sv_port, sa_port = sv_ppal, sa_ppal
        sv_ayto_r, sa_ayto_r = sv_comp, sa_comp
    else:
        sv_ayto_r, sa_ayto_r = sv_ppal, sa_ppal
        sv_port, sa_port = sv_comp, sa_comp

    for i, d in enumerate(distritos):
        v32_port = np.median(sv_port[:, i, -1])
        v32_ayto = np.median(sv_ayto_r[:, i, -1])
        cagr_port = ((v32_port / pv_port[i]) ** (1/n_yr_port) - 1) * 100
        cagr_ayto = ((v32_ayto / pv_ayto[i]) ** (1/n_yr_ayto) - 1) * 100
        pdf.set_fill_color(240 if i%2==0 else 255)
        for val, w in zip([
            clean_str(d),
            f"{int(pv_port[i]):,}", f"{int(v32_port):,}", f"{cagr_port:+.2f}%",
            f"{int(pv_ayto[i]):,}", f"{int(v32_ayto):,}", f"{cagr_ayto:+.2f}%",
        ], wdts_c):
            pdf.cell(w, 6, val, 1, 0, 'C', 1)
        pdf.ln()

    pdf.ln(4)
    # Tabla comparativa alquiler
    pdf.set_font('Arial','B',10)
    pdf.cell(0,7,clean_str("Proyeccion Alquiler EUR/m2/mes a 2032"),0,1)
    pa_port = DATASET_A['alquiler'][:,-1]
    pa_ayto = DATASET_B['alquiler'][:,-1]
    hdrs_ca = ["Distrito",
               f"Port. Base ({DATASET_A['ano_fin']})",
               "Port. Alq 2032",
               "Port. Yield'32",
               f"Ayto Base ({DATASET_B['ano_fin']})",
               "Ayto Alq 2032",
               "Ayto Yield'32"]
    wdts_ca = [42, 26, 24, 20, 26, 24, 20]
    _encabezado_tabla(pdf, hdrs_ca, wdts_ca)
    for i, d in enumerate(distritos):
        a32_port = np.median(sa_port[:, i, -1])
        a32_ayto = np.median(sa_ayto_r[:, i, -1])
        v32_port = np.median(sv_port[:, i, -1])
        v32_ayto = np.median(sv_ayto_r[:, i, -1])
        y_port   = (a32_port*12)/v32_port*100
        y_ayto   = (a32_ayto*12)/v32_ayto*100
        pdf.set_fill_color(240 if i%2==0 else 255)
        for val, w in zip([
            clean_str(d),
            f"{pa_port[i]:.1f}", f"{a32_port:.1f}", f"{y_port:.1f}%",
            f"{pa_ayto[i]:.1f}", f"{a32_ayto:.1f}", f"{y_ayto:.1f}%",
        ], wdts_ca):
            pdf.cell(w, 6, val, 1, 0, 'C', 1)
        pdf.ln()

    pdf.ln(4)
    # Comparativa de drift calibrado
    pdf.set_font('Arial','B',10)
    pdf.cell(0,7,clean_str("Comparativa de parametros calibrados por set"),0,1)
    hdrs_cp = ["Distrito",
               "Drift Portales %/a", "Beta Portales",
               "Drift Ayto %/a",     "Beta Ayto",
               "Diferencia Drift pp"]
    wdts_cp = [42, 30, 26, 30, 26, 32]
    _encabezado_tabla(pdf, hdrs_cp, wdts_cp)

    if 'Portal' in params['dataset_key']:
        dv_port = modelo_ppal['drift_venta']
        bt_port = modelo_ppal['betas']
        dv_ayto = modelo_comp['drift_venta']
        bt_ayto = modelo_comp['betas']
    else:
        dv_ayto = modelo_ppal['drift_venta']
        bt_ayto = modelo_ppal['betas']
        dv_port = modelo_comp['drift_venta']
        bt_port = modelo_comp['betas']

    for i, d in enumerate(distritos):
        diff = (dv_port[i] - dv_ayto[i]) * 100
        pdf.set_fill_color(240 if i%2==0 else 255)
        for val, w in zip([
            clean_str(d),
            f"{dv_port[i]*100:.2f}%", f"{bt_port[i]:.3f}",
            f"{dv_ayto[i]*100:.2f}%", f"{bt_ayto[i]:.3f}",
            f"{diff:+.2f} pp",
        ], wdts_cp):
            pdf.cell(w, 6, val, 1, 0, 'C', 1)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font('Arial','I',9)
    pdf.multi_cell(0,5,clean_str(
        "NOTA METODOLOGICA: Ambas simulaciones usan identico motor MS-VAR, "
        "identicas probabilidades de regimen y los mismos parametros macroeconomicos. "
        "Las diferencias en proyecciones se deben exclusivamente a: (1) precios "
        "base distintos en el ultimo dato disponible, (2) drift historico calibrado "
        "sobre periodos y fuentes diferentes, y (3) estructura de correlacion "
        "Cholesky estimada sobre distintas series temporales. "
        f"Portales: metodo {DATASET_A['nombre']}, {DATASET_A['anos_venta'][0]}-{DATASET_A['ano_fin']}. "
        f"Ayuntamiento: {DATASET_B['nombre']}, {DATASET_B['anos_venta'][0]}-{DATASET_B['ano_fin']}."
    ))

    return pdf.output(dest='S').encode('latin-1')


# ============================================================================
# 15. EXPLICACION METODOLOGICA DETALLADA
# ============================================================================

def generate_metodologia(params, n_sim, modelo) -> bytes:
    pdf = ProfessionalPDF()
    pdf.add_page()
    pdf.ln(6)
    pdf.set_font('Times','B',16)
    pdf.multi_cell(0,9,clean_str(
        "EXPLICACION METODOLOGICA DETALLADA DEL MODELO\n"
        "BARCELONA STRATEGIC MODEL v33.0"), align='C')
    pdf.ln(3)
    pdf.set_font('Times','I',11)
    pdf.multi_cell(0,6,clean_str(
        "MS-VAR: Markov Switching + AR(1)/VAR(1) Adaptativo + Cholesky + DCC\n"
        "Sistema de doble fuente de datos con calibracion independiente"),
        align='C')
    pdf.ln(5)
    pdf.set_font('Times','',9)
    ds = DATASETS[params['dataset_key']]
    for ln in [
        f"Version: Barcelona Strategic Model v33.0",
        f"Set de datos activo  : {params['dataset_key']}",
        f"Fuente               : {ds['fuente']}",
        f"Horizonte proyeccion : {ds['ano_fin']+1}-2032",
        f"Metodo estimacion    : {modelo['metodo_var']}",
        f"Generado             : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]:
        pdf.cell(0,5,clean_str(ln),0,1,'C')
    pdf.ln(4)
    pdf.line(25,pdf.get_y(),185,pdf.get_y())
    pdf.ln(3)
    _bloque_trazabilidad(pdf, params, "Todos los distritos", n_sim, modelo)

    # SECCION 1
    pdf.add_page()
    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("1. SISTEMA DE DOBLE FUENTE DE DATOS"),0,1)
    pdf.set_font('Times','',10)
    pdf.multi_cell(0,5.5,clean_str(
        "El modelo v33.0 incorpora dos fuentes de datos independientes que el usuario "
        "puede seleccionar desde el panel de control. Cada fuente se calibra de forma "
        "completamente independiente: betas CAPM, drift historico, matriz de "
        "correlacion (Cholesky) y coeficientes AR(1)/VAR(1) se estiman sobre los "
        "datos de la fuente seleccionada.\n\n"
        "SET A - DATOS PORTALES INMOBILIARIOS (Idealista / Incasol):\n"
        "Precios de oferta (asking price) por distrito, 2019-2026. Representan la "
        "banda alta del mercado: el precio al que los vendedores quieren vender, "
        "no necesariamente el precio de cierre. Son datos de alta frecuencia y "
        "granularidad distrital, actualizados a Q1 2026.\n\n"
        "SET B - DATOS OFICIALES AYUNTAMIENTO DE BARCELONA:\n"
        "Estadisticas oficiales del Portal de Dades del Ajuntament. Venta: 2012-2025 "
        "(14 anos). Alquiler: 2000-2025 (26 anos). Son precios de transaccion real "
        "(lo que efectivamente se paga), que historicamente estan por debajo de los "
        "precios de oferta. La mayor longitud de la serie permite activar el VAR(1) "
        "multivariado completo con regularizacion Ridge.\n\n"
        "LOGICA DE ESTIMACION ADAPTATIVA:\n"
        "El modelo detecta automaticamente el numero de retornos anuales disponibles "
        "en el set seleccionado. Si n >= 10 (suficiente para VAR(10x10) con n>p), "
        "se activa el VAR(1) Ridge. Si n < 10, se usa AR(1) univariado diagonal. "
        "Con el set del Ayuntamiento (n=13 retornos de venta), el VAR completo es "
        "estadisticamente viable. Con el set de Portales (n=7), se usa AR(1)."
    ))

    # SECCION 2 — reutilizar arquitectura v32
    pdf.add_page()
    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("2. ARQUITECTURA DEL MODELO Y ECUACION GOBERNANTE"),0,1)
    pdf.set_font('Times','',10)
    pdf.multi_cell(0,5.5,clean_str(
        "El motor de simulacion es identico para ambos sets de datos. "
        "Solo cambian los parametros calibrados (drift, A_var, L, vol). "
        "La ecuacion gobernante es:\n"
    ))
    pdf.set_font('Courier','B',9)
    pdf.multi_cell(0,5,clean_str(
        "dP_i(t)/P_i(t) = [mu_hist_i + lambda*A_var*r(t-1) + f(E,I,P)] * gamma_i * dt\n"
        "               + sigma_i(S_t) * SUM_j[L_ij(S_t)*dZ_j]\n"
        "               + rho * R_alq(t-1)  +  J_VT(t,i,ano_shock)"
    ))
    pdf.set_font('Times','',10)
    pdf.multi_cell(0,5.5,clean_str(
        "\nComponentes:\n"
        "mu_hist_i    : drift historico por distrito (media retornos observados)\n"
        "A_var        : AR(1) diagonal (n<10) o VAR(1) Ridge (n>=10)\n"
        "lambda=0.15  : ponderacion del componente autoregresivo\n"
        "f(E,I,P)     : ajuste diferencial Euribor/IPC/Paro al drift\n"
        "gamma_i      : elasticidad distrital al ciclo\n"
        "sigma_i(S_t) : volatilidad dinamica segun estado Markov\n"
        "L_ij(S_t)    : Cholesky modulada por estado (DCC simplificado)\n"
        "rho=0.45     : retroalimentacion alquiler->venta\n"
        "J_VT(t,i)    : shock regulatorio en ano_shock (2028 o 2029) si activado"
    ))

    # SECCION 3 — VAR Ridge
    pdf.add_page()
    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("3. ESTIMACION ADAPTATIVA: AR(1) vs VAR(1) RIDGE"),0,1)
    pdf.set_font('Times','',10)
    pdf.multi_cell(0,5.5,clean_str(
        "AR(1) UNIVARIADO DIAGONAL (set Portales, n=7):\n"
        "Con 7 retornos anuales y 10 variables, el VAR multivariado OLS es "
        "estadisticamente inviable (n<<p, radio espectral explosivo > 2). "
        "La solucion correcta es el AR(1) univariado:\n"
        "   rho_i = Cov(r_i(t), r_i(t-1)) / Var(r_i(t-1)), truncado a [-0.35, 0.35]\n"
        "A_var = diag(rho_1,...,rho_10). Radio espectral <= 0.35 garantizado.\n\n"
        "VAR(1) MULTIVARIADO CON REGULARIZACION RIDGE (set Ayuntamiento, n=13):\n"
        "Con 13 retornos y 10 variables (n>p), el VAR es estadisticamente viable. "
        "Para mayor estabilidad se usa regularizacion Ridge (L2) con lambda=0.5:\n"
        "   A_ridge = (X'X + lambda*I)^{-1} X'Y\n"
        "Esta penalizacion encoge los coeficientes hacia cero, reduciendo el "
        "sobreajuste y garantizando un radio espectral controlado. Si el radio "
        "espectral resultante supera 0.85, se reescala A_var = A_var * (0.85/rho).\n"
        "El VAR captura interdependencias entre distritos: si Eixample tuvo retorno "
        "elevado en t-1, esto puede predecir retornos en Gracia en t (efecto "
        "desbordamiento geografico documentado en Barcelona).\n\n"
        f"Metodo activo en esta ejecucion: {modelo['metodo_var']}\n"
        f"Radio espectral: {modelo['rho_espectral']:.4f}"
    ))

    # SECCION 4 — Shock ano
    pdf.add_page()
    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("4. SHOCK REGULATORIO VT: AÑO CONFIGURABLE"),0,1)
    pdf.set_font('Times','',10)
    shock_txt = (f"ACTIVADO en {params.get('shock_ano',2029)}"
                 if params.get('shock_vt') else "DESACTIVADO")
    pdf.multi_cell(0,5.5,clean_str(
        f"Estado del shock en este escenario: {shock_txt}\n\n"
        "El shock regulatorio modela el impacto de la restriccion de licencias "
        "de vivienda turistica (VT) sobre los precios de venta y alquiler. "
        "Se implementa como proceso de salto discreto (Jump Process) activado "
        "en el ano configurado por el usuario (2028 o 2029), reflejando la "
        "incertidumbre sobre el calendario de implementacion efectiva.\n\n"
        "Funcion de impacto:\n"
        "   J_vt(i) = -0.10 * (D_i / D_max) sobre precio de venta\n"
        "   J_alq(i) = -0.06 * (D_i / D_max) sobre precio de alquiler\n\n"
        "D_i = densidad VT distrital, D_max = 0.46 (Eixample).\n"
        "La eleccion de 2028 o 2029 permite modelizar escenarios de adelanto o "
        "retraso regulatorio sin modificar la magnitud del impacto."
    ))

    # SECCION 5 — Tests
    pdf.add_page()
    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("5. TESTS DE INTEGRIDAD MATEMATICA"),0,1)
    pdf.set_font('Courier','',8)
    test_res = run_tests(modelo, params['dataset_key'])
    for tname, tpass, tmsg in test_res:
        pdf.cell(0,5,clean_str(f"[{'PASS' if tpass else 'FAIL'}]  {tname}"),0,1)
        pdf.cell(0,4,clean_str(f"        {tmsg}"),0,1)

    # SECCION 6 — Referencias
    pdf.add_page()
    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("6. REFERENCIAS"),0,1)
    pdf.set_font('Times','',9)
    refs = [
        "Hamilton, J.D. (1989). A new approach to nonstationary time series. Econometrica.",
        "Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.",
        "Engle, R.F. (2002). Dynamic Conditional Correlation. JBES, 20(3), 339-350.",
        "Sims, C.A. (1980). Macroeconomics and Reality. Econometrica, 48(1), 1-48.",
        "Hoerl, A.E. & Kennard, R.W. (1970). Ridge Regression. Technometrics, 12(1), 55-67.",
        "Sharpe, W.F. (1964). Capital asset prices. Journal of Finance, 19(3), 425-442.",
        "Ajuntament de Barcelona (2025). Portal de Dades Obertes. portaldades.ajuntament.barcelona.cat",
        "Idealista Research (2026). Informe de Precios Q1 2026. Madrid: Idealista.",
        "Incasol - Institut Catala del Sol (2026). Estadistiques sector immobiliari.",
        "INE (2026). Indice de Precios de Vivienda (IPV). Madrid: INE.",
        "Ley 12/2023, de 24 de mayo, por el derecho a la vivienda. BOE num. 124.",
    ]
    for r in refs:
        pdf.multi_cell(0,5,clean_str(f"- {r}"))
        pdf.ln(1)

    return pdf.output(dest='S').encode('latin-1')


# ============================================================================
# 16. BOTONES EXPORTACION
# ============================================================================
st.markdown("---")
st.subheader("Exportar documentacion")

fig_pdf = st.session_state.fig_distrito
idx_pdf = st.session_state.idx_distrito

if fig_pdf is None:
    fig_pdf, (_a1, _a2) = plt.subplots(1, 2, figsize=(13, 5))
    _idx = idx_pdf
    _p50 = np.percentile(sv_act[:, _idx, :], 50, axis=0)
    _last = ds_activo['venta'][_idx, -1]
    _aplot = np.concatenate([[ds_activo['ano_fin']], anos_proy_act])
    _a1.plot(anos_macro, modelo_act['reconstruidos'][_idx], 'o-', color='#2c3e50')
    _a1.plot(_aplot, np.concatenate([[_last], _p50]), '--', color='#e74c3c')
    _a1.set_title(f"Venta - {distritos[_idx]}"); _a1.grid(True, linestyle=':', alpha=0.5)
    _p50a = np.percentile(sa_act[:, _idx, :], 50, axis=0)
    _lasta = ds_activo['alquiler'][_idx, -1]
    _a2.plot(ds_activo['anos_alq'], ds_activo['alquiler'][_idx], 'o-', color='#2c3e50')
    _a2.plot(_aplot, np.concatenate([[_lasta], _p50a]), '--', color='#27ae60')
    _a2.set_title(f"Alquiler - {distritos[_idx]}"); _a2.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    st.session_state.fig_distrito = fig_pdf

col_e1, col_e2 = st.columns(2)

with col_e1:
    if st.button("Generar Informe Ejecutivo (con comparativa)"):
        with st.spinner("Generando informe ejecutivo con comparativa..."):
            pdf_exec = generate_exec_report(
                idx_pdf, params_act, fig_pdf, n_sim_ui,
                modelo_act, sv_act, sa_act, anos_proy_act,
                modelo_alt, sv_alt, sa_alt, anos_proy_alt,
            )
        fname = (f"Informe_Ejecutivo_{distritos[idx_pdf].replace(' ','_')}_v33.pdf")
        st.download_button(f"Descargar {fname}", pdf_exec, fname, "application/pdf")

with col_e2:
    if st.button("Generar Explicacion Metodologica Detallada"):
        with st.spinner("Generando documento metodologico..."):
            pdf_met = generate_metodologia(params_act, n_sim_ui, modelo_act)
        st.download_button(
            "Descargar Explicacion Metodologica Detallada v33.0.pdf",
            pdf_met,
            "BCN_Model_v33_Explicacion_Metodologica_Detallada.pdf",
            "application/pdf"
        )

st.markdown("---")
st.caption(
    f"Barcelona Strategic Model v33.0 | {ds_key} | "
    f"Motor: {modelo_act['metodo_var']} | Rho espectral: {modelo_act['rho_espectral']:.4f} | "
    f"Regimen: {reg_sel} | Euribor: {euribor_val}% | IPC: {ipc_val}% | "
    f"Paro: {paro_val}% | Shock: {'ON (' + str(shock_ano) + ')' if shock_vt else 'OFF'} | "
    f"n_sim: {n_sim_ui}"
)


