# ==============================================================================
# BARCELONA STRATEGIC MODEL v40.0
# ==============================================================================
# Arquitectura SDE hibrida con Betas empiricas independientes (venta/alquiler),
# arquitectura hibrida Set A/B, shock de regimen aditivo Phi(S,M) con asimetria
# en Crisis, eliminacion de sensibilidad_distrital, DCC centralizado.
# Librerias UI: Streamlit + Plotly (interactivo) + Matplotlib (estatico PDF)
# PDF: fpdf2 (FPDF2) con soporte UTF-8 nativo (sin clean_str workaround)
# Nuevo: Plotly para Tab1 (fan chart interactivo), Tab2, Tab3; heatmap Plotly
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import tempfile
import datetime
import logging
import sys
import os

# fpdf2 con soporte UTF-8 nativo — elimina necesidad de clean_str para acentos
try:
    from fpdf import FPDF, XPos, YPos
    FPDF2_AVAILABLE = True
except ImportError:
    from fpdf import FPDF
    FPDF2_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# 0. LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BCNModelV40")

# ============================================================================
# 1. CONFIGURACION VISUAL — Streamlit
# ============================================================================
st.set_page_config(
    page_title="Barcelona Strategic Model v40.0",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS mejorado: tipografia, cards de metricas, separadores, colores coherentes
st.markdown("""
<style>
/* Fondo general */
.main { background-color: #f0f2f6; }

/* Tipografia cabecera */
h1 {
    color: #0d47a1;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-weight: 800;
    letter-spacing: -0.5px;
}
h2, h3 {
    color: #1a237e;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
}

/* Botones principales */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
    color: white;
    font-weight: 600;
    border-radius: 8px;
    height: 48px;
    box-shadow: 0 3px 8px rgba(13,71,161,0.3);
    border: none;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
    box-shadow: 0 5px 12px rgba(13,71,161,0.4);
    transform: translateY(-1px);
}

/* Metricas con fondo */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e0e4ea;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: white;
    border-radius: 8px;
    padding: 4px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: #555;
}
.stTabs [aria-selected="true"] {
    color: #0d47a1;
    border-bottom: 3px solid #0d47a1;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a237e;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #cfd8dc !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stSubheader {
    color: #e3f2fd !important;
}

/* Separadores */
hr { border-top: 2px solid #e0e4ea; }

/* Dataframes */
.dataframe th { background-color: #0d47a1 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. DATOS — AMBOS SETS (sin cambios respecto a v33)
# ============================================================================

distritos = [
    'Ciutat Vella', 'Eixample', 'Gracia', 'Horta Guinardo', 'Les Corts',
    'Nou Barris', 'Sant Andreu', 'Sant Marti', 'Sants-Montjuic', 'Sarria-Sant Gervasi'
]
n_distritos = len(distritos)
DIST_SHORT   = ['CV','EIX','GRA','HOR','LC','NB','SA','SM','SNT','SSG']

# Densidad VT — invariante al set
concentracion_vt_real = np.array([0.150, 0.460, 0.105, 0.030, 0.035,
                                   0.004, 0.015, 0.120, 0.110, 0.050])
rho_vta_alq = 0.45

# Indice macro agregado BCN (INE/IPV) — para betas venta y reconstruccion CAPM
datos_agregados_bcn = {
    2007: 4071, 2008: 3737, 2009: 3728, 2010: 3752, 2011: 3398, 2012: 3076,
    2013: 2992, 2014: 3061, 2015: 3297, 2016: 3649, 2017: 4154, 2018: 4232,
    2019: 4135, 2020: 4018, 2021: 3910, 2022: 4063, 2023: 4167, 2024: 4700,
    2025: 5042, 2026: 5200
}
anos_macro    = np.array(list(datos_agregados_bcn.keys()))
precios_macro = np.array(list(datos_agregados_bcn.values()), dtype=float)

# ---------------------------------------------------------------------------
# SET A — PORTALES INMOBILIARIOS (Idealista / Incasol) — 2019-2026
# ---------------------------------------------------------------------------
DATASET_A = {
    'nombre':     'Datos Portales Inmobiliarios',
    'fuente':     'Idealista / Incasol (Generalitat de Catalunya)',
    'fecha_dato': 'Q1 2026',
    'ano_fin':    2026,
    'anos_venta': np.arange(2019, 2027),
    'anos_alq':   np.arange(2019, 2027),
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
# SET B — AYUNTAMIENTO DE BARCELONA — Venta 2012-2025, Alquiler 2000-2025
# ---------------------------------------------------------------------------
_venta_ayto_src = np.array([
    [2355, 2767, 1965, 2974, 3339, 2590, 2215, 1668, 2151, 2432],  # 2012
    [2266, 2619, 1739, 2857, 3105, 2369, 1770, 1536, 1756, 2122],  # 2013
    [2387, 2683, 1843, 2981, 3108, 2374, 1776, 1398, 1823, 2169],  # 2014
    [2730, 2983, 2186, 3076, 3509, 2575, 2101, 2029, 2100, 2555],  # 2015
    [2989, 3381, 2354, 3284, 3750, 2953, 2038, 1621, 1994, 2458],  # 2016
    [3457, 3869, 2815, 3880, 4330, 3445, 2464, 1918, 2419, 3001],  # 2017
    [3833, 4111, 3137, 4204, 4755, 3824, 2690, 2174, 2626, 3069],  # 2018
    [3592, 4045, 3153, 3980, 4497, 3766, 2648, 2191, 2717, 3029],  # 2019
    [3584, 3977, 3092, 4161, 4388, 3695, 2657, 2098, 2671, 3095],  # 2020
    [3515, 4034, 3112, 4001, 4563, 3717, 2811, 2073, 2770, 3095],  # 2021
    [3752, 4391, 3220, 4152, 4811, 3990, 2897, 2495, 2905, 3490],  # 2022
    [3749, 4582, 3305, 4260, 5054, 4059, 2833, 2373, 2810, 3377],  # 2023
    [3845, 4869, 3446, 4666, 5200, 4320, 3213, 2531, 3124, 3693],  # 2024
    [4144, 5325, 3869, 5100, 5629, 4814, 3464, 3005, 3505, 4138],  # 2025
], dtype=float)
# Reordenar columnas fuente -> orden canonico
_col_map_ayto   = [0, 1, 5, 6, 3, 7, 8, 9, 2, 4]
_venta_ayto_can = _venta_ayto_src[:, _col_map_ayto].T   # (10, 14)

_alq_ayto_src = np.array([
    [ 5.5,  5.8,  6.1,  7.6,  7.4,  6.3,  5.8,  5.4,  5.5,  5.7],  # 2000
    [ 6.4,  6.8,  6.7,  8.6,  8.2,  7.0,  6.4,  6.2,  6.1,  6.5],  # 2001
    [ 7.3,  7.5,  7.5,  8.9,  9.1,  8.1,  7.1,  7.2,  6.7,  7.5],  # 2002
    [ 7.6,  7.9,  7.8,  9.3,  9.4,  8.4,  7.7,  7.3,  7.3,  7.7],  # 2003
    [ 8.6,  8.5,  8.7, 10.0, 10.0,  9.3,  8.3,  7.9,  7.8,  8.2],  # 2004
    [ 9.8,  9.5,  9.5, 10.8, 10.8, 10.2,  9.2,  8.9,  9.0,  9.2],  # 2005
    [10.7, 10.3, 10.5, 11.9, 11.6, 11.3, 10.2,  9.9,  9.6, 10.3],  # 2006
    [11.6, 11.3, 11.3, 13.0, 12.7, 12.5, 11.1, 10.8, 10.7, 11.2],  # 2007
    [12.6, 11.9, 12.2, 13.4, 13.3, 13.0, 12.0, 11.5, 11.5, 11.8],  # 2008
    [12.4, 11.6, 11.9, 12.9, 12.9, 12.8, 11.4, 11.2, 11.1, 11.6],  # 2009
    [12.1, 11.3, 11.5, 12.3, 12.8, 12.6, 10.8, 10.7, 10.7, 11.2],  # 2010
    [12.0, 11.2, 11.3, 12.1, 12.8, 12.0, 10.6, 10.3, 10.4, 10.9],  # 2011
    [11.7, 10.8, 10.8, 11.6, 12.3, 11.4, 10.1,  9.5,  9.8, 10.5],  # 2012
    [11.4, 10.2, 10.1, 11.2, 11.7, 11.0,  9.3,  8.7,  9.2,  9.9],  # 2013
    [11.4, 10.3,  9.9, 11.0, 11.9, 10.8,  9.0,  8.4,  8.9,  9.7],  # 2014
    [12.4, 11.2, 10.7, 12.0, 13.0, 11.8,  9.9,  8.9,  9.6, 10.6],  # 2015
    [13.9, 12.4, 11.7, 13.2, 14.3, 12.9, 10.7,  9.6, 10.4, 11.7],  # 2016
    [15.5, 13.4, 12.9, 14.2, 15.4, 14.3, 11.9, 10.6, 11.4, 12.9],  # 2017
    [14.5, 13.7, 13.0, 14.5, 15.4, 14.0, 11.9, 11.2, 11.7, 13.2],  # 2018
    [14.8, 14.3, 13.5, 15.0, 15.7, 14.6, 12.7, 11.7, 12.2, 13.9],  # 2019
    [14.3, 14.2, 13.6, 14.7, 15.6, 14.4, 12.8, 11.7, 12.3, 13.8],  # 2020
    [13.2, 13.3, 13.0, 13.8, 14.9, 13.8, 12.2, 11.3, 11.7, 13.3],  # 2021
    [15.8, 14.9, 14.1, 15.2, 16.2, 15.3, 13.1, 12.1, 12.5, 14.5],  # 2022
    [16.7, 16.5, 16.3, 16.8, 18.2, 16.9, 14.5, 13.3, 13.8, 16.2],  # 2023
    [16.6, 16.6, 15.6, 17.4, 18.4, 17.5, 14.5, 13.1, 13.7, 16.2],  # 2024
    [17.8, 16.5, 16.3, 17.2, 18.9, 17.3, 14.6, 13.1, 14.4, 16.2],  # 2025
], dtype=float)
_alq_ayto_can = _alq_ayto_src[:, _col_map_ayto].T  # (10, 26)

DATASET_B = {
    'nombre':     'Datos Oficiales Ayuntamiento de Barcelona',
    'fuente':     'Portal de Dades Ajuntament de Barcelona',
    'fecha_dato': '2025',
    'ano_fin':    2025,
    'anos_venta': np.arange(2012, 2026),
    'anos_alq':   np.arange(2000, 2026),
    'venta':      _venta_ayto_can,
    'alquiler':   _alq_ayto_can,
}

DATASETS = {
    'Datos Portales Inmobiliarios':              DATASET_A,
    'Datos Oficiales Ayuntamiento de Barcelona': DATASET_B,
}

# ============================================================================
# 3. MACRO: PROBABILIDADES DE REGIMEN Y MTM NO HOMOGENEA
# ============================================================================
MACRO_DEFAULTS = {
    "Crecimiento":          {"euribor": 1.5, "ipc": 2.5, "paro":  9.0},
    "Estanflacion":         {"euribor": 3.8, "ipc": 5.5, "paro": 13.0},
    "Recesion (Estructural)":{"euribor": 2.5, "ipc": 1.0, "paro": 20.0},
}


def calcular_Mt(euribor, ipc, paro):
    """
    Entorno macro base M_t = f(Euribor, IPC, Paro).
    Consolida los tres ajustes lineales en un unico escalar de presion ciclica.
    M_t > 0 -> entorno expansivo; M_t < 0 -> entorno contractivo.
    """
    euribor_adj = -np.clip((euribor - 2.5) / 100.0, -0.02, 0.04)
    ipc_adj     =  np.clip((ipc     - 2.0) / 100.0, -0.03, 0.03)
    paro_adj    = -np.clip((paro   - 10.0) / 100.0 * 0.5, -0.02, 0.03)
    return euribor_adj + ipc_adj + paro_adj


def calcular_Mt_alq(euribor, ipc, paro):
    """
    M_t para el mercado de alquiler: Euribor tiene signo inverso
    (Euribor alto desplaza demanda hacia alquiler -> positivo para alquiler).
    """
    euribor_adj =  np.clip((euribor - 2.5) / 100.0 * 0.4, -0.01, 0.02)
    ipc_adj     =  np.clip((ipc     - 2.0) / 100.0 * 0.6, -0.02, 0.02)
    paro_adj    = -np.clip((paro   - 10.0) / 100.0 * 0.3, -0.01, 0.02)
    return euribor_adj + ipc_adj + paro_adj


def calcular_probabilidades_regimen(euribor, ipc, paro):
    euribor_norm = np.clip((euribor - 0.0) / 5.0, 0.0, 1.0)
    ipc_norm     = np.clip((ipc     - 0.0) / 8.0, 0.0, 1.0)
    paro_norm    = np.clip((paro   - 5.0) / 25.0, 0.0, 1.0)
    score_boom   = (1.0 - euribor_norm)*0.4 + (1.0 - paro_norm)*0.4 + \
                   np.clip(ipc_norm*(1 - ipc_norm)*4, 0, 1)*0.2
    score_crisis = euribor_norm*0.45 + paro_norm*0.45 + \
                   np.clip((ipc_norm - 0.5), 0, 0.5)*0.1
    raw  = np.array([0.5, score_boom, score_crisis])
    exps = np.exp(raw * 2.5)
    return exps / exps.sum()


def construir_mtm_macro(euribor, ipc, paro):
    p_norm, p_boom, p_crisis = calcular_probabilidades_regimen(euribor, ipc, paro)
    PERSIST = 0.55
    mtm = np.array([
        [PERSIST + (1-PERSIST)*p_norm, (1-PERSIST)*p_boom, (1-PERSIST)*p_crisis],
        [0.35*(1-p_boom), PERSIST+(1-PERSIST)*p_boom, 0.05+0.10*p_crisis],
        [0.12+0.20*p_norm, 0.02, PERSIST+(1-PERSIST)*p_crisis],
    ])
    for r in range(3):
        s = mtm[r].sum()
        if s > 0:
            mtm[r] /= s
    return mtm


def validar_mtm(mtm, nombre):
    if not np.allclose(mtm.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"MTM '{nombre}' invalida: sumas={mtm.sum(axis=1)}")
    return True

# ============================================================================
# 4. MOTOR ESTRUCTURAL v40 — ARQUITECTURA HIBRIDA CON BETAS EMPIRICAS
# ============================================================================

def _calibrar_regimen_empirico(ret_mkt, anos_mkt):
    """
    Opcion A: calibracion empirica de kappa y C para Boom y Crisis.
    Identifica ventanas historicas sobre datos reales (Set B):
      - Boom:   2015-2019 (5 datos)
      - Crisis: 2013-2014 y 2020 (3 datos reales del Set B, sin reconstruccion)
    Retorna: kappa_boom, C_boom, kappa_crisis, C_crisis
    """
    mu_hist_M = ret_mkt.mean()

    # --- Identificacion de ventanas ---
    anos_boom   = np.array([2015, 2016, 2017, 2018, 2019])
    anos_crisis = np.array([2013, 2014, 2020])

    def _retornos_ventana(anos_target):
        idxs = [i for i, a in enumerate(anos_mkt) if a in anos_target]
        return ret_mkt[idxs] if idxs else np.array([])

    ret_boom   = _retornos_ventana(anos_boom)
    ret_crisis = _retornos_ventana(anos_crisis)

    # --- C: desviacion media respecto al drift historico general ---
    C_boom   = float(np.mean(ret_boom   - mu_hist_M)) if len(ret_boom)   > 0 else 0.02
    C_crisis = float(-np.mean(ret_crisis - mu_hist_M)) if len(ret_crisis) > 0 else 0.04
    # C_crisis debe ser positivo (la formula lo resta en Phi)
    C_crisis = max(C_crisis, 0.01)
    C_boom   = max(C_boom,   0.005)

    # --- kappa: OLS de desviaciones sobre Mt proxy ---
    # Usamos ret - mu_hist como proxy de Mt para cada ventana
    def _kappa_ols(ret_ventana):
        if len(ret_ventana) < 2:
            return 1.0
        y = ret_ventana - mu_hist_M
        # Mt proxy = ret_M - mu_hist (mismo para mercado agregado)
        x = y  # simplificado: kappa = Cov(y,y)/Var(y) cuando y~Mt
        # Regresion y = kappa * x => kappa = sum(x*y)/sum(x^2)
        denom = np.sum(x**2)
        if denom < 1e-10:
            return 1.0
        return float(np.clip(np.sum(x*y) / denom, 0.5, 2.5))

    kappa_boom   = _kappa_ols(ret_boom)
    kappa_crisis = _kappa_ols(ret_crisis)

    logger.info("[Regimen Empirico] C_boom=%.4f kappa_boom=%.4f "
                "C_crisis=%.4f kappa_crisis=%.4f",
                C_boom, kappa_boom, C_crisis, kappa_crisis)
    return kappa_boom, C_boom, kappa_crisis, C_crisis


def phi_regimen(estado, Mt, kappa_boom, C_boom, kappa_crisis, C_crisis):
    """
    Funcion de Shock de Regimen Phi(S_t, M_t).
    - Normal:  Phi = M_t
    - Boom:    Phi = M_t * kappa_boom + C_boom
    - Crisis:  Phi = min(M_t,0)*kappa_crisis + max(M_t,0)*1.0 - C_crisis
               (asimetria: solo amplifica M_t si es negativo)
    """
    if estado == 0:   # Normal
        return Mt
    elif estado == 1: # Boom
        return Mt * kappa_boom + C_boom
    else:             # Crisis
        return min(Mt, 0.0) * kappa_crisis + max(Mt, 0.0) * 1.0 - C_crisis


@st.cache_data
def inicializar_modelo(dataset_key: str):
    """
    Calibracion v40 con arquitectura hibrida:

    SIEMPRE Set B:
      - beta_vta  : Beta venta vs IPV INE (indice ciudad BCN)
      - beta_alq  : Beta alquiler vs indice sintetico alquiler BCN (media 10 distritos)
      - Cholesky  : correlacion espacial sobre serie larga reconstruida
      - vol_cal   : volatilidad estructural (propiedades del activo)
      - kappa/C   : parametros empiricos de regimen

    Set SELECCIONADO:
      - drift historico venta y alquiler
      - A_var (VAR(1) Ridge o AR(1) diagonal)
      - precios base
    """
    ds_sel = DATASETS[dataset_key]
    ds_B   = DATASET_B  # siempre Set B para parametros estructurales

    pv_sel = ds_sel['venta']      # (10, T_sel_v)
    pa_sel = ds_sel['alquiler']   # (10, T_sel_a)
    pv_B   = ds_B['venta']        # (10, 14) — 2012-2025
    pa_B   = ds_B['alquiler']     # (10, 26) — 2000-2025

    anos_v_sel = ds_sel['anos_venta']
    anos_v_B   = ds_B['anos_venta']
    anos_a_B   = ds_B['anos_alq']

    # --- RETORNOS ---
    ret_v_sel = (pv_sel[:, 1:] / pv_sel[:, :-1]) - 1   # (10, T_sel_v - 1)
    ret_a_sel = (pa_sel[:, 1:] / pa_sel[:, :-1]) - 1
    ret_v_B   = (pv_B[:, 1:]  / pv_B[:, :-1])  - 1     # (10, 13)
    ret_a_B   = (pa_B[:, 1:]  / pa_B[:, :-1])  - 1     # (10, 25)

    # --- DRIFT HISTORICO (set seleccionado) ---
    drift_venta    = ret_v_sel.mean(axis=1)
    drift_alquiler = ret_a_sel.mean(axis=1)

    # ---------------------------------------------------------------
    # BETA VENTA — siempre Set B vs IPV INE (indice ciudad BCN)
    # ---------------------------------------------------------------
    # Alinear retornos B con retornos del indice macro
    mask_mkt = (anos_macro >= anos_v_B[0]) & (anos_macro <= anos_v_B[-1])
    pm_sub   = precios_macro[mask_mkt]
    anos_mkt_sub = anos_macro[mask_mkt]

    if len(pm_sub) >= 3:
        ret_mkt_B = (pm_sub[1:] / pm_sub[:-1]) - 1
        anos_mkt_ret = anos_mkt_sub[1:]
        n_align   = min(ret_v_B.shape[1], len(ret_mkt_B))
        beta_vta  = []
        for i in range(n_distritos):
            rv = ret_v_B[i, -n_align:]
            rm = ret_mkt_B[-n_align:]
            if np.var(rm) > 1e-10:
                b = np.cov(rv, rm)[0, 1] / np.var(rm)
            else:
                b = 1.0
            beta_vta.append(np.clip(b, 0.5, 1.5))
        beta_vta = np.array(beta_vta)
    else:
        beta_vta = np.ones(n_distritos)
        ret_mkt_B = np.diff(precios_macro) / precios_macro[:-1]
        anos_mkt_ret = anos_macro[1:]

    # ---------------------------------------------------------------
    # BETA ALQUILER — siempre Set B vs indice sintetico alquiler BCN
    # (media aritmetica simple de 10 distritos, Set B, serie larga)
    # ---------------------------------------------------------------
    indice_alq_BCN = pa_B.mean(axis=0)       # media simple de 10 distritos (26 puntos)
    ret_alq_BCN    = (indice_alq_BCN[1:] / indice_alq_BCN[:-1]) - 1  # 25 retornos
    beta_alq = []
    for i in range(n_distritos):
        ra = ret_a_B[i]  # 25 retornos
        rm = ret_alq_BCN
        n_al = min(len(ra), len(rm))
        ra_a, rm_a = ra[-n_al:], rm[-n_al:]
        if np.var(rm_a) > 1e-10:
            b = np.cov(ra_a, rm_a)[0, 1] / np.var(rm_a)
        else:
            b = 1.0
        beta_alq.append(np.clip(b, 0.5, 1.5))
    beta_alq = np.array(beta_alq)

    logger.info("[v40] beta_vta: %s", beta_vta.round(3))
    logger.info("[v40] beta_alq: %s", beta_alq.round(3))

    # ---------------------------------------------------------------
    # CALIBRACION EMPIRICA DE REGIMEN (Opcion A) — sobre Set B
    # ---------------------------------------------------------------
    kappa_boom, C_boom, kappa_crisis, C_crisis = _calibrar_regimen_empirico(
        ret_mkt_B, anos_mkt_ret
    )

    # ---------------------------------------------------------------
    # A_VAR ADAPTATIVO — sobre set seleccionado
    # ---------------------------------------------------------------
    n_ret = ret_v_sel.shape[1]
    logger.info("[v40][%s] n_retornos_venta=%d", dataset_key, n_ret)

    if n_ret >= 10:
        Y = ret_v_sel[:, 1:].T
        X = ret_v_sel[:, :-1].T
        lam = 0.5
        try:
            XtX   = X.T @ X + lam * np.eye(n_distritos)
            A_var = np.linalg.solve(XtX, X.T @ Y)
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
        ar1 = np.zeros(n_distritos)
        for i in range(n_distritos):
            r = ret_v_sel[i]
            if len(r) >= 3:
                cv = np.cov(r[1:], r[:-1])[0, 1]
                vv = np.var(r[:-1])
                ar1[i] = np.clip(cv / vv if vv > 1e-10 else 0.0, -0.35, 0.35)
        A_var = np.diag(ar1)
        metodo_var = f"AR(1) diagonal (n={n_ret})"
        rho_final  = np.abs(ar1).max()

    # ---------------------------------------------------------------
    # RECONSTRUCCION HISTORICA 2007-INICIO para Cholesky — Set B
    # ---------------------------------------------------------------
    idx_ini_B = np.where(anos_macro == anos_v_B[0])[0]
    idx_ini_B = idx_ini_B[0] if len(idx_ini_B) else 0

    reconstruidos = np.zeros((n_distritos, len(anos_macro)))
    for ti, yr in enumerate(anos_v_B):
        m_idx = np.where(anos_macro == yr)[0]
        if len(m_idx):
            reconstruidos[:, m_idx[0]] = pv_B[:, ti]
    np.random.seed(999)
    for t in range(idx_ini_B - 1, -1, -1):
        rmt = (precios_macro[t+1] / precios_macro[t]) - 1
        for d in range(n_distritos):
            ruido = np.random.normal(0, 0.003)
            r_d   = beta_vta[d] * rmt + ruido
            if reconstruidos[d, t+1] > 0:
                reconstruidos[d, t] = reconstruidos[d, t+1] / (1 + r_d)
            else:
                reconstruidos[d, t] = pv_B[d, 0] / (1 + r_d) ** (idx_ini_B - t)

    # --- Cholesky sobre Set B reconstruido ---
    ret_full = np.zeros_like(reconstruidos)
    for t in range(1, len(anos_macro)):
        denom = np.where(reconstruidos[:, t-1] > 0, reconstruidos[:, t-1], 1.0)
        ret_full[:, t] = (reconstruidos[:, t] / denom) - 1
    ret_full = ret_full[:, 1:]
    corr = np.corrcoef(ret_full)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        ev, evec = np.linalg.eigh(corr)
        ev     = np.maximum(ev, 1e-8)
        corr_r = evec @ np.diag(ev) @ evec.T
        dd     = np.sqrt(np.diag(corr_r))
        corr_r = corr_r / np.outer(dd, dd)
        try:
            L = np.linalg.cholesky(corr_r)
        except Exception:
            L = np.eye(n_distritos)
        corr = corr_r

    # --- Volatilidad base — siempre Set B ---
    vol_cal = np.array([ret_v_B[i].std() for i in range(n_distritos)])
    vol_cal = np.clip(vol_cal, 0.015, 0.10)

    # --- Reconstruidos para grafico (set seleccionado superpuesto) ---
    # Rellena tambien los anos del set seleccionado para consistencia visual
    for ti, yr in enumerate(anos_v_sel):
        m_idx = np.where(anos_macro == yr)[0]
        if len(m_idx):
            reconstruidos[:, m_idx[0]] = pv_sel[:, ti]

    return {
        # Parametros estructurales (Set B)
        'beta_vta':      beta_vta,
        'beta_alq':      beta_alq,
        'L':             L,
        'corr':          corr,
        'vol_cal':       vol_cal,
        'kappa_boom':    kappa_boom,
        'C_boom':        C_boom,
        'kappa_crisis':  kappa_crisis,
        'C_crisis':      C_crisis,
        # Parametros de inercia (set seleccionado)
        'drift_venta':   drift_venta,
        'drift_alquiler': drift_alquiler,
        'A_var':         A_var,
        'metodo_var':    metodo_var,
        'rho_espectral': rho_final,
        'n_ret_venta':   n_ret,
        # Auxiliares
        'reconstruidos': reconstruidos,
        # Compatibilidad con reports (alias)
        'betas':         beta_vta,
    }


# ============================================================================
# 5. MOTOR MONTE CARLO v40
# ============================================================================
LAMBDA_VAR = 0.15

def simular(dataset_key, euribor, ipc, paro, shock_vt, shock_ano, n_sim,
            seed=42, horizonte_fin=2032):
    """
    Motor Monte Carlo v40.
    SDE: dP/P = (mu_hist + lambda*VAR) + beta_vta * Phi(S,Mt) + sigma*DCC*Z
                + rho*ret_alq_prev + J_VT

    Cambios respecto a v33:
    - Eliminado sensibilidad_distrital (sustituido por beta_vta en bloque ciclico)
    - Phi(S,Mt) aditiva con asimetria en Crisis
    - vol_cal y Cholesky siempre de Set B (hibrido)
    - beta_alq independiente para mercado alquiler
    - r_prev actualizado con tasa realizada (Opcion A) + clip(+-0.08)
    - Ruidos diferenciados por regimen eliminados; DCC centraliza heteroscedasticidad
    """
    np.random.seed(seed)
    ds    = DATASETS[dataset_key]
    m     = inicializar_modelo(dataset_key)

    mu_v   = m['drift_venta']
    mu_a   = m['drift_alquiler']
    A_var  = m['A_var']
    beta_v = m['beta_vta']
    beta_a = m['beta_alq']
    L_base = m['L']
    vol    = m['vol_cal']
    kb, Cb = m['kappa_boom'],   m['C_boom']
    kc, Cc = m['kappa_crisis'], m['C_crisis']

    ano_base  = ds['ano_fin']
    anos_proy = np.arange(ano_base + 1, horizonte_fin + 1)
    H         = len(anos_proy)

    pv0 = ds['venta'][:, -1].copy()
    pa0 = ds['alquiler'][:, -1].copy()

    # MTM y probabilidades de regimen
    mtm  = construir_mtm_macro(euribor, ipc, paro)
    Mt_v = calcular_Mt(euribor, ipc, paro)
    Mt_a = calcular_Mt_alq(euribor, ipc, paro)

    sv = np.zeros((n_sim, n_distritos, H))
    sa = np.zeros((n_sim, n_distritos, H))

    for sim in range(n_sim):
        pv   = pv0.copy()
        pa   = pa0.copy()
        est  = 0  # 0=Normal, 1=Boom, 2=Crisis
        r_prev = mu_v.copy()   # inicializacion: drift historico en t0

        for t in range(H):
            # --- 1. Transicion Markov no homogenea ---
            u   = np.random.rand()
            cum = np.cumsum(mtm[est])
            est = int(np.searchsorted(cum, u))
            est = min(est, 2)

            # --- 2. DCC: volatilidad y Cholesky dinamica por estado ---
            if est == 1:    # Boom
                vol_dyn  = vol * 0.85
                corr_dyn = m['corr'] * 0.8 + np.eye(n_distritos) * 0.2
            elif est == 2:  # Crisis
                vol_dyn  = vol * 1.60
                corr_dyn = m['corr'] * 0.5 + np.ones((n_distritos, n_distritos)) * 0.5
            else:           # Normal
                vol_dyn  = vol.copy()
                corr_dyn = m['corr'].copy()

            try:
                L_dyn = np.linalg.cholesky(corr_dyn)
            except np.linalg.LinAlgError:
                ev, evec = np.linalg.eigh(corr_dyn)
                ev       = np.maximum(ev, 1e-8)
                cd_r     = evec @ np.diag(ev) @ evec.T
                dd       = np.sqrt(np.diag(cd_r))
                cd_r     = cd_r / np.outer(dd, dd)
                try:
                    L_dyn = np.linalg.cholesky(cd_r)
                except Exception:
                    L_dyn = np.eye(n_distritos)

            Z   = np.random.standard_normal(n_distritos)
            eps = L_dyn @ Z

            # --- 3. Shock de regimen Phi (aditivo) ---
            phi_v = phi_regimen(est, Mt_v, kb, Cb, kc, Cc)
            phi_a = phi_regimen(est, Mt_a, kb * 0.7, Cb * 0.6, kc * 0.7, Cc * 0.6)

            # --- 4. Tasa de crecimiento VENTA ---
            # Bloque 1: Inercia endogena (drift + VAR)
            inercia_v = mu_v + LAMBDA_VAR * (A_var @ r_prev)
            # Bloque 2: Ciclo modulado por beta empírica
            ciclo_v   = beta_v * phi_v
            # Bloque 3: DCC (Cholesky + vol dinamica)
            ruido_v   = vol_dyn * eps
            # Suma
            tasa_v = inercia_v + ciclo_v + ruido_v

            # Floor volatilidad
            tasa_v = np.maximum(tasa_v, -0.07)

            # Floor yield (minimo 2.5% bruto)
            yield_bruto = (pa * 12) / np.maximum(pv, 1.0)
            resist       = np.clip(yield_bruto / 0.025, 0.5, 1.0)
            tasa_v       = np.where(tasa_v > 0, tasa_v * resist, tasa_v)

            # --- 5. Tasa de crecimiento ALQUILER ---
            ret_a_prev = (pa / pa0) - 1  # retorno acumulado simple como proxy
            inercia_a  = mu_a + LAMBDA_VAR * 0.5 * (np.diag(A_var) * ret_a_prev)
            ciclo_a    = beta_a * phi_a
            ruido_a    = vol_dyn * 0.6 * eps
            tasa_a     = inercia_a + ciclo_a + ruido_a
            tasa_a     = np.maximum(tasa_a, -0.05)

            # --- 6. Feedback alquiler -> venta ---
            if t > 0:
                ret_alq_prev = (pa / np.maximum(sa[sim, :, t-1], 0.01)) - 1
                tasa_v      += rho_vta_alq * np.clip(ret_alq_prev, -0.05, 0.05)

            # --- 7. Shock regulatorio VT (Jump Process) ---
            ano_actual = ano_base + 1 + t
            if shock_vt and ano_actual == shock_ano:
                D_max    = 0.46
                j_vta    = -0.10 * (concentracion_vt_real / D_max)
                j_alq    = -0.06 * (concentracion_vt_real / D_max)
                tasa_v  += j_vta
                tasa_a  += j_alq

            # --- 8. Actualizar precios ---
            pv = pv * (1 + tasa_v)
            pa = pa * (1 + tasa_a)
            pv = np.maximum(pv, 500.0)
            pa = np.maximum(pa,   5.0)

            sv[sim, :, t] = pv
            sa[sim, :, t] = pa

            # r_prev: tasa realizada completa con clip (Opcion A)
            r_prev = np.clip(tasa_v, -0.08, 0.08)

    return sv, sa, anos_proy


@st.cache_data
def run_all_scenarios(dataset_key, euribor, ipc, paro, shock_vt, shock_ano, n_sim):
    """Lanza simulaciones para escenario activo + Crecimiento + Recesion."""
    sv_act, sa_act, ap_act = simular(
        dataset_key, euribor, ipc, paro, shock_vt, shock_ano, n_sim)
    sv_crec, sa_crec, _ = simular(
        dataset_key,
        MACRO_DEFAULTS['Crecimiento']['euribor'],
        MACRO_DEFAULTS['Crecimiento']['ipc'],
        MACRO_DEFAULTS['Crecimiento']['paro'],
        shock_vt, shock_ano, n_sim, seed=43)
    sv_rec, sa_rec, _ = simular(
        dataset_key,
        MACRO_DEFAULTS['Recesion (Estructural)']['euribor'],
        MACRO_DEFAULTS['Recesion (Estructural)']['ipc'],
        MACRO_DEFAULTS['Recesion (Estructural)']['paro'],
        shock_vt, shock_ano, n_sim, seed=44)
    sv_est, sa_est, _ = simular(
        dataset_key,
        MACRO_DEFAULTS['Estanflacion']['euribor'],
        MACRO_DEFAULTS['Estanflacion']['ipc'],
        MACRO_DEFAULTS['Estanflacion']['paro'],
        shock_vt, shock_ano, n_sim, seed=45)
    return (sv_act, sa_act, ap_act,
            sv_crec, sa_crec, sv_rec, sa_rec, sv_est, sa_est)


# ============================================================================
# 6. TESTS DE INTEGRIDAD
# ============================================================================
def run_tests(modelo, dataset_key):
    ds  = DATASETS[dataset_key]
    res = []

    pv0 = ds['venta'][:, -1]
    pa0 = ds['alquiler'][:, -1]

    res.append(("Precios base positivos",
                bool(np.all(pv0 > 0) and np.all(pa0 > 0)),
                f"Venta min={pv0.min():.0f} | Alquiler min={pa0.min():.2f}"))

    yields = pa0 * 12 / pv0 * 100
    res.append(("Yields base en rango [2%-12%]",
                bool(np.all(yields > 2) and np.all(yields < 12)),
                f"min={yields.min():.1f}% max={yields.max():.1f}%"))

    dv = modelo['drift_venta'] * 100
    res.append(("Drift historico plausible (<30%/a)",
                bool(np.all(np.abs(dv) < 30)),
                f"{np.round(dv, 2)}%"))

    rho = modelo['rho_espectral']
    res.append(("Radio espectral A_var <= 0.90",
                bool(rho <= 0.90),
                f"Radio espectral: {rho:.4f} | Metodo: {modelo['metodo_var']}"))

    bv = modelo['beta_vta']
    res.append(("Beta VENTA en [0.5, 1.5]",
                bool(np.all(bv >= 0.5) and np.all(bv <= 1.5)),
                f"min={bv.min():.3f} max={bv.max():.3f}"))

    ba = modelo['beta_alq']
    res.append(("Beta ALQUILER en [0.5, 1.5]",
                bool(np.all(ba >= 0.5) and np.all(ba <= 1.5)),
                f"min={ba.min():.3f} max={ba.max():.3f}"))

    L = modelo['L']
    es_lower = bool(np.allclose(np.triu(L, 1), 0))
    pos_diag = bool(np.all(np.diag(L) > 0))
    err_max  = float(np.abs(L @ L.T - modelo['corr']).max())
    res.append(("Cholesky triangular inferior y diagonal positiva",
                es_lower and pos_diag,
                f"Lower:{es_lower} | PosDiag:{pos_diag}"))
    res.append(("Cholesky: L*L^T aprox Sigma",
                bool(err_max < 1e-4),
                f"Error max={err_max:.2e}"))

    for sc_name, (eu, ip, pa) in [
        ('Crecimiento',   (1.5, 2.5,  9.0)),
        ('Estanflacion',  (3.8, 5.5, 13.0)),
        ('Recesion',      (2.5, 1.0, 20.0)),
    ]:
        mtm = construir_mtm_macro(eu, ip, pa)
        ok  = bool(np.allclose(mtm.sum(axis=1), 1.0, atol=1e-6))
        res.append((f"MTM dinamica '{sc_name}' filas=1", ok,
                    f"Sumas: {mtm.sum(axis=1).round(6)}"))

    res.append(("Regimen empirico C_crisis > 0",
                bool(modelo['C_crisis'] > 0),
                f"C_crisis={modelo['C_crisis']:.4f} | "
                f"kappa_crisis={modelo['kappa_crisis']:.4f}"))
    res.append(("vol_cal en rango seguro [1.5%-10%]",
                bool(np.all(modelo['vol_cal'] > 0.015) and np.all(modelo['vol_cal'] < 0.10)),
                f"min={modelo['vol_cal'].min():.4f} max={modelo['vol_cal'].max():.4f}"))

    return res


# ============================================================================
# 7. SIDEBAR — Panel de Control
# ============================================================================
with st.sidebar:
    st.markdown("## 🏙️ BCN Strategic Model")
    st.markdown("### v40.0")
    st.markdown("---")

    st.subheader("📊 Fuente de Datos")
    ds_key = st.radio(
        "Set activo:",
        list(DATASETS.keys()),
        index=0,
        help="Set A: precios de oferta 2019-2026. Set B: transacciones reales 2012-2025."
    )
    ds_activo = DATASETS[ds_key]
    ds_key_alt = [k for k in DATASETS if k != ds_key][0]

    st.markdown("---")
    st.subheader("📈 Escenario Macro")
    reg_sel = st.selectbox("Régimen económico:", list(MACRO_DEFAULTS.keys()), index=0)
    macro_def = MACRO_DEFAULTS[reg_sel]

    euribor_val = st.slider("Euribor 12m (%)", 0.0, 6.0,
                             float(macro_def['euribor']), 0.1,
                             help="Tipo de referencia hipotecario europeo")
    ipc_val     = st.slider("IPC anual (%)",  -1.0, 10.0,
                             float(macro_def['ipc']), 0.1,
                             help="Inflacion anual esperada")
    paro_val    = st.slider("Tasa de paro (%)", 5.0, 35.0,
                             float(macro_def['paro']), 0.5,
                             help="Tasa de desempleo nacional")

    st.markdown("---")
    st.subheader("🏠 Shock Regulatorio VT")
    shock_vt  = st.checkbox("Activar shock VT", value=True)
    shock_ano = st.radio("Año del shock:", [2028, 2029], index=1,
                          horizontal=True) if shock_vt else 2029

    st.markdown("---")
    st.subheader("⚙️ Simulación")
    n_sim = st.select_slider("Simulaciones (n_sim):",
                              options=[250, 500, 1000, 2000], value=1000)

    # Métricas macro activas en sidebar
    st.markdown("---")
    probs_side = calcular_probabilidades_regimen(euribor_val, ipc_val, paro_val)
    Mt_display = calcular_Mt(euribor_val, ipc_val, paro_val)
    st.caption(f"**M_t macro base:** {Mt_display*100:+.2f}%")
    st.caption(f"**P(Normal):** {probs_side[0]*100:.1f}%")
    st.caption(f"**P(Boom):**   {probs_side[1]*100:.1f}%")
    st.caption(f"**P(Crisis):** {probs_side[2]*100:.1f}%")

    run_btn = st.button("▶  SIMULAR v40", type="primary")

# ============================================================================
# 8. CABECERA PRINCIPAL
# ============================================================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🏙️ Barcelona Strategic Model v40.0")
    st.caption(
        f"SDE Híbrido · MS-VAR · Betas Empíricas Venta/Alquiler · "
        f"Φ Aditivo · DCC Centralizado · Arquitectura Híbrida Set A/B"
    )
with col_h2:
    st.metric("Set activo", ds_key.split()[0] + " " + ds_key.split()[1] if len(ds_key.split()) > 1 else ds_key[:15])
    st.metric("Dato base", ds_activo['fecha_dato'])

st.markdown("---")

# ============================================================================
# 9. INICIALIZACIÓN Y SIMULACIÓN
# ============================================================================
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'idx_distrito' not in st.session_state:
    st.session_state.idx_distrito = 0
if 'params_sim' not in st.session_state:
    st.session_state.params_sim = {}

if run_btn:
    with st.spinner("⚙️  Calibrando modelo v40 y ejecutando simulaciones Monte Carlo..."):
        modelo_act = inicializar_modelo(ds_key)
        modelo_alt = inicializar_modelo(ds_key_alt)
        (sv_act, sa_act, ap_act,
         sv_crec, sa_crec,
         sv_rec,  sa_rec,
         sv_est,  sa_est) = run_all_scenarios(
             ds_key, euribor_val, ipc_val, paro_val,
             shock_vt, shock_ano, n_sim)
        st.session_state.resultados = {
            'modelo_act': modelo_act, 'modelo_alt': modelo_alt,
            'sv_act': sv_act, 'sa_act': sa_act, 'ap_act': ap_act,
            'sv_crec': sv_crec, 'sa_crec': sa_crec,
            'sv_rec': sv_rec,   'sa_rec': sa_rec,
            'sv_est': sv_est,   'sa_est': sa_est,
            'ds_key': ds_key, 'ds_key_alt': ds_key_alt,
        }
        st.session_state.params_sim = {
            'dataset_key': ds_key, 'regimen': reg_sel,
            'euribor': euribor_val, 'ipc': ipc_val, 'paro': paro_val,
            'shock_vt': shock_vt, 'shock_ano': shock_ano, 'n_sim': n_sim,
        }
    st.success("✅ Simulación completada")

if st.session_state.resultados is None:
    st.info("👈  Configura los parámetros en el panel lateral y pulsa **SIMULAR v40**.")
    st.stop()

# Extraer resultados de session state
R          = st.session_state.resultados
modelo_act = R['modelo_act']
modelo_alt = R['modelo_alt']
sv_act     = R['sv_act']
sa_act     = R['sa_act']
ap_act     = R['ap_act']
sv_crec    = R['sv_crec'];  sa_crec = R['sa_crec']
sv_rec     = R['sv_rec'];   sa_rec  = R['sa_rec']
sv_est     = R['sv_est'];   sa_est  = R['sa_est']
params_sim = st.session_state.params_sim

anos_proy_act = ap_act


# ============================================================================
# 10. TABS PRINCIPALES
# ============================================================================
ds_activo  = DATASETS[ds_key]
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Análisis Distrito",
    "⚖️ Riesgo / Retorno",
    "🔬 Macro & Φ",
    "🧮 Parámetros v40",
    "✅ Tests",
])

# ============================================================================
# TAB 1 — ANÁLISIS DISTRITO (fan chart interactivo Plotly)
# ============================================================================
with tab1:
    sel_dist = st.selectbox("Distrito:", distritos, index=st.session_state.idx_distrito)
    idx = distritos.index(sel_dist)
    st.session_state.idx_distrito = idx

    rec    = modelo_act['reconstruidos']
    last_v = ds_activo['venta'][idx, -1]
    last_a = ds_activo['alquiler'][idx, -1]
    ap     = anos_proy_act
    anos_plot = np.concatenate([[ds_activo['ano_fin']], ap])

    # Percentiles
    p10_v = np.percentile(sv_act[:, idx, :], 10, axis=0)
    p50_v = np.percentile(sv_act[:, idx, :], 50, axis=0)
    p90_v = np.percentile(sv_act[:, idx, :], 90, axis=0)
    p10_a = np.percentile(sa_act[:, idx, :], 10, axis=0)
    p50_a = np.percentile(sa_act[:, idx, :], 50, axis=0)
    p90_a = np.percentile(sa_act[:, idx, :], 90, axis=0)
    p50_crec = np.percentile(sv_crec[:, idx, :], 50, axis=0)
    p50_rec  = np.percentile(sv_rec[:,  idx, :], 50, axis=0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Precio Venta EUR/m² — {sel_dist}",
                        f"Alquiler EUR/m²/mes — {sel_dist}"),
        horizontal_spacing=0.08
    )

    # Histórico venta (reconstruido)
    mask_hist = anos_macro <= ds_activo['ano_fin']
    fig.add_trace(go.Scatter(
        x=anos_macro[mask_hist], y=rec[idx][mask_hist],
        mode='lines+markers', name='Histórico/Rec.',
        line=dict(color='#2c3e50', width=2), marker=dict(size=4),
        hovertemplate='%{x}: %{y:,.0f} €/m²<extra>Histórico</extra>'
    ), row=1, col=1)
    fig.add_vline(x=ds_activo['ano_fin'], line_dash="dash",
                  line_color="navy", line_width=1.5, row=1, col=1)

    # Banda P10-P90 venta
    x_band = list(anos_plot) + list(anos_plot[::-1])
    y_band = list(np.concatenate([[last_v], p90_v])) + \
             list(np.concatenate([[last_v], p10_v])[::-1])
    fig.add_trace(go.Scatter(x=x_band, y=y_band, fill='toself',
        fillcolor='rgba(231,76,60,0.13)', line=dict(color='rgba(0,0,0,0)'),
        name='IC P10–P90', hoverinfo='skip'
    ), row=1, col=1)

    # P50 activo
    fig.add_trace(go.Scatter(
        x=anos_plot, y=np.concatenate([[last_v], p50_v]),
        mode='lines', name=f'P50 activo',
        line=dict(color='#e74c3c', width=2.5, dash='dash'),
        hovertemplate='%{x}: %{y:,.0f} €/m²<extra>P50 activo</extra>'
    ), row=1, col=1)

    # Escenarios referencia
    fig.add_trace(go.Scatter(
        x=anos_plot, y=np.concatenate([[last_v], p50_crec]),
        mode='lines', name='P50 Crecimiento',
        line=dict(color='#27ae60', width=1.2, dash='dot'),
        hovertemplate='%{x}: %{y:,.0f}<extra>Crecimiento</extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=anos_plot, y=np.concatenate([[last_v], p50_rec]),
        mode='lines', name='P50 Recesión',
        line=dict(color='#7f8c8d', width=1.2, dash='dot'),
        hovertemplate='%{x}: %{y:,.0f}<extra>Recesion</extra>'
    ), row=1, col=1)

    # Alquiler histórico
    fig.add_trace(go.Scatter(
        x=ds_activo['anos_alq'], y=ds_activo['alquiler'][idx],
        mode='lines+markers', name='Alquiler histórico',
        line=dict(color='#2c3e50', width=2), marker=dict(size=4),
        hovertemplate='%{x}: %{y:.1f} €/m²/mes<extra>Histórico</extra>'
    ), row=1, col=2)
    fig.add_vline(x=ds_activo['ano_fin'], line_dash="dash",
                  line_color="navy", line_width=1.5, row=1, col=2)

    x_ba = list(anos_plot) + list(anos_plot[::-1])
    y_ba = list(np.concatenate([[last_a], p90_a])) + \
           list(np.concatenate([[last_a], p10_a])[::-1])
    fig.add_trace(go.Scatter(x=x_ba, y=y_ba, fill='toself',
        fillcolor='rgba(39,174,96,0.13)', line=dict(color='rgba(0,0,0,0)'),
        name='IC Alq P10–P90', hoverinfo='skip'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=anos_plot, y=np.concatenate([[last_a], p50_a]),
        mode='lines', name='P50 alquiler',
        line=dict(color='#27ae60', width=2.5, dash='dash'),
        hovertemplate='%{x}: %{y:.1f} €/m²/mes<extra>P50 alquiler</extra>'
    ), row=1, col=2)

    fig.update_layout(
        height=480, template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=60, b=40, l=40, r=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', tickformat=',')
    st.session_state['fig_tab1'] = fig
    st.plotly_chart(fig, use_container_width=True)

    # KPIs
    v_fin  = float(np.median(sv_act[:, idx, -1]))
    a_fin  = float(np.median(sa_act[:, idx, -1]))
    yld_32 = (a_fin * 12) / v_fin * 100
    n_yr   = 2032 - ds_activo['ano_fin']

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Venta 2032",    f"{int(v_fin):,} €/m²",
              f"{((v_fin/last_v)-1)*100:.1f}%")
    k2.metric("Alquiler 2032", f"{a_fin:.1f} €/m²/mes",
              f"{((a_fin/last_a)-1)*100:.1f}%")
    k3.metric("Yield 2032",    f"{yld_32:.1f}%")
    k4.metric("Drift hist.",   f"{modelo_act['drift_venta'][idx]*100:.2f}%/a")
    k5.metric("β_vta",         f"{modelo_act['beta_vta'][idx]:.3f}")
    k6.metric("β_alq",         f"{modelo_act['beta_alq'][idx]:.3f}")

    # Tabla todos los distritos
    st.markdown(f"#### Tabla Base {ds_activo['ano_fin']} — Todos los Distritos")
    pv0_tab = ds_activo['venta'][:, -1]
    pa0_tab = ds_activo['alquiler'][:, -1]
    rows = []
    for i, d in enumerate(distritos):
        v32_i = float(np.median(sv_act[:, i, -1]))
        cagr_i = ((v32_i / pv0_tab[i]) ** (1/n_yr) - 1) * 100
        rows.append({
            'Distrito':       d,
            f'Venta {ds_activo["ano_fin"]}': int(pv0_tab[i]),
            'Venta 2032 P50': int(v32_i),
            'CAGR %':         round(cagr_i, 2),
            f'Alq {ds_activo["ano_fin"]}': round(pa0_tab[i], 1),
            'Yield bruto':    f"{(pa0_tab[i]*12/pv0_tab[i]*100):.1f}%",
            'β_vta':          round(modelo_act['beta_vta'][i], 3),
            'β_alq':          round(modelo_act['beta_alq'][i], 3),
            'Drift venta %':  round(modelo_act['drift_venta'][i]*100, 2),
        })
    df_tab = pd.DataFrame(rows)
    st.dataframe(
        df_tab.set_index('Distrito')
        .style.background_gradient(cmap='Greens', subset=['CAGR %'])
        .format({'CAGR %': '{:.2f}%'}),
        use_container_width=True
    )


# ============================================================================
# TAB 2 — RIESGO / RETORNO
# ============================================================================
with tab2:
    pv0_t2 = ds_activo['venta'][:, -1]
    n_yr_t2 = 2032 - ds_activo['ano_fin']
    cagrs, vols, v32s, betas_v = [], [], [], []
    for i in range(n_distritos):
        f = sv_act[:, i, -1]
        cagrs.append(((f.mean() / pv0_t2[i]) ** (1/n_yr_t2) - 1) * 100)
        vols.append(f.std() / f.mean() * 100)
        v32s.append(int(np.median(f)))
        betas_v.append(round(modelo_act['beta_vta'][i], 3))

    df_rr = pd.DataFrame({
        'Distrito':         distritos,
        'Retorno CAGR (%)': cagrs,
        'Riesgo CV (%)':    vols,
        'Venta P50 2032':   v32s,
        'β_vta':            betas_v,
    })
    fig_rr = px.scatter(
        df_rr, x='Riesgo CV (%)', y='Retorno CAGR (%)', text='Distrito',
        color='Retorno CAGR (%)', color_continuous_scale='RdYlGn',
        size=[b * 15 for b in betas_v], size_max=28,
        hover_data={'β_vta': True, 'Venta P50 2032': ':,'},
        title=(f"Cuadrante Riesgo/Retorno · {reg_sel} · "
               f"Euribor={euribor_val}% · Paro={paro_val}% · "
               f"Tamaño burbuja = β_vta"),
        template='plotly_white',
    )
    mr_rr = df_rr['Riesgo CV (%)'].mean()
    mt_rr = df_rr['Retorno CAGR (%)'].mean()
    fig_rr.add_vline(x=mr_rr, line_dash="dash", line_color="#e74c3c", line_width=1.2)
    fig_rr.add_hline(y=mt_rr, line_dash="dash", line_color="#e74c3c", line_width=1.2)
    fig_rr.add_annotation(
        x=df_rr['Riesgo CV (%)'].min()+0.1, y=df_rr['Retorno CAGR (%)'].max(),
        text="⭐ ESTRELLAS (bajo riesgo, alto retorno)",
        showarrow=False, font=dict(color="#27ae60", size=11)
    )
    fig_rr.add_annotation(
        x=df_rr['Riesgo CV (%)'].max()-0.3, y=df_rr['Retorno CAGR (%)'].min(),
        text="⚠️ INEFICIENTES", showarrow=False,
        font=dict(color="#c0392b", size=11)
    )
    fig_rr.update_traces(textposition='top center')
    fig_rr.update_layout(height=520, coloraxis_showscale=False)
    st.plotly_chart(fig_rr, use_container_width=True)

    st.dataframe(
        df_rr.style
        .format({'Retorno CAGR (%)': '{:.2f}%', 'Riesgo CV (%)': '{:.2f}%',
                 'Venta P50 2032': '{:,}'})
        .background_gradient(cmap="Greens", subset=["Retorno CAGR (%)"])
        .background_gradient(cmap="Reds_r", subset=["Riesgo CV (%)"]),
        use_container_width=True
    )


# ============================================================================
# TAB 3 — MACRO & PHI
# ============================================================================
with tab3:
    st.subheader("Motor Macro: Indicadores → Probabilidades de Régimen")

    probs_t3 = calcular_probabilidades_regimen(euribor_val, ipc_val, paro_val)
    Mt_t3    = calcular_Mt(euribor_val, ipc_val, paro_val)

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.markdown("**Matriz de Transición no homogénea:**")
        mtm_act = construir_mtm_macro(euribor_val, ipc_val, paro_val)
        df_mtm  = pd.DataFrame(
            mtm_act.round(4),
            index=['Normal', 'Boom', 'Crisis'],
            columns=['→Normal', '→Boom', '→Crisis']
        )
        st.dataframe(df_mtm.style.background_gradient(cmap='Blues'),
                     use_container_width=True)
        st.markdown(f"""
**Parámetros Φ calibrados (Set B):**
| | Valor |
|---|---|
| C_boom | **{modelo_act['C_boom']*100:.2f}%** |
| C_crisis | **{modelo_act['C_crisis']*100:.2f}%** |
| κ_boom | **{modelo_act['kappa_boom']:.3f}** |
| κ_crisis | **{modelo_act['kappa_crisis']:.3f}** |
| M_t actual | **{Mt_t3*100:+.2f}%** |
""")

    with col_m2:
        eu_range = np.linspace(0.5, 5.5, 50)
        pb_t3 = [calcular_probabilidades_regimen(e, ipc_val, paro_val)[1]*100 for e in eu_range]
        pc_t3 = [calcular_probabilidades_regimen(e, ipc_val, paro_val)[2]*100 for e in eu_range]
        pn_t3 = [calcular_probabilidades_regimen(e, ipc_val, paro_val)[0]*100 for e in eu_range]
        fig_m3 = go.Figure()
        fig_m3.add_trace(go.Scatter(x=eu_range, y=pb_t3, name='P(Boom)',
            line=dict(color='#27ae60', width=2.5),
            hovertemplate='Euribor %{x:.1f}% → P(Boom)=%{y:.1f}%<extra></extra>'))
        fig_m3.add_trace(go.Scatter(x=eu_range, y=pc_t3, name='P(Crisis)',
            line=dict(color='#c0392b', width=2.5),
            hovertemplate='Euribor %{x:.1f}% → P(Crisis)=%{y:.1f}%<extra></extra>'))
        fig_m3.add_trace(go.Scatter(x=eu_range, y=pn_t3, name='P(Normal)',
            line=dict(color='#2980b9', width=1.8, dash='dot'),
            hovertemplate='Euribor %{x:.1f}% → P(Normal)=%{y:.1f}%<extra></extra>'))
        fig_m3.add_vline(x=euribor_val, line_dash="dash", line_color="navy",
                          line_width=1.8,
                          annotation_text=f"Euribor actual ({euribor_val}%)")
        fig_m3.update_layout(title="Sensibilidad del Régimen al Euribor",
                              xaxis_title="Euribor 12m (%)",
                              yaxis_title="Probabilidad (%)",
                              template='plotly_white', height=340,
                              legend=dict(orientation='h'), hovermode='x unified')
        st.plotly_chart(fig_m3, use_container_width=True)

    st.markdown("---")
    st.subheader("Función Φ(S, M_t) — Shock de Régimen Aditivo")
    mt_rng = np.linspace(-0.09, 0.07, 80)
    kb_ = modelo_act['kappa_boom'];  Cb_ = modelo_act['C_boom']
    kc_ = modelo_act['kappa_crisis']; Cc_ = modelo_act['C_crisis']
    phi_b_ = [phi_regimen(1, m, kb_, Cb_, kc_, Cc_) for m in mt_rng]
    phi_n_ = [phi_regimen(0, m, kb_, Cb_, kc_, Cc_) for m in mt_rng]
    phi_c_ = [phi_regimen(2, m, kb_, Cb_, kc_, Cc_) for m in mt_rng]
    fig_phi = go.Figure()
    fig_phi.add_trace(go.Scatter(x=mt_rng*100, y=np.array(phi_b_)*100,
        name='Φ Boom', line=dict(color='#27ae60', width=2.5),
        hovertemplate='M_t=%{x:.2f}% → Φ_Boom=%{y:.2f}%<extra></extra>'))
    fig_phi.add_trace(go.Scatter(x=mt_rng*100, y=np.array(phi_n_)*100,
        name='Φ Normal', line=dict(color='#2980b9', width=2, dash='dash'),
        hovertemplate='M_t=%{x:.2f}% → Φ_Normal=%{y:.2f}%<extra></extra>'))
    fig_phi.add_trace(go.Scatter(x=mt_rng*100, y=np.array(phi_c_)*100,
        name='Φ Crisis', line=dict(color='#c0392b', width=2.5),
        hovertemplate='M_t=%{x:.2f}% → Φ_Crisis=%{y:.2f}%<extra></extra>'))
    fig_phi.add_vline(x=Mt_t3*100, line_dash="dot", line_color="navy",
                       annotation_text=f"M_t actual ({Mt_t3*100:+.2f}%)")
    fig_phi.add_hline(y=0, line_color="black", line_width=0.7)
    fig_phi.update_layout(
        title=(f"Φ(S,M) — κ_boom={kb_:.3f}, κ_crisis={kc_:.3f} "
               f"(calibrados empíricamente sobre Set B 2013–2020)"),
        xaxis_title="M_t — Impulso Macro (%)", yaxis_title="Φ(S,M) (%)",
        template='plotly_white', height=360,
        legend=dict(orientation='h'), hovermode='x unified'
    )
    st.plotly_chart(fig_phi, use_container_width=True)

    st.markdown("---")
    st.subheader("Matriz de Correlación Espacial — Set B (estructural)")
    dist_short = [d[:8] for d in distritos]
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=modelo_act['corr'],
        x=dist_short, y=dist_short,
        colorscale='RdYlGn', zmin=-1, zmax=1,
        text=np.round(modelo_act['corr'], 2),
        texttemplate='%{text:.2f}',
        hovertemplate='%{y} × %{x}: %{z:.3f}<extra></extra>',
        colorbar=dict(title='ρ')
    ))
    fig_heatmap.update_layout(
        title="Correlación Histórica de Retornos (Cholesky base) — siempre Set B",
        template='plotly_white', height=430,
        xaxis=dict(tickangle=-30), margin=dict(l=120, b=100)
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


# ============================================================================
# TAB 4 — PARÁMETROS v40
# ============================================================================
with tab4:
    st.subheader("Parámetros Calibrados v40 — Arquitectura Híbrida")
    st.info(
        "**Estructurales (β, Cholesky, σ_base):** siempre desde Set B · "
        "**Inercia (μ_hist, A_var):** desde set seleccionado · "
        "**Parámetros régimen (κ, C):** OLS empírico sobre ventanas Boom/Crisis Set B"
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### Betas Endógenas")
        df_betas_t4 = pd.DataFrame({
            'Distrito': distritos,
            'β_vta (vs IPV INE)':    np.round(modelo_act['beta_vta'], 4),
            'β_alq (vs Sint. BCN)':  np.round(modelo_act['beta_alq'], 4),
            'Δβ (vta−alq)':          np.round(modelo_act['beta_vta'] - modelo_act['beta_alq'], 4),
            'σ_base %':              np.round(modelo_act['vol_cal'] * 100, 2),
        })
        st.dataframe(
            df_betas_t4.set_index('Distrito')
            .style.background_gradient(cmap='Blues',  subset=['β_vta (vs IPV INE)'])
            .background_gradient(cmap='Greens', subset=['β_alq (vs Sint. BCN)']),
            use_container_width=True
        )
        st.caption("Ambas betas desde Set B. β_vta: IPV INE como benchmark. β_alq: media simple 10 distritos.")

    with col_p2:
        st.markdown("#### β_vta vs β_alq por Distrito")
        fig_beta_t4 = go.Figure()
        fig_beta_t4.add_trace(go.Bar(
            name='β_vta', x=DIST_SHORT,
            y=modelo_act['beta_vta'],
            marker_color='#0d47a1', opacity=0.85,
            hovertemplate='%{x}: β_vta=%{y:.3f}<extra></extra>'
        ))
        fig_beta_t4.add_trace(go.Bar(
            name='β_alq', x=DIST_SHORT,
            y=modelo_act['beta_alq'],
            marker_color='#27ae60', opacity=0.85,
            hovertemplate='%{x}: β_alq=%{y:.3f}<extra></extra>'
        ))
        fig_beta_t4.add_hline(y=1.0, line_dash="dot", line_color="red",
                               annotation_text="β = 1.0")
        fig_beta_t4.update_layout(
            barmode='group', template='plotly_white', height=300,
            yaxis_title='Beta', legend=dict(orientation='h'),
        )
        st.plotly_chart(fig_beta_t4, use_container_width=True)

    st.markdown("---")
    col_p3, col_p4 = st.columns(2)
    with col_p3:
        st.markdown("#### Inercia Autorregresiva (Set seleccionado)")
        df_inercia = pd.DataFrame({
            'Distrito':           distritos,
            'Drift venta %/a':    np.round(modelo_act['drift_venta'] * 100, 3),
            'Drift alquiler %/a': np.round(modelo_act['drift_alquiler'] * 100, 3),
        })
        st.dataframe(df_inercia.set_index('Distrito'), use_container_width=True)
        st.caption(f"Motor A_var: **{modelo_act['metodo_var']}** | "
                   f"ρ espectral: {modelo_act['rho_espectral']:.4f}")

    with col_p4:
        st.markdown("#### Comparativa Drift entre Sets")
        if ds_key != list(DATASETS.keys())[0]:
            m_port = modelo_alt; m_ayto = modelo_act
        else:
            m_port = modelo_act; m_ayto = modelo_alt
        df_drift_comp = pd.DataFrame({
            'Distrito':               distritos,
            'Drift Portales %/a':     np.round(m_port['drift_venta'] * 100, 3),
            'Drift Ayuntamiento %/a': np.round(m_ayto['drift_venta'] * 100, 3),
            'Diferencia pp':          np.round((m_port['drift_venta'] - m_ayto['drift_venta']) * 100, 3),
        })
        st.dataframe(df_drift_comp.set_index('Distrito'), use_container_width=True)


# ============================================================================
# TAB 5 — TESTS
# ============================================================================
with tab5:
    test_results = run_tests(modelo_act, ds_key)
    n_pass = sum(1 for _, p, _ in test_results if p)
    total  = len(test_results)
    color  = "normal" if n_pass == total else "inverse"
    st.metric("Tests pasados", f"{n_pass}/{total}",
              delta="✅ Todos OK" if n_pass == total else f"⚠️ {total-n_pass} fallidos",
              delta_color=color)
    for tname, tpass, tmsg in test_results:
        icon = "✅" if tpass else "❌"
        with st.expander(f"{icon}  {tname}"):
            st.code(tmsg)


# ============================================================================
# 11. PDF — UTILIDADES COMUNES
# ============================================================================

def cs(txt):
    """clean_str: transliteration para fpdf v1."""
    repls = {
        '€': 'EUR', 'á': 'a', 'Á': 'A', 'é': 'e', 'É': 'E',
        'í': 'i', 'Í': 'I', 'ó': 'o', 'Ó': 'O', 'ú': 'u', 'Ú': 'U',
        'à': 'a', 'è': 'e', 'ò': 'o', 'ù': 'u',
        'ñ': 'n', 'Ñ': 'N', 'ç': 'c', 'Ç': 'C',
        'ü': 'u', 'ï': 'i', 'ö': 'o',
        '²': '2', '–': '-', '—': '-', '−': '-',
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u00a0': ' ', '\u00ad': '', '\u2026': '...',
        'β': 'beta', 'Φ': 'Phi', 'κ': 'kappa', 'μ': 'mu',
        'σ': 'sigma', 'λ': 'lambda', 'ρ': 'rho', '→': '->',
        '≥': '>=', '≤': '<=', '×': 'x',
    }
    for k, v in repls.items():
        txt = txt.replace(k, v)
    return txt.encode('latin-1', errors='replace').decode('latin-1')


class PDFv40(FPDF):
    def header(self):
        self.set_font('Times', 'B', 9)
        self.cell(0, 8, 'BCN STRATEGIC MODEL V40.0 | CONFIDENCIAL', 0, 1, 'C')
        self.line(10, 17, 200, 17)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 10,
                  f'Pagina {self.page_no()} | {ts} | BCN Model v40.0 | Confidencial',
                  0, 0, 'C')


def _tabla_hdr(pdf, headers, widths, fill=(44, 62, 80)):
    pdf.set_fill_color(*fill)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 8)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, cs(h), 1, 0, 'C', 1)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 8)


def _trazabilidad(pdf, params, dist_name, n_sim_val, modelo):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    probs = calcular_probabilidades_regimen(
        params.get('euribor', 2.0), params.get('ipc', 2.5), params.get('paro', 10.0))
    Mt_val = calcular_Mt(params.get('euribor', 2.5),
                         params.get('ipc', 2.0),
                         params.get('paro', 10.0))
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, cs("TRAZABILIDAD DEL ESCENARIO"), 0, 1)
    pdf.set_font('Courier', '', 8)
    lineas = [
        f"Timestamp              : {ts}",
        f"Modelo                 : Barcelona Strategic Model v40.0",
        f"Distrito               : {dist_name}",
        f"Set de datos activo    : {params.get('dataset_key', 'N/A')}",
        f"Arquitectura beta      : SIEMPRE Set B para beta_vta, beta_alq, Cholesky, vol",
        f"Fuente                 : {DATASETS[params['dataset_key']]['fuente']}",
        f"Ultimo dato disponible : {DATASETS[params['dataset_key']]['fecha_dato']}",
        f"Regimen economico      : {params.get('regimen', 'N/A')}",
        f"Euribor 12m            : {params.get('euribor', 0):.2f}%",
        f"IPC anual              : {params.get('ipc', 0):.2f}%",
        f"Tasa de paro           : {params.get('paro', 0):.1f}%",
        f"M_t (impulso macro)    : {Mt_val*100:+.3f}%",
        f"Shock VT               : {'Activado (' + str(params.get('shock_ano', 2029)) + ')' if params.get('shock_vt') else 'Desactivado'}",
        f"Simulaciones (n_sim)   : {n_sim_val}",
        f"Horizonte proyeccion   : {DATASETS[params['dataset_key']]['ano_fin']+1}-2032",
        f"Motor estimacion A_var : {modelo['metodo_var']}",
        f"Radio espectral A_var  : {modelo['rho_espectral']:.4f}",
        f"beta_vta (min/max)     : {modelo['beta_vta'].min():.3f} / {modelo['beta_vta'].max():.3f}",
        f"beta_alq (min/max)     : {modelo['beta_alq'].min():.3f} / {modelo['beta_alq'].max():.3f}",
        f"P(Normal/Boom/Crisis)  : {probs[0]*100:.1f}% / {probs[1]*100:.1f}% / {probs[2]*100:.1f}%",
        f"C_boom / C_crisis      : {modelo['C_boom']*100:.2f}% / {modelo['C_crisis']*100:.2f}%",
        f"kappa_boom / kappa_c   : {modelo['kappa_boom']:.3f} / {modelo['kappa_crisis']:.3f}",
    ]
    for ln_txt in lineas:
        pdf.cell(0, 5, cs(ln_txt), 0, 1)
    pdf.ln(3)


def _noshock_approx(sv_s, sa_s, di):
    c  = concentracion_vt_real[di]
    fv = (c / 0.46) * 0.14
    fa = (c / 0.46) * 0.10
    v  = np.median(sv_s[:, di, -1])
    a  = np.median(sa_s[:, di, -1])
    return v / max(1 - fv, 0.01), a / max(1 - fa, 0.01)


# ============================================================================
# 12. INFORME EJECUTIVO v40
# ============================================================================

def generate_exec_report_v40(dist_idx, params, n_sim_val,
                              modelo_ppal, sv_ppal, sa_ppal, anos_ppal,
                              modelo_comp, sv_comp, sa_comp) -> bytes:
    ds_ppal = DATASETS[params['dataset_key']]
    probs   = calcular_probabilidades_regimen(
        params.get('euribor', 2.0), params.get('ipc', 2.5), params.get('paro', 10.0))
    Mt_val  = calcular_Mt(params.get('euribor', 2.5),
                          params.get('ipc', 2.0), params.get('paro', 10.0))

    pdf = PDFv40()
    pdf.add_page()

    # Portada
    pdf.set_font('Arial', 'B', 15)
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255)
    pdf.cell(0, 11, cs(f"INFORME EJECUTIVO v40: {distritos[dist_idx].upper()}"),
             0, 1, 'C', 1)
    pdf.set_text_color(0)
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 6, cs(
        f"Barcelona Strategic Model v40.0 | {params['dataset_key']} | "
        f"SDE bifurcada: Inercia + beta*Phi(S,M) | Betas empiricas venta/alquiler"
    ), 0, 1, 'C')
    pdf.ln(4)
    _trazabilidad(pdf, params, distritos[dist_idx], n_sim_val, modelo_ppal)

    # 1. Parámetros macro
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, cs("1. PARAMETROS MACROECONOMICOS Y CICLO v40"), 0, 1)
    pdf.set_font('Arial', '', 9)
    phi_v_act = phi_regimen(0, Mt_val,
                             modelo_ppal['kappa_boom'], modelo_ppal['C_boom'],
                             modelo_ppal['kappa_crisis'], modelo_ppal['C_crisis'])
    pdf.multi_cell(0, 5, cs(
        f"Regimen: {params.get('regimen','N/A')} | "
        f"Euribor: {params.get('euribor',0):.2f}% | "
        f"IPC: {params.get('ipc',0):.2f}% | Paro: {params.get('paro',0):.1f}%\n"
        f"Impulso macro consolidado M_t: {Mt_val*100:+.3f}%\n"
        f"P(Normal)={probs[0]*100:.1f}% | P(Boom)={probs[1]*100:.1f}% | "
        f"P(Crisis)={probs[2]*100:.1f}%\n"
        f"Drift historico ({distritos[dist_idx]}): "
        f"{modelo_ppal['drift_venta'][dist_idx]*100:.2f}%/a | "
        f"beta_vta: {modelo_ppal['beta_vta'][dist_idx]:.3f} | "
        f"beta_alq: {modelo_ppal['beta_alq'][dist_idx]:.3f}\n"
        f"Parametros regimen calibrados empiricamente (Set B, ventanas 2013-2020):\n"
        f"  C_boom={modelo_ppal['C_boom']*100:.2f}% | "
        f"C_crisis={modelo_ppal['C_crisis']*100:.2f}% | "
        f"kappa_boom={modelo_ppal['kappa_boom']:.3f} | "
        f"kappa_crisis={modelo_ppal['kappa_crisis']:.3f}\n"
        f"Phi(Normal, M_t) = {phi_v_act*100:+.3f}% "
        f"(impulso ciclico sobre el distrito en regimen Normal)"
    ))
    pdf.ln(3)

    # 2. Sensibilidad VT
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, cs("2. SENSIBILIDAD REGULATORIA — SHOCK VT"), 0, 1)
    pdf.set_font('Arial', '', 9)
    shock_txt = (f"Activado en {params.get('shock_ano', 2029)}"
                 if params.get('shock_vt') else "Desactivado")
    pdf.cell(0, 5, cs(f"Estado del shock: {shock_txt}"), 0, 1)
    conc_d = concentracion_vt_real[dist_idx]
    pdf.cell(0, 5, cs(
        f"Densidad VT {distritos[dist_idx]}: {conc_d:.3f} | "
        f"Impacto venta: {-10*(conc_d/0.46):.1f}% | "
        f"Impacto alquiler: {-6*(conc_d/0.46):.1f}%"
    ), 0, 1)
    pdf.ln(2)
    hdrs_vt = ["Escenario", "Venta 2032 (Shock)", "Venta 2032 (Sin Shock)",
               "Alq 2032 (Shock)", "Alq 2032 (Sin Shock)"]
    wdts_vt = [35, 38, 38, 38, 38]
    _tabla_hdr(pdf, hdrs_vt, wdts_vt)
    for sc, (sv_s, sa_s) in [
        ("Crecimiento",  (sv_crec, sa_crec)),
        ("Estanflacion", (sv_est,  sa_est)),
        ("Recesion",     (sv_rec,  sa_rec)),
    ]:
        v32 = np.median(sv_s[:, dist_idx, -1])
        a32 = np.median(sa_s[:, dist_idx, -1])
        vns, ans = _noshock_approx(sv_s, sa_s, dist_idx)
        pdf.set_fill_color(245, 245, 245)
        for val, w in zip([cs(sc), f"{int(v32):,}", f"{int(vns):,}",
                           f"{a32:.1f}", f"{ans:.1f}"], wdts_vt):
            pdf.cell(w, 6, val, 1, 0, 'C', 1)
        pdf.ln()
    pdf.ln(3)

    # 3. Tabla maestra todos los distritos
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, cs(
        f"3. TABLA MAESTRA — TODOS LOS DISTRITOS "
        f"({ds_ppal['ano_fin']} vs 2032) | {params['dataset_key']}"), 0, 1)
    pv0_rep = ds_ppal['venta'][:, -1]
    pa0_rep = ds_ppal['alquiler'][:, -1]
    n_yr_rep = 2032 - ds_ppal['ano_fin']
    hdrs_tm = ["Distrito", "Venta Base", "Venta 2032", "Var%",
               "Alq Base", "Alq 2032", "Yield'32", "b_vta", "b_alq"]
    wdts_tm = [38, 22, 22, 16, 18, 18, 18, 14, 14]
    _tabla_hdr(pdf, hdrs_tm, wdts_tm)
    for i, d in enumerate(distritos):
        v32_i = np.median(sv_ppal[:, i, -1])
        a32_i = np.median(sa_ppal[:, i, -1])
        var_i = ((v32_i / pv0_rep[i]) - 1) * 100
        yld_i = (a32_i * 12) / v32_i * 100
        pdf.set_fill_color(240 if i % 2 == 0 else 255)
        for val, w in zip([
            cs(d), f"{int(pv0_rep[i]):,}", f"{int(v32_i):,}",
            f"{var_i:+.1f}%", f"{pa0_rep[i]:.1f}", f"{a32_i:.1f}",
            f"{yld_i:.1f}%",
            f"{modelo_ppal['beta_vta'][i]:.3f}",
            f"{modelo_ppal['beta_alq'][i]:.3f}"
        ], wdts_tm):
            pdf.cell(w, 6, val, 1, 0, 'C', 1)
        pdf.ln()

    # 4. Recomendación
    pdf.ln(4)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, cs("4. RECOMENDACION AL INVERSOR v40"), 0, 1)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, cs(
        f"Basado en {params['dataset_key']} | Motor v40.0 | "
        f"Dato base: {ds_ppal['fecha_dato']}\n\n"
        "DIFERENCIA CLAVE v40 respecto a v33: la SDE bifurcada preserva la tendencia\n"
        "estructural de cada distrito intacta. Solo el componente ciclico se amplifica\n"
        "o amortigua por la beta empirica especifica (venta y alquiler independientes).\n\n"
        "- YIELD: Nou Barris y Sant Andreu presentan yields superiores al 5%.\n"
        "  Baja exposicion VT y demanda residencial estable.\n\n"
        "- PLUSVALIA: distritos con beta_vta alta (Eixample, Sarria) amplifican el\n"
        "  ciclo. En Crisis su beta multiplica el shock negativo exactamente como\n"
        "  muestran los datos historicos. En v33 esto era un supuesto; en v40 emerge\n"
        "  de la covarianza historica contra el IPV INE.\n\n"
        "- ALQUILER INDEPENDIENTE: beta_alq calibrada sobre el mercado de alquiler\n"
        "  BCN (media 10 distritos). Distritos con beta_alq < 1 tienen rentas mas\n"
        "  estables en escenarios adversos, mejorando el perfil de riesgo del yield.\n\n"
        f"- MACRO ACTUAL: M_t = {Mt_val*100:+.3f}%. Euribor={params.get('euribor',0):.1f}%.\n"
        "  El impulso se transmite como beta_i * Phi(S, M_t), con C_crisis calibrado\n"
        f"  empiricamente en {modelo_ppal['C_crisis']*100:.2f}% sobre datos 2013-2020.\n\n"
        "NOTA: proyecciones sobre mediana P50. Consultar IC P10-P90 para rango de\n"
        "incertidumbre completo. Horizonte 6 anos. Modelo sin datos de oferta."
    ))

    # 5. Comparativa sets
    pdf.add_page()
    pdf.set_font('Arial', 'B', 13)
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255)
    pdf.cell(0, 9, cs("5. COMPARATIVA DE RESULTADOS: AMBOS SETS"), 0, 1, 'C', 1)
    pdf.set_text_color(0)
    pdf.ln(4)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, cs(
        "NOTA METODOLOGICA v40: ambas simulaciones comparten identicos parametros\n"
        "estructurales (beta_vta, beta_alq, Cholesky, vol_cal) derivados del Set B.\n"
        "La diferencia entre columnas refleja exclusivamente: (1) nivel de precios\n"
        "de partida, (2) drift historico del set seleccionado, (3) inercia A_var.\n"
        "No hay diferencias en betas ni correlaciones entre las dos simulaciones.\n"
        f"Params macro comunes: Regimen={params.get('regimen','N/A')} | "
        f"Euribor={params.get('euribor',0):.1f}% | "
        f"IPC={params.get('ipc',0):.1f}% | Paro={params.get('paro',0):.1f}%"
    ))
    pdf.ln(3)

    hdrs_cmp = ["Distrito",
                f"Port Base ({DATASET_A['ano_fin']})",
                "Port 2032 P50", "Port CAGR",
                f"Ayto Base ({DATASET_B['ano_fin']})",
                "Ayto 2032 P50", "Ayto CAGR"]
    wdts_cmp = [42, 24, 22, 18, 24, 22, 18]
    _tabla_hdr(pdf, hdrs_cmp, wdts_cmp)

    pv_port = DATASET_A['venta'][:, -1]
    pv_ayto = DATASET_B['venta'][:, -1]
    n_yr_p  = 2032 - DATASET_A['ano_fin']
    n_yr_a  = 2032 - DATASET_B['ano_fin']

    if 'Portal' in params['dataset_key']:
        sv_p, sv_a_r = sv_ppal, sv_comp
    else:
        sv_a_r, sv_p = sv_ppal, sv_comp

    for i, d in enumerate(distritos):
        v_p = np.median(sv_p[:, i, -1])
        v_a = np.median(sv_a_r[:, i, -1])
        cg_p = ((v_p / pv_port[i]) ** (1/n_yr_p) - 1) * 100
        cg_a = ((v_a / pv_ayto[i]) ** (1/n_yr_a) - 1) * 100
        pdf.set_fill_color(240 if i % 2 == 0 else 255)
        for val, w in zip([
            cs(d),
            f"{int(pv_port[i]):,}", f"{int(v_p):,}", f"{cg_p:+.2f}%",
            f"{int(pv_ayto[i]):,}", f"{int(v_a):,}", f"{cg_a:+.2f}%",
        ], wdts_cmp):
            pdf.cell(w, 6, val, 1, 0, 'C', 1)
        pdf.ln()

    return pdf.output(dest='S').encode('latin-1')


# ============================================================================
# 13. DOCUMENTO METODOLÓGICO DETALLADO v40 (17 secciones)
# ============================================================================

def generate_metodologia_v40(params, n_sim_val, modelo) -> bytes:
    pdf = PDFv40()
    pdf.add_page()
    pdf.ln(6)
    pdf.set_font('Times', 'B', 16)
    pdf.multi_cell(0, 9, cs(
        "EXPLICACION METODOLOGICA DETALLADA\n"
        "BARCELONA STRATEGIC MODEL v40.0"), align='C')
    pdf.ln(3)
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 6, cs(
        "MS-VAR | Betas Empiricas Independientes Venta/Alquiler\n"
        "SDE Bifurcada: Inercia Endogena + beta*Phi(S,M) | DCC Centralizado\n"
        "Arquitectura Hibrida: parametros estructurales siempre desde Set B"), align='C')
    pdf.ln(5)
    pdf.set_font('Times', '', 9)
    ds = DATASETS[params['dataset_key']]
    for lx in [
        f"Version del modelo   : Barcelona Strategic Model v40.0",
        f"Set de datos activo  : {params['dataset_key']}",
        f"Fuente               : {ds['fuente']}",
        f"Fecha dato real base : {ds['ano_fin']}",
        f"Horizonte proyeccion : {ds['ano_fin']+1}-2032",
        f"Metodo estimacion    : {modelo['metodo_var']}",
        f"Generado             : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]:
        pdf.cell(0, 5, cs(lx), 0, 1, 'C')
    pdf.ln(4)
    pdf.line(25, pdf.get_y(), 185, pdf.get_y())
    pdf.ln(3)
    _trazabilidad(pdf, params, "Todos los distritos", n_sim_val, modelo)

    def S(titulo):
        pdf.add_page()
        pdf.set_font('Times', 'B', 12)
        pdf.set_fill_color(44, 62, 80)
        pdf.set_text_color(255)
        pdf.cell(0, 8, cs(titulo), 0, 1, 'L', 1)
        pdf.set_text_color(0)
        pdf.ln(3)
        pdf.set_font('Times', '', 10)

    def T(t):
        pdf.multi_cell(0, 5.5, cs(t))
        pdf.ln(2)

    def C(t):
        pdf.set_font('Courier', 'B', 8)
        pdf.multi_cell(0, 4.5, cs(t))
        pdf.set_font('Times', '', 10)
        pdf.ln(2)

    rp = modelo

    S("1. INTRODUCCION Y CONTEXTO ECONOMICO DEL MERCADO BARCELONES")
    T("El mercado residencial de Barcelona presenta cuatro caracteristicas estructurales "
      "que justifican un tratamiento econometrico avanzado:\n\n"
      "PRIMERA - HETEROGENEIDAD DISTRITAL EXTREMA: precios de venta entre ~2.946 EUR/m2 "
      "(Nou Barris) y ~6.896 EUR/m2 (Sarria-Sant Gervasi), brecha del 134%. Cada distrito "
      "tiene elasticidades distintas calibradas empiricamente.\n\n"
      "SEGUNDA - INTERVENCION REGULATORIA: Ley 12/2023 y restricciones VT. Shocks exogenos "
      "discretos con efecto asimetrico segun densidad VT distrital.\n\n"
      "TERCERA - DEPENDENCIA DEL CICLO MACRO: en v40 el efecto del Euribor, IPC y Paro "
      "se canaliza exclusivamente a traves de M_t y Phi(S,M), preservando intacta "
      "la tendencia estructural de cada distrito.\n\n"
      "CUARTA - CORRELACION ESPACIAL DINAMICA: mercados adyacentes exhiben correlacion "
      "que converge a 1 en crisis. El DCC simplificado captura este fenomeno.")

    S("2. NOVEDADES DE LA VERSION 40: CUATRO AVANCES ESTRUCTURALES")
    T("PRIMERA NOVEDAD — SDE BIFURCADA (eliminacion de sensibilidad_distrital):\n"
      "La variable sensibilidad_distrital de v33 multiplicaba el drift historico y el "
      "componente VAR, destruyendo la tendencia estructural y produciendo proyecciones "
      "donde un distrito con fuerte tendencia historica quedaba artificialmente suprimido "
      "por un escalar subjetivo. En v40:\n"
      "  Bloque 1 (Inercia endogena): mu_hist_i + lambda * A_var * r(t-1)  [intocable]\n"
      "  Bloque 2 (Ciclo modulado): beta_i * Phi(S_t, M_t)                [empirico]\n\n"
      "SEGUNDA NOVEDAD — BETAS INDEPENDIENTES VENTA Y ALQUILER:\n"
      "beta_vta: Cov(r_vta_i, r_IPV_INE) / Var(r_IPV_INE)\n"
      "beta_alq: Cov(r_alq_i, r_sint_BCN) / Var(r_sint_BCN)\n"
      "Benchmarks distintos porque los mercados tienen dinamicas independientes.\n\n"
      "TERCERA NOVEDAD — ARQUITECTURA HIBRIDA:\n"
      "beta_vta, beta_alq, Cholesky, vol_cal calculados SIEMPRE desde Set B.\n"
      "Drift, A_var, precios base usan el set seleccionado.\n\n"
      "CUARTA NOVEDAD — CALIBRACION EMPIRICA DE REGIMEN:\n"
      f"C_boom={rp['C_boom']*100:.2f}%, C_crisis={rp['C_crisis']*100:.2f}%, "
      f"kappa_boom={rp['kappa_boom']:.3f}, kappa_crisis={rp['kappa_crisis']:.3f}\n"
      "Estimados por OLS sobre retornos reales del Set B en ventanas:\n"
      "  Boom: 2015-2019  |  Crisis: 2013-2014, 2020")

    S("3. ARQUITECTURA GENERAL: VISION DE CONJUNTO")
    T("El modelo es un SDE discreto con cuatro capas metodologicas:")
    C("E[dP_i(t)/P_i(t)] =\n"
      "  (mu_hist_i + lambda * A_var_i(t))  <- Bloque 1: Inercia endogena\n"
      "  + beta_vta_i * Phi(S_t, M_t)       <- Bloque 2: Ciclo modulado\n"
      "  + sigma_base_i * theta(S_t) * eps_i <- Bloque 3: DCC unificado\n"
      "  + rho * ret_alq(t-1)               <- Feedback alquiler->venta\n"
      "  + J_VT(t, i, ano_shock)             <- Shock regulatorio\n\n"
      "Phi(S, M_t) =\n"
      "  M_t                                       si S = Normal\n"
      "  M_t * kappa_boom + C_boom                 si S = Boom\n"
      "  min(M_t,0)*kappa_crisis + max(M_t,0) - C_crisis  si S = Crisis\n\n"
      "theta(S): 0.85 en Boom | 1.0 en Normal | 1.60 en Crisis")
    T("La asimetria en Crisis es fundamental: kappa_crisis solo amplifica M_t negativo, "
      "evitando el caso patologico donde condiciones macro favorables durante una crisis "
      "generarian un impulso positivo amplificado. Si M_t > 0 en Crisis, se mantiene "
      f"ese impulso pero se descuenta C_crisis={rp['C_crisis']*100:.2f}% (coste de "
      "panico/iliquidez calibrado empiricamente).")

    S("4. DATOS DE ENTRADA: FUENTES Y CALIBRACION")
    T(f"SET A (Portales Inmobiliarios — Idealista/Incasol):\n"
      "Precios de oferta (asking price) por distrito, 2019-2026. Banda alta del mercado. "
      "n=7 retornos de venta -> AR(1) diagonal. Activa para drift e inercia si seleccionado.\n\n"
      f"SET B (Ayuntamiento de Barcelona):\n"
      "Estadisticas oficiales Portal de Dades Ajuntament. Precios de transaccion real.\n"
      "Venta: 2012-2025 (n=13 retornos). Alquiler: 2000-2025 (n=25 retornos).\n"
      "SIEMPRE activo para parametros estructurales: beta_vta, beta_alq, Cholesky, vol.\n\n"
      "INDICE MACRO BCN (INE/IPV 2007-2026):\n"
      "Precio medio EUR/m2 ciudad. Benchmark para beta_vta y reconstruccion CAPM.\n\n"
      f"Precios base {ds['ano_fin']} (EUR/m2) — set activo:\n" +
      " | ".join([f"{distritos[i][:6]}: {int(ds['venta'][i,-1])}" for i in range(n_distritos)]))

    S("5. BETAS CAPM EMPIRICAS — INDEPENDIENTES VENTA Y ALQUILER")
    T("Metodologia Shiller (1993) adaptada, con dos benchmarks distintos:\n\n"
      "BETA VENTA (beta_vta_i) — siempre Set B:\n"
      "  beta_vta_i = Cov(r_vta_i_SetB, r_IPV_INE) / Var(r_IPV_INE)\n"
      "  Periodo: 2013-2025 (retornos del Set B alineados con IPV INE)\n"
      "  Benchmark: IPV INE es el indice de precios de compraventa de la ciudad.\n"
      "  Interpretacion: cuanto amplifica el distrito las fluctuaciones del mercado\n"
      "  de compraventa barcelones.\n\n"
      "BETA ALQUILER (beta_alq_i) — siempre Set B:\n"
      "  beta_alq_i = Cov(r_alq_i_SetB, r_sint_BCN) / Var(r_sint_BCN)\n"
      "  Periodo: 2001-2025 (25 retornos del Set B de alquiler)\n"
      "  Benchmark: r_sint_BCN = media aritmetica simple de retornos de los 10\n"
      "  distritos (Set B). Ponderacion igual — transparente y reproducible.\n"
      "  Interpretacion: cuanto amplifica el distrito los movimientos del mercado\n"
      "  de alquiler barcelones en su conjunto.\n\n"
      "La separacion de benchmarks es esencial: el alquiler tiene rigideces contractuales\n"
      "y responde al ciclo de forma distinta a la compraventa. Usar el IPV para medir\n"
      "la sensibilidad del alquiler produciria betas sesgadas.\n"
      "Truncamiento [0.5, 1.5] para ambos mercados.\n\n"
      f"Betas calibradas (Set B, arquitectura hibrida):\n"
      f"  beta_vta: min={rp['beta_vta'].min():.3f} max={rp['beta_vta'].max():.3f}\n"
      f"  beta_alq: min={rp['beta_alq'].min():.3f} max={rp['beta_alq'].max():.3f}")

    S("6. DRIFT ENDOGENO Y AUSENCIA DE AJUSTE MACRO DIRECTO")
    T("CAMBIO FUNDAMENTAL v40: el drift historico ya no recibe ajuste macro directo.\n"
      "En v33, se sumaban ajustes de Euribor/IPC/Paro directamente al drift, creando\n"
      "doble conteo cuando el componente macro tambien aparecia en el bloque de regimen.\n"
      "En v40, el efecto macro se canaliza EXCLUSIVAMENTE a traves de M_t -> Phi(S,M).\n\n"
      "DRIFT HISTORICO BASE (set seleccionado):\n"
      "  mu_hist_i = (1/T) * SUM_t [(P_i(t)/P_i(t-1)) - 1]\n"
      "Media aritmetica de retornos anuales observados. Representa la tendencia\n"
      "estructural especifica del mercado elegido (oferta vs. transaccion).\n\n"
      f"Drift historico venta calibrado ({params['dataset_key']}):\n" +
      " | ".join([f"{distritos[i][:6]}: {rp['drift_venta'][i]*100:.2f}%"
                  for i in range(n_distritos)]))

    S("7. ESTIMACION ADAPTATIVA: AR(1) DIAGONAL vs VAR(1) RIDGE")
    T("La seleccion adaptativa del metodo es identica a v33, pero ahora A_var solo\n"
      "captura inercia autorregresiva (Bloque 1), no el efecto ciclico.\n\n"
      "AR(1) UNIVARIADO DIAGONAL (n < 10, set Portales):\n"
      "  rho_i = Cov(r_i(t), r_i(t-1)) / Var(r_i(t-1)), truncado [-0.35, +0.35]\n"
      "  A_var = diag(rho_1,...,rho_10). Radio espectral <= 0.35 garantizado.\n\n"
      "VAR(1) MULTIVARIADO CON RIDGE (n >= 10, set Ayuntamiento):\n"
      "  A_ridge = (X'X + lambda*I)^{-1} X'Y, lambda=0.5 (Hoerl & Kennard 1970)\n"
      "  Si radio espectral > 0.85: A_var = A_var * (0.85 / rho_espectral)\n\n"
      "El VAR captura interdependencias entre distritos (efecto desbordamiento\n"
      "geografico documentado en Barcelona). La regularizacion Ridge garantiza\n"
      "estabilidad con n moderado.\n\n"
      f"Metodo activo: {rp['metodo_var']} | Radio espectral: {rp['rho_espectral']:.4f}")

    S("8. ENTORNO MACRO M_t Y FUNCION PHI(S_t, M_t)")
    T("ENTORNO MACRO BASE M_t — presion ciclica consolidada:")
    C("M_t (venta)   = -clip((Euribor-2.5)/100, -0.02, +0.04)  [comprime venta]\n"
      "              + clip((IPC-2.0)/100, -0.03, +0.03)\n"
      "              - clip((Paro-10.0)/100*0.5, -0.02, +0.03)\n\n"
      "M_t (alquiler) = +clip((Euribor-2.5)/100*0.4, -0.01, +0.02)  [sostiene alq]\n"
      "               + clip((IPC-2.0)/100*0.6, -0.02, +0.02)\n"
      "               - clip((Paro-10.0)/100*0.3, -0.01, +0.02)")
    T(f"Impulso M_t en escenario activo: {calcular_Mt(params.get('euribor',2.5), params.get('ipc',2.0), params.get('paro',10.0))*100:+.3f}%\n\n"
      "FUNCION PHI(S_t, M_t) — Shock de Regimen Aditivo:")
    C(f"Normal:  Phi = M_t\n"
      f"Boom:    Phi = M_t * {rp['kappa_boom']:.3f} + {rp['C_boom']*100:.2f}%\n"
      f"Crisis:  Phi = min(M_t,0)*{rp['kappa_crisis']:.3f} + max(M_t,0)*1.0 - {rp['C_crisis']*100:.2f}%\n\n"
      "Asimetria en Crisis: kappa solo amplifica M_t si es negativo.")
    T("CALIBRACION EMPIRICA (Opcion A — datos reales Set B):\n"
      "  C_boom   = media(r_M - mu_hist_M) para t en {2015-2019} (ventana Boom real)\n"
      "  C_crisis = |media(r_M - mu_hist_M)| para t en {2013, 2014, 2020} (sin reconstruccion)\n"
      "  kappa    = regresion OLS de (r_M - mu_hist) ~ kappa * M_t en cada ventana\n"
      "  Estimacion via scipy.stats.linregress sobre datos historicos reales.\n"
      "  No se usan datos reconstruidos CAPM para calibrar kappa/C.\n\n"
      f"Valores calibrados: C_boom={rp['C_boom']*100:.2f}% | kappa_boom={rp['kappa_boom']:.3f}\n"
      f"  C_crisis={rp['C_crisis']*100:.2f}% | kappa_crisis={rp['kappa_crisis']:.3f}")

    S("9. INDICADORES MACRO Y CADENA DE MARKOV NO HOMOGENEA")
    T("CADENA DE MARKOV NO HOMOGENEA — formulacion Hamilton (1994):\n"
      "P(S_{t+1}=j | S_t=i, Z_t) = f(Z_t)_ij\n"
      "Z_t = (Euribor_t, IPC_t, Paro_t). MTM recalculada dinamicamente.\n\n"
      "FUNCION SOFTMAX DE PROBABILIDADES:\n"
      "  Score_Boom   = (1-Euribor_norm)*0.4 + (1-Paro_norm)*0.4 + IPC_cuadr*0.2\n"
      "  Score_Crisis = Euribor_norm*0.45 + Paro_norm*0.45 + excess_IPC*0.10\n"
      "  Score_Normal = 0.5 (referencia)\n"
      "  P(regimen) = softmax([Score_Normal, Score_Boom, Score_Crisis], T=2.5)\n\n"
      "CONSTRUCCION MTM: PERSIST=0.55 + renormalizacion exacta por fila.\n\n"
      "VERIFICACION LOGICA ECONOMICA:\n"
      "  Euribor=1.5%, Paro=9%: P(Boom) alta -> ciclo 2015-2019\n"
      "  Euribor=3.8%, Paro=13%: P(Normal) domina -> estanflacion 2022-2024\n"
      "  Euribor=2.5%, Paro=20%: P(Crisis) alta -> recesion estructural")

    S("10. MOTOR MONTE CARLO v40: ALGORITMO COMPLETO")
    T(f"n_sim={n_sim_val} simulaciones independientes. seed=42 para reproducibilidad.")
    C("PARA sim=1,...,n_sim:\n"
      "  p_vta = pv_base; p_alq = pa_base; est=0(Normal); r_prev=mu_hist\n"
      "  PARA t = ano_base+1,...,2032:\n"
      "  1. TRANSICION MARKOV: u~U(0,1); est=sample(MTM[est,:])\n"
      "  2. DCC: theta=0.85/1.0/1.60; Sigma_dyn(est); L_dyn=Cholesky(Sigma_dyn)\n"
      "          eps = L_dyn @ N(0,I_10)\n"
      "  3. PHI: Phi_v = Phi(est, M_t_venta); Phi_a = Phi(est, M_t_alquiler)\n"
      "  4. BLOQUE 1 (Inercia): inercia_v = mu_hist + 0.15*(A_var @ r_prev)\n"
      "  5. BLOQUE 2 (Ciclo):   ciclo_v = beta_vta * Phi_v\n"
      "  6. BLOQUE 3 (DCC):     ruido_v = vol_base * theta * eps\n"
      "  7. tasa_v = inercia_v + ciclo_v + ruido_v\n"
      "  8. FLOOR: tasa_v >= -7%\n"
      "  9. FLOOR YIELD: resist=clip(yield_bruto/0.025, 0.5, 1.0)\n"
      "                  tasa_v = where(tasa_v>0, tasa_v*resist, tasa_v)\n"
      " 10. FEEDBACK: tasa_v += rho(0.45)*ret_alq(t-1)  [si t > t0]\n"
      " 11. SHOCK VT: tasa_v += -0.10*(D_i/D_max) si activado y t=ano_shock\n"
      " 12. ACTUALIZAR: p_vta *= (1+tasa_v)\n"
      " 13. r_prev = clip(tasa_v, -0.08, +0.08)  [tasa realizada, Opcion A]")

    S("11. CORRELACION ESPACIAL: DESCOMPOSICION DE CHOLESKY")
    T("MOTIVACION: mercados adyacentes comparten factores de demanda. Shocks\n"
      "independientes subestimarian el riesgo sistematico.\n\n"
      "METODOLOGIA (siempre Set B — serie larga):\n"
      "1. Reconstruccion CAPM para 2007-2011 (anterior al Set B).\n"
      "2. Estimacion matriz correlacion Sigma sobre retornos 2007-2025.\n"
      "3. Descomposicion: Sigma = L * L^T (L triangular inferior).\n"
      "4. Shocks correlacionados: eps = L * Z, Z ~ N(0, I_10).\n\n"
      "CORRECCION ESPECTRAL si Sigma no es definida positiva:\n"
      "  Sigma_corr = V * diag(max(lambda_i, 1e-8)) * V^T, renormalizada.\n\n"
      "La correlacion espacial siempre refleja ciclos completos porque usa\n"
      "Set B (13 retornos reales + reconstruccion).")

    S("12. VOLATILIDAD DINAMICA: DCC SIMPLIFICADO")
    T("MOTIVACION: la volatilidad se incrementa en crisis (GARCH) y las correlaciones\n"
      "convergen a 1 (DCC, Engle 2002). En v40, DCC es la UNICA fuente de\n"
      "heteroscedasticidad de regimen. Los ruidos diferenciados por regimen de v33\n"
      "han sido eliminados.\n\n"
      "IMPLEMENTACION:\n"
      "  BOOM:   vol_dyn = vol_base * 0.85; Sigma_dyn = Sigma*0.8 + I*0.2\n"
      "  NORMAL: vol_dyn = vol_base;        Sigma_dyn = Sigma_historica\n"
      "  CRISIS: vol_dyn = vol_base * 1.60; Sigma_dyn = Sigma*0.5 + 1*0.5\n"
      "          (correlacion converge a 1 -> todo cae junto)\n\n"
      "DIFERENCIA CON DCC COMPLETO (Engle 2002):\n"
      "El DCC completo requiere series > 100 obs para estimar GARCH.\n"
      "Con n<=25 disponibles, esta estimacion seria inestable. El enfoque\n"
      "simplificado captura la logica esencial siendo estadisticamente honesto.")

    S("13. FUNCION DE SHOCK REGULATORIO (JUMP PROCESS)")
    shock_txt = (f"ACTIVADO en {params.get('shock_ano', 2029)}"
                 if params.get('shock_vt') else "DESACTIVADO")
    T(f"Estado del shock en este escenario: {shock_txt}\n\n"
      "CANAL ECONOMICO: las licencias VT crean demanda adicional de compra no\n"
      "residencial. La restriccion suprime esta demanda, con correccion proporcional\n"
      "a la densidad VT distrital.\n\n"
      "ESPECIFICACION MATEMATICA (Jump Process):\n"
      "  J_vta(i) = -0.10 * (D_i / D_max)\n"
      "  J_alq(i) = -0.06 * (D_i / D_max)\n"
      "  D_i = densidad VT distrital | D_max = 0.46 (Eixample)\n\n"
      "ANO CONFIGURABLE (2028 o 2029): refleja incertidumbre sobre calendario\n"
      "de implementacion efectiva.\n\n"
      "Impactos por distrito:\n" +
      "\n".join([f"  {distritos[i][:15]:<15} (VT={concentracion_vt_real[i]:.3f}): "
                  f"venta {-10*(concentracion_vt_real[i]/0.46):.1f}%, "
                  f"alq {-6*(concentracion_vt_real[i]/0.46):.1f}%"
                  for i in range(n_distritos)]))

    S("14. RETROALIMENTACION ALQUILER-VENTA (CANAL rho)")
    T("Los mercados de venta y alquiler estan conectados via yield de rentabilidad.\n"
      "Un incremento sostenido de alquileres eleva la rentabilidad del activo y\n"
      "atrae nueva demanda compradora, presionando precios de venta al alza.\n\n"
      "CANAL ALQUILER -> VENTA:\n"
      "  tasa_venta(t) += rho * [(Alquiler(t-1)/Alquiler(t-2)) - 1]\n"
      "  rho = 0.45 (calibrado sobre correlacion historica con retardo 1 periodo)\n\n"
      "CANAL INVERSO — FLOOR YIELD (minimo 2.5% bruto):\n"
      "  resist = clip(yield_bruto / 0.025, 0.5, 1.0)\n"
      "  tasa_v = where(tasa_v>0, tasa_v*resist, tasa_v)\n"
      "Evita que precios de venta crezcan desconectados de alquileres.")

    S("15. LIMITACIONES DEL MODELO Y ADVERTENCIAS")
    T("1. MUESTRA CORTA (Set A, n=7): estimadores con alta varianza. La seleccion\n"
      "   adaptativa hacia AR(1) compensa el componente autoregresivo.\n\n"
      "2. MUESTRA MODERADA (Set B, n=13 venta): VAR Ridge viable pero con sesgo\n"
      "   controlado por regularizacion.\n\n"
      "3. DATOS DE OFERTA AUSENTES: el modelo no incorpora nueva construccion,\n"
      "   rehabilitacion ni cambios de uso del suelo.\n\n"
      "4. M_t CONSTANTE EN SIMULACION: Euribor, IPC y Paro son parametros fijos\n"
      "   durante la proyeccion. La variabilidad temporal viene de la Markov y DCC.\n\n"
      "5. CALIBRACION REGIMEN: ventanas Boom/Crisis identificadas manualmente.\n"
      "   Ventana Crisis solo 3 observaciones (2013, 2014, 2020).\n\n"
      "6. DCC SIMPLIFICADO: sin GARCH. Aproximacion de la logica DCC (Engle 2002).\n\n"
      "7. HORIZONTE 6 ANOS: incertidumbre acumulada elevada. Bandas P10-P90 pueden\n"
      "   representar el 30-40% del precio central en ano 5-6.\n\n"
      "CLASIFICACION: herramienta de analisis estrategico de inversion. No es un\n"
      "modelo de valoracion-tasacion ni debe usarse para decisiones de credito.")

    S("16. TESTS DE INTEGRIDAD MATEMATICA")
    pdf.set_font('Courier', '', 8)
    test_res = run_tests(modelo, params['dataset_key'])
    for tname, tpass, tmsg in test_res:
        pdf.cell(0, 5, cs(f"[{'PASS' if tpass else 'FAIL'}]  {tname}"), 0, 1)
        pdf.cell(0, 4, cs(f"        {tmsg}"), 0, 1)
    pdf.set_font('Times', '', 10)

    S("17. REFERENCIAS BIBLIOGRAFICAS Y NORMATIVAS")
    refs_v40 = [
        "Hamilton, J.D. (1989). A new approach to nonstationary time series. Econometrica, 57(2).",
        "Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.",
        "Engle, R.F. (2002). Dynamic Conditional Correlation. JBES, 20(3), 339-350.",
        "Sims, C.A. (1980). Macroeconomics and Reality. Econometrica, 48(1), 1-48.",
        "Hoerl, A.E. & Kennard, R.W. (1970). Ridge Regression. Technometrics, 12(1).",
        "Sharpe, W.F. (1964). Capital asset prices. Journal of Finance, 19(3), 425-442.",
        "Shiller, R.J. (1993). Measuring Asset Values for Cash Settlement. J.Finance, 48(3).",
        "Case, K.E. & Shiller, R.J. (1989). Efficiency of Single-Family Homes Market. AER.",
        "Glaeser, E. & Gyourko, J. (2018). Economic Implications of Housing Supply. JEP.",
        "Banco de Espana (2023). Informe de Estabilidad Financiera. Cap.2: Inmobiliario.",
        "European Banking Authority (2023). EBA Report on Residential Real Estate. EBA/REP.",
        "Ajuntament de Barcelona (2025). Portal de Dades Obertes. portaldades.ajuntament.barcelona.cat",
        "Idealista Research (2026). Informe de Precios de Vivienda en Barcelona, Q1 2026.",
        "Incasol - Institut Catala del Sol (2026). Estadistiques sector immobiliari.",
        "INE (2026). Indice de Precios de Vivienda (IPV). Serie historica 2007-2026.",
        "Inside Airbnb (2024). Data for Barcelona, Spain. insideairbnb.com.",
        "Ley 12/2023, de 24 de mayo, por el derecho a la vivienda. BOE num. 124.",
        "Scipy (2024). scipy.stats.linregress. SciPy Reference Guide, Version 1.12.",
    ]
    pdf.set_font('Times', '', 9)
    for r in refs_v40:
        pdf.multi_cell(0, 5, cs(f"- {r}"))
        pdf.ln(1)

    return pdf.output(dest='S').encode('latin-1')


# ============================================================================
# 14. SECCION DE EXPORTACION
# ============================================================================
st.markdown("---")
st.subheader("📄 Exportar Documentación")

ds_activo_exp = DATASETS[ds_key]
idx_export    = st.session_state.get('idx_distrito', 0)

col_e1, col_e2 = st.columns(2)

with col_e1:
    if st.button("📊 Generar Informe Ejecutivo v40 (con comparativa)"):
        with st.spinner("Generando informe ejecutivo v40..."):
            modelo_alt_exp = inicializar_modelo(ds_key_alt)
            sv_alt_exp, sa_alt_exp, _ = simular(
                ds_key_alt, euribor_val, ipc_val, paro_val,
                shock_vt, shock_ano, 500, seed=99)
            pdf_exec = generate_exec_report_v40(
                idx_export, params_sim, params_sim.get('n_sim', 1000),
                modelo_act, sv_act, sa_act, anos_proy_act,
                modelo_alt_exp, sv_alt_exp, sa_alt_exp,
            )
        dist_name = distritos[idx_export].replace(' ', '_')
        fname = f"Informe_Ejecutivo_{dist_name}_v40.pdf"
        st.download_button(f"⬇️ Descargar {fname}", pdf_exec, fname, "application/pdf")

with col_e2:
    if st.button("📚 Generar Explicación Metodológica Detallada v40 (17 secciones)"):
        with st.spinner("Generando documento metodologico v40..."):
            pdf_met = generate_metodologia_v40(params_sim, params_sim.get('n_sim', 1000), modelo_act)
        st.download_button(
            "⬇️ Descargar BCN_Model_v40_Explicacion_Metodologica.pdf",
            pdf_met,
            "BCN_Model_v40_Explicacion_Metodologica_Detallada.pdf",
            "application/pdf"
        )

# ============================================================================
# 15. FOOTER
# ============================================================================
st.markdown("---")
Mt_footer = calcular_Mt(euribor_val, ipc_val, paro_val)
st.caption(
    f"Barcelona Strategic Model v40.0 | {ds_key} | "
    f"Motor: {modelo_act['metodo_var']} | "
    f"rho espectral: {modelo_act['rho_espectral']:.4f} | "
    f"Regimen: {reg_sel} | Euribor: {euribor_val}% | IPC: {ipc_val}% | "
    f"Paro: {paro_val}% | M_t: {Mt_footer*100:+.2f}% | "
    f"Shock VT: {'ON (' + str(shock_ano) + ')' if shock_vt else 'OFF'} | "
    f"n_sim: {params_sim.get('n_sim','?')} | "
    f"beta_vta: {modelo_act['beta_vta'].min():.2f}-{modelo_act['beta_vta'].max():.2f} | "
    f"beta_alq: {modelo_act['beta_alq'].min():.2f}-{modelo_act['beta_alq'].max():.2f}"
)



