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
# 0. LOGGING Y CONFIGURACION VISUAL
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BCNModel_V40")

st.set_page_config(
    page_title="Barcelona Strategic Model v40.0",
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
# 1. DATOS ESTRUCTURALES Y SETS
# ============================================================================

distritos = [
    'Ciutat Vella', 'Eixample', 'Gracia', 'Horta Guinardo', 'Les Corts',
    'Nou Barris', 'Sant Andreu', 'Sant Marti', 'Sants-Montjuic', 'Sarria-Sant Gervasi'
]
n_distritos = len(distritos)

concentracion_vt_real = np.array([0.150, 0.460, 0.105, 0.030, 0.035,
                                   0.004, 0.015, 0.120, 0.110, 0.050])
rho_vta_alq = 0.45

# Índice macro agregado Barcelona EUR/m2 (INE/IPV)
datos_agregados_bcn = {
    2007: 4071, 2008: 3737, 2009: 3728, 2010: 3752, 2011: 3398, 2012: 3076,
    2013: 2992, 2014: 3061, 2015: 3297, 2016: 3649, 2017: 4154, 2018: 4232,
    2019: 4135, 2020: 4018, 2021: 3910, 2022: 4063, 2023: 4167, 2024: 4700,
    2025: 5042, 2026: 5200
}
anos_macro   = np.array(list(datos_agregados_bcn.keys()))
precios_macro = np.array(list(datos_agregados_bcn.values()), dtype=float)
# Retornos del mercado base (IPV)
ret_M_ipv_full = (precios_macro[1:] / precios_macro[:-1]) - 1
anos_ret_M = anos_macro[1:]

# ---------------------------------------------------------------------------
# SET A — PORTALES INMOBILIARIOS (Idealista / Incasol)
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
# SET B — AYUNTAMIENTO DE BARCELONA (Venta: 2012-2025, Alquiler: 2000-2025)
# ---------------------------------------------------------------------------
_venta_ayto_src = np.array([
    [2355, 2767, 1965, 2974, 3339, 2590, 2215, 1668, 2151, 2432],
    [2266, 2619, 1739, 2857, 3105, 2369, 1770, 1536, 1756, 2122],
    [2387, 2683, 1843, 2981, 3108, 2374, 1776, 1398, 1823, 2169],
    [2730, 2983, 2186, 3076, 3509, 2575, 2101, 2029, 2100, 2555],
    [2989, 3381, 2354, 3284, 3750, 2953, 2038, 1621, 1994, 2458],
    [3457, 3869, 2815, 3880, 4330, 3445, 2464, 1918, 2419, 3001],
    [3833, 4111, 3137, 4204, 4755, 3824, 2690, 2174, 2626, 3069],
    [3592, 4045, 3153, 3980, 4497, 3766, 2648, 2191, 2717, 3029],
    [3584, 3977, 3092, 4161, 4388, 3695, 2657, 2098, 2671, 3095],
    [3515, 4034, 3112, 4001, 4563, 3717, 2811, 2073, 2770, 3095],
    [3752, 4391, 3220, 4152, 4811, 3990, 2897, 2495, 2905, 3490],
    [3749, 4582, 3305, 4260, 5054, 4059, 2833, 2373, 2810, 3377],
    [3845, 4869, 3446, 4666, 5200, 4320, 3213, 2531, 3124, 3693],
    [4144, 5325, 3869, 5100, 5629, 4814, 3464, 3005, 3505, 4138],
], dtype=float)
_col_map_ayto = [0, 1, 5, 6, 3, 7, 8, 9, 2, 4]
_venta_ayto_canon = _venta_ayto_src[:, _col_map_ayto].T

_alq_ayto_src = np.array([
    [5.5,  5.8,  6.1,  7.6,  7.4,  6.3,  5.8, 5.4, 5.5,  5.7],
    [6.4,  6.8,  6.7,  8.6,  8.2,  7.0,  6.4, 6.2, 6.1,  6.5],
    [7.3,  7.5,  7.5,  8.9,  9.1,  8.1,  7.1, 7.2, 6.7,  7.5],
    [7.6,  7.9,  7.8,  9.3,  9.4,  8.4,  7.7, 7.3, 7.3,  7.7],
    [8.6,  8.5,  8.7, 10.0, 10.0,  9.3,  8.3, 7.9, 7.8,  8.2],
    [9.8,  9.5,  9.5, 10.8, 10.8, 10.2,  9.2, 8.9, 9.0,  9.2],
    [10.7, 10.3, 10.5, 11.9, 11.6, 11.3, 10.2, 9.9, 9.6, 10.3],
    [11.6, 11.3, 11.3, 13.0, 12.7, 12.5, 11.1, 10.8, 10.7, 11.2],
    [12.6, 11.9, 12.2, 13.4, 13.3, 13.0, 12.0, 11.5, 11.5, 11.8],
    [12.4, 11.6, 11.9, 12.9, 12.9, 12.8, 11.4, 11.2, 11.1, 11.6],
    [12.1, 11.3, 11.5, 12.3, 12.8, 12.6, 10.8, 10.7, 10.7, 11.2],
    [12.0, 11.2, 11.3, 12.1, 12.8, 12.0, 10.6, 10.3, 10.4, 10.9],
    [11.7, 10.8, 10.8, 11.6, 12.3, 11.4, 10.1, 9.5,  9.8, 10.5],
    [11.4, 10.2, 10.1, 11.2, 11.7, 11.0,  9.3, 8.7,  9.2,  9.9],
    [11.4, 10.3,  9.9, 11.0, 11.9, 10.8,  9.0, 8.4,  8.9,  9.7],
    [12.4, 11.2, 10.7, 12.0, 13.0, 11.8,  9.9, 8.9,  9.6, 10.6],
    [13.9, 12.4, 11.7, 13.2, 14.3, 12.9, 10.7, 9.6, 10.4, 11.7],
    [15.5, 13.4, 12.9, 14.2, 15.4, 14.3, 11.9, 10.6, 11.4, 12.9],
    [14.5, 13.7, 13.0, 14.5, 15.4, 14.0, 11.9, 11.2, 11.7, 13.2],
    [14.8, 14.3, 13.5, 15.0, 15.7, 14.6, 12.7, 11.7, 12.2, 13.9],
    [14.3, 14.2, 13.6, 14.7, 15.6, 14.4, 12.8, 11.7, 12.3, 13.8],
    [13.2, 13.3, 13.0, 13.8, 14.9, 13.8, 12.2, 11.3, 11.7, 13.3],
    [15.8, 14.9, 14.1, 15.2, 16.2, 15.3, 13.1, 12.1, 12.5, 14.5],
    [16.7, 16.5, 16.3, 16.8, 18.2, 16.9, 14.5, 13.3, 13.8, 16.2],
    [16.6, 16.6, 15.6, 17.4, 18.4, 17.5, 14.5, 13.1, 13.7, 16.2],
    [17.8, 16.5, 16.3, 17.2, 18.9, 17.3, 14.6, 13.1, 14.4, 16.2],
], dtype=float)
_alq_ayto_canon = _alq_ayto_src[:, _col_map_ayto].T

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
    'Datos Portales Inmobiliarios': DATASET_A,
    'Datos Oficiales Ayuntamiento de Barcelona': DATASET_B,
}

# ============================================================================
# 2. FUNCIONES MACRO Y MARKOV NO HOMOGÉNEA
# ============================================================================
def calcular_M_t(euribor, ipc, paro):
    """Calcula la presión macroeconómica consolidada (M_t)"""
    ipc_real_adj    = np.clip((ipc  - 2.0) / 100.0, -0.03, 0.03)
    euribor_adj_vta = -np.clip((euribor - 2.5) / 100.0, -0.02, 0.04)
    paro_adj        = -np.clip((paro  - 10.0) / 100.0 * 0.5, -0.02, 0.03)
    return ipc_real_adj + euribor_adj_vta + paro_adj

def calcular_probabilidades_regimen(euribor, ipc, paro):
    euribor_norm = np.clip((euribor - 0.0) / 5.0, 0.0, 1.0)
    ipc_norm     = np.clip((ipc     - 0.0) / 8.0, 0.0, 1.0)
    paro_norm    = np.clip((paro   - 5.0) / 25.0, 0.0, 1.0)
    score_boom   = (1.0 - euribor_norm) * 0.4 + (1.0 - paro_norm) * 0.4 + np.clip(ipc_norm * (1 - ipc_norm) * 4, 0, 1) * 0.2
    score_crisis = euribor_norm * 0.45 + paro_norm * 0.45 + np.clip((ipc_norm - 0.5), 0, 0.5) * 0.1
    raw  = np.array([0.5, score_boom, score_crisis])
    exps = np.exp(raw * 2.5)
    return exps / exps.sum()

def construir_mtm_macro(euribor, ipc, paro):
    p_norm, p_boom, p_crisis = calcular_probabilidades_regimen(euribor, ipc, paro)
    PERSIST = 0.55
    mtm = np.array([
        [PERSIST + (1 - PERSIST) * p_norm, (1 - PERSIST) * p_boom, (1 - PERSIST) * p_crisis],
        [0.35 * (1 - p_boom), PERSIST + (1 - PERSIST) * p_boom, 0.05 + 0.10 * p_crisis],
        [0.12 + 0.20 * p_norm, 0.02, PERSIST + (1 - PERSIST) * p_crisis],
    ])
    for r in range(3):
        s = mtm[r].sum()
        if s > 0: mtm[r] /= s
    return mtm

# ============================================================================
# 3. MOTOR V40: ARQUITECTURA HÍBRIDA (CALIBRACIÓN)
# ============================================================================
@st.cache_data
def inicializar_modelo_v40(dataset_key: str):
    """Calibra el modelo V40 con separación estricta: inercia local, estructura Set B."""
    ds_select = DATASETS[dataset_key]
    ds_B      = DATASETS['Datos Oficiales Ayuntamiento de Barcelona']

    # --- 1. INERCIA ENDÓGENA (Siempre del Set Seleccionado) ---
    ret_v_sel = (ds_select['venta'][:, 1:] / ds_select['venta'][:, :-1]) - 1
    ret_a_sel = (ds_select['alquiler'][:, 1:] / ds_select['alquiler'][:, :-1]) - 1
    mu_hist_v = ret_v_sel.mean(axis=1)
    mu_hist_a = ret_a_sel.mean(axis=1)

    n_ret_sel = ret_v_sel.shape[1]
    if n_ret_sel >= 10:
        Y = ret_v_sel[:, 1:].T; X = ret_v_sel[:, :-1].T
        lam = 0.5
        try:
            XtX = X.T @ X + lam * np.eye(n_distritos)
            A_var = np.linalg.solve(XtX, X.T @ Y)
            rho_spec = np.abs(np.linalg.eigvals(A_var)).max()
            if rho_spec > 0.85: A_var = A_var * (0.85 / rho_spec)
            metodo_var = f"VAR(1) Ridge (lambda={lam}, n={n_ret_sel})"
        except Exception:
            A_var = np.diag(np.zeros(n_distritos))
            metodo_var = "VAR(1) fallback=0"
    else:
        ar1 = np.zeros(n_distritos)
        for i in range(n_distritos):
            r = ret_v_sel[i]
            if len(r) >= 3:
                cv = np.cov(r[1:], r[:-1])[0, 1]
                vv = np.var(r[:-1])
                ar1[i] = np.clip(cv / vv if vv > 1e-10 else 0.0, -0.35, 0.35)
        A_var = np.diag(ar1)
        metodo_var = f"AR(1) diagonal (n={n_ret_sel})"
    rho_final = np.abs(np.linalg.eigvals(A_var)).max() if np.any(A_var) else 0.0

    # --- 2. PARÁMETROS ESTRUCTURALES (Siempre del Set B) ---
    ret_v_B = (ds_B['venta'][:, 1:] / ds_B['venta'][:, :-1]) - 1
    ret_a_B = (ds_B['alquiler'][:, 1:] / ds_B['alquiler'][:, :-1]) - 1

    # 2.1 Betas Venta (Benchmark: IPV INE)
    idx_ipv_B = np.where((anos_ret_M >= ds_B['anos_venta'][1]) & (anos_ret_M <= ds_B['anos_venta'][-1]))[0]
    ret_M_ipv_B = ret_M_ipv_full[idx_ipv_B]
    
    betas_vta = []
    for i in range(n_distritos):
        rv = ret_v_B[i, -len(ret_M_ipv_B):]
        rm = ret_M_ipv_B
        var = np.var(rm)
        b = np.cov(rv, rm)[0, 1] / var if var > 1e-10 else 1.0
        betas_vta.append(np.clip(b, 0.5, 1.5))
    betas_vta = np.array(betas_vta)

    # 2.2 Betas Alquiler (Benchmark: Índice Sintético Alquiler BCN Set B)
    r_M_alq = ret_a_B.mean(axis=0) # Media aritmética simple
    betas_alq = []
    for i in range(n_distritos):
        var = np.var(r_M_alq)
        b = np.cov(ret_a_B[i], r_M_alq)[0, 1] / var if var > 1e-10 else 1.0
        betas_alq.append(np.clip(b, 0.5, 1.5))
    betas_alq = np.array(betas_alq)

    # 2.3 Volatilidad Estructural
    vol_base = np.clip(ret_v_B.std(axis=1), 0.015, 0.10)

    # 2.4 Matriz Cholesky (Reconstrucción CAPM para serie larga 2007-2025)
    reconstruidos = np.zeros((n_distritos, len(anos_macro)))
    idx_ini = np.where(anos_macro == ds_B['anos_venta'][0])[0][0]
    for ti, yr in enumerate(ds_B['anos_venta']):
        m_idx = np.where(anos_macro == yr)[0]
        if len(m_idx): reconstruidos[:, m_idx[0]] = ds_B['venta'][:, ti]
    np.random.seed(999)
    for t in range(idx_ini - 1, -1, -1):
        rmt = (precios_macro[t + 1] / precios_macro[t]) - 1
        for d in range(n_distritos):
            ruido = np.random.normal(0, 0.003)
            r_d = betas_vta[d] * rmt + ruido
            if reconstruidos[d, t + 1] > 0:
                reconstruidos[d, t] = reconstruidos[d, t + 1] / (1 + r_d)
            else:
                reconstruidos[d, t] = ds_B['venta'][d, 0] / (1 + r_d) ** (idx_ini - t)

    ret_full = np.zeros_like(reconstruidos)
    for t in range(1, len(anos_macro)):
        denom = np.where(reconstruidos[:, t - 1] > 0, reconstruidos[:, t - 1], 1.0)
        ret_full[:, t] = (reconstruidos[:, t] / denom) - 1
    ret_full = ret_full[:, 1:]
    corr = np.corrcoef(ret_full)
    try:
        L_matrix = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-8)
        corr_r = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_diag = np.sqrt(np.diag(corr_r))
        corr_r = corr_r / np.outer(d_diag, d_diag)
        try:
            L_matrix = np.linalg.cholesky(corr_r)
        except Exception:
            L_matrix = np.eye(n_distritos)
        corr = corr_r

    # --- 3. CALIBRACIÓN EMPÍRICA (OLS) DE SHOCKS DE RÉGIMEN ---
    macro_hist = {
        2013: {'e': 0.5, 'i': 1.4, 'p': 24.0}, 2014: {'e': 0.5, 'i': -0.2, 'p': 22.0},
        2015: {'e': 0.2, 'i': -0.5, 'p': 19.0}, 2016: {'e': 0.0, 'i': -0.2, 'p': 16.0},
        2017: {'e': -0.1, 'i': 2.0, 'p': 13.0}, 2018: {'e': -0.2, 'i': 1.7, 'p': 11.0},
        2019: {'e': -0.3, 'i': 0.7, 'p': 10.0}, 2020: {'e': -0.3, 'i': -0.3, 'p': 13.0}
    }
    
    # Alinear retornos del IPV con los años históricos
    mu_hist_M = ret_M_ipv_full.mean()
    idx_yr = lambda yr: np.where(anos_ret_M == yr)[0][0]
    
    idx_boom = [idx_yr(y) for y in range(2015, 2020)]
    M_t_boom = [calcular_M_t(macro_hist[y]['e'], macro_hist[y]['i'], macro_hist[y]['p']) for y in range(2015, 2020)]
    r_M_boom_dev = ret_M_ipv_full[idx_boom] - mu_hist_M
    C_boom = np.mean(r_M_boom_dev)
    kappa_boom = np.polyfit(M_t_boom, r_M_boom_dev, 1)[0]
    
    idx_crisis = [idx_yr(y) for y in [2013, 2014, 2020]]
    M_t_crisis = [calcular_M_t(macro_hist[y]['e'], macro_hist[y]['i'], macro_hist[y]['p']) for y in [2013, 2014, 2020]]
    r_M_crisis_dev = ret_M_ipv_full[idx_crisis] - mu_hist_M
    C_crisis = np.abs(np.mean(r_M_crisis_dev))
    kappa_crisis = np.polyfit(M_t_crisis, r_M_crisis_dev, 1)[0]

    return {
        'reconstruidos': reconstruidos,
        'L':             L_matrix,
        'corr':          corr,
        'vol_cal':       vol_base,
        'betas_vta':     betas_vta,
        'betas_alq':     betas_alq,
        'mu_hist_v':     mu_hist_v,
        'mu_hist_a':     mu_hist_a,
        'A_var':         A_var,
        'metodo_var':    metodo_var,
        'rho_espectral': rho_final,
        'n_ret_venta':   n_ret_sel,
        'reg_params': {
            'C_boom': C_boom, 'kappa_boom': kappa_boom,
            'C_crisis': C_crisis, 'kappa_crisis': kappa_crisis
        }
    }

# ============================================================================
# 4. MOTOR DE SIMULACION (SDE V40)
# ============================================================================
def _params_hash(params: dict) -> tuple:
    return (params['dataset_key'], params['regimen'], round(params['euribor'], 4),
            round(params['ipc'], 4), round(params['paro'], 4), params['shock_vt'],
            params['shock_ano'], params.get('n_sim', 1000))

@st.cache_data
def simular_cached(params_hash: tuple, _modelo: dict):
    (dataset_key, regimen, euribor, ipc, paro, shock_vt, shock_ano, n_sim) = params_hash
    ds = DATASETS[dataset_key]
    pv_base  = ds['venta'][:, -1].copy()
    pa_base  = ds['alquiler'][:, -1].copy()
    ano_base = ds['ano_fin']
    anos_proy_loc = np.arange(ano_base + 1, 2033)

    mtm = construir_mtm_macro(euribor, ipc, paro)
    
    lambda_var = 0.15
    A_var      = _modelo['A_var']
    L_base     = _modelo['L']
    corr_base  = _modelo['corr']
    vol_base   = _modelo['vol_cal']
    betas_vta  = _modelo['betas_vta']
    betas_alq  = _modelo['betas_alq']
    reg_p      = _modelo['reg_params']
    
    n_anos  = len(anos_proy_loc)
    np.random.seed(42)
    sim_vta = np.zeros((n_sim, n_distritos, n_anos))
    sim_alq = np.zeros((n_sim, n_distritos, n_anos))

    for sim in range(n_sim):
        p_vta  = pv_base.copy()
        p_alq  = pa_base.copy()
        estado = 0
        r_prev_v = _modelo['mu_hist_v'].copy() # Inicialización T0

        for i, ano in enumerate(anos_proy_loc):
            # Transición Markov
            rand, cum = np.random.random(), 0.0
            for j in range(3):
                cum += mtm[estado, j]
                if rand < cum:
                    estado = j; break

            # 1. Presión Macro Base
            M_t = calcular_M_t(euribor, ipc, paro)
            
            # Ajuste alquiler para M_t (históricamente euribor desplaza demanda a alquiler)
            M_t_a = M_t + np.clip((euribor - 2.5)/100.0 * 0.4, -0.01, 0.02)

            # 2. Función Phi (Asimetría en Crisis) y Theta Volatilidad
            if estado == 1:    # Boom
                phi_v = M_t * reg_p['kappa_boom'] + reg_p['C_boom']
                phi_a = M_t_a * reg_p['kappa_boom'] + reg_p['C_boom'] * 0.8
                theta_vol = 0.85
            elif estado == 2:  # Crisis
                phi_v = np.minimum(M_t, 0) * reg_p['kappa_crisis'] + np.maximum(M_t, 0) * 1.0 - reg_p['C_crisis']
                phi_a = np.minimum(M_t_a, 0) * reg_p['kappa_crisis'] + np.maximum(M_t_a, 0) * 1.0 - reg_p['C_crisis'] * 0.8
                theta_vol = 1.60
            else:              # Normal
                phi_v = M_t
                phi_a = M_t_a
                theta_vol = 1.0

            # 3. Inercia Endógena
            var_comp_v = lambda_var * (A_var @ r_prev_v)
            var_comp_a = var_comp_v * 0.7
            
            # 4. Cálculo Determinista Separado
            tasa_v = (_modelo['mu_hist_v'] + var_comp_v) + (betas_vta * phi_v)
            tasa_a = (_modelo['mu_hist_a'] + var_comp_a) + (betas_alq * phi_a)

            # 5. Volatilidad Estocástica (DCC)
            vol_dyn = vol_base * theta_vol
            if estado == 2:
                cd = corr_base * 0.5 + np.ones_like(corr_base) * 0.5
                np.fill_diagonal(cd, 1.0)
                try:
                    ev, evec = np.linalg.eigh(cd)
                    ev = np.maximum(ev, 1e-8)
                    cd = evec @ np.diag(ev) @ evec.T
                    dd = np.sqrt(np.diag(cd))
                    cd = cd / np.outer(dd, dd)
                    L_dyn = np.linalg.cholesky(cd)
                except Exception: L_dyn = L_base
            elif estado == 1:
                cd = corr_base * 0.8 + np.eye(n_distritos) * 0.2
                try: L_dyn = np.linalg.cholesky(cd)
                except Exception: L_dyn = L_base
            else: L_dyn = L_base

            Z = np.random.normal(0, 1, n_distritos)
            eps = L_dyn @ Z
            tasa_v += vol_dyn * eps
            tasa_a += vol_dyn * 0.75 * eps
            tasa_v = np.maximum(tasa_v, -0.07)

            # Floor yield 2.5%
            yld = (p_alq * 12) / p_vta
            resist = np.clip(yld / 0.025, 0.5, 1.0)
            tasa_v = np.where(tasa_v > 0, tasa_v * resist, tasa_v)

            # Feedback alquiler -> venta
            if i > 0:
                pa_prev = sim_alq[sim, :, i - 1]
                pa_prev2 = pa_base if i == 1 else sim_alq[sim, :, i - 2]
                safe2 = np.where(pa_prev2 > 0, pa_prev2, 1.0)
                tasa_v += rho_vta_alq * ((pa_prev / safe2) - 1)

            # Shock regulatorio VT
            shk_v = shk_a = 0.0
            if shock_vt and ano == shock_ano:
                factor = concentracion_vt_real / 0.46
                shk_v = -0.10 * factor
                shk_a = -0.06 * factor

            p_vta = p_vta * (1 + tasa_v) * (1 + shk_v)
            p_alq = p_alq * (1 + tasa_a) * (1 + shk_a)

            sim_vta[sim, :, i] = p_vta
            sim_alq[sim, :, i] = p_alq

            # 6. Actualización r_prev (Tasa realizada con clip)
            r_prev_v = np.clip(tasa_v.copy(), -0.08, 0.08)

    return sim_vta, sim_alq, anos_proy_loc

def simular(params: dict, modelo: dict, n_sim: int = 1000):
    h = _params_hash({**params, 'n_sim': n_sim})
    return simular_cached(h, modelo)

# ============================================================================
# 5. UI Y CONTROLES (STREAMLIT)
# ============================================================================
if 'ds_key' not in st.session_state: st.session_state.ds_key = 'Datos Portales Inmobiliarios'
if 'regimen' not in st.session_state: st.session_state.regimen = 'Crecimiento'
if 'idx_distrito' not in st.session_state: st.session_state.idx_distrito = 1

def reset_macro(): st.session_state.regimen = st.session_state.regimen_selector

MACRO_DEFAULTS = {
    "Crecimiento": {"euribor": 1.5, "ipc": 2.5, "paro": 9.0},
    "Estanflacion": {"euribor": 3.8, "ipc": 5.5, "paro": 13.0},
    "Recesion (Estructural)": {"euribor": 2.5, "ipc": 1.0, "paro": 20.0},
}

with st.sidebar:
    st.header("Configuracion V40")
    ds_key = st.radio("Set de datos (Estado Inicial):", list(DATASETS.keys()),
                      index=list(DATASETS.keys()).index(st.session_state.ds_key))
    st.session_state.ds_key = ds_key
    ds_activo = DATASETS[ds_key]
    st.caption(f"Estructurales: Siempre Ayuntamiento | Inercia: {ds_key.split(' ')[1]}")
    st.markdown("---")
    
    reg_sel = st.radio("Regimen Economico:", ("Crecimiento", "Estanflacion", "Recesion (Estructural)"),
                       key="regimen_selector", on_change=reset_macro)
    st.markdown("---")
    defaults = MACRO_DEFAULTS.get(reg_sel, MACRO_DEFAULTS["Crecimiento"])
    euribor_val = st.slider("Euribor 12m (%)", 0.0, 6.0, float(defaults['euribor']), 0.1)
    ipc_val     = st.slider("IPC anual (%)",   -1.0, 10.0, float(defaults['ipc']), 0.1)
    paro_val    = st.slider("Tasa paro BCN (%)", 4.0, 30.0, float(defaults['paro']), 0.5)

    probs = calcular_probabilidades_regimen(euribor_val, ipc_val, paro_val)
    st.markdown("**Probabilidades MTM:**")
    cp1, cp2, cp3 = st.columns(3)
    cp1.metric("Normal", f"{probs[0]*100:.0f}%")
    cp2.metric("Boom",   f"{probs[1]*100:.0f}%")
    cp3.metric("Crisis", f"{probs[2]*100:.0f}%")
    st.markdown("---")
    
    shock_vt  = st.checkbox("Shock Regulatorio VT", value=False)
    shock_ano = st.radio("Año del shock:", [2028, 2029], index=1, horizontal=True, disabled=not shock_vt)
    n_sim_ui = st.select_slider("Simulaciones:", options=[200, 500, 1000, 2000], value=1000)

    params_act = {
        'dataset_key': ds_key, 'regimen': reg_sel, 'euribor': euribor_val,
        'ipc': ipc_val, 'paro': paro_val, 'shock_vt': shock_vt,
        'shock_ano': shock_ano, 'n_sim': n_sim_ui,
    }

with st.spinner("Calibrando modelo estructural V40 (OLS + Betas Independientes)..."):
    modelo_act = inicializar_modelo_v40(ds_key)
    ds_key_alt = [k for k in DATASETS.keys() if k != ds_key][0]
    modelo_alt = inicializar_modelo_v40(ds_key_alt)

with st.spinner(f"Simulando {n_sim_ui} trayectorias SDE..."):
    sv_act, sa_act, anos_proy_act = simular(params_act, modelo_act, n_sim_ui)

# ============================================================================
# 6. DASHBOARD & VISUALIZACIÓN CON PLOTLY
# ============================================================================
st.title("Barcelona Strategic Model v40.0")
st.markdown(f"**SDE Híbrido | Betas Endógenas | OLS Regímenes | {ds_activo['fecha_dato']} | Inercia: {modelo_act['metodo_var']}**")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Regimen", reg_sel.replace(" (Estructural)",""))
c2.metric("Euribor", f"{euribor_val:.1f}%")
c3.metric("IPC", f"{ipc_val:.1f}%")
c4.metric("Paro", f"{paro_val:.1f}%")
c5.metric("Shock VT", f"{shock_ano}" if shock_vt else "OFF")
c6.metric("n_sim", str(n_sim_ui))

tab1, tab2, tab3 = st.tabs(["Analisis Interactivo", "Riesgo/Retorno", "Calibracion V40 (Set B)"])

with tab1:
    sel_dist = st.selectbox("Distrito:", distritos, index=st.session_state.idx_distrito)
    idx = distritos.index(sel_dist)
    st.session_state.idx_distrito = idx

    anos_plot = np.concatenate([[ds_activo['ano_fin']], anos_proy_act])
    last_v = ds_activo['venta'][idx, -1]
    p50_v = np.percentile(sv_act[:, idx, :], 50, axis=0)
    p10_v = np.percentile(sv_act[:, idx, :], 10, axis=0)
    p90_v = np.percentile(sv_act[:, idx, :], 90, axis=0)
    
    fig = go.Figure()
    # Historia
    fig.add_trace(go.Scatter(x=anos_macro, y=modelo_act['reconstruidos'][idx], mode='lines+markers', name='Hist/Rec.', line=dict(color='#2c3e50')))
    # IC 80%
    fig.add_trace(go.Scatter(x=anos_plot, y=np.concatenate([[last_v], p90_v]), mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=anos_plot, y=np.concatenate([[last_v], p10_v]), mode='lines', fill='tonexty', fillcolor='rgba(231, 76, 60, 0.2)', line=dict(width=0), name='IC P10-P90'))
    # P50
    fig.add_trace(go.Scatter(x=anos_plot, y=np.concatenate([[last_v], p50_v]), mode='lines', name='P50 Venta', line=dict(color='#e74c3c', width=3, dash='dash')))
    fig.update_layout(title=f"Proyección Venta EUR/m2 - {sel_dist}", xaxis_title="Año", yaxis_title="EUR/m2", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    v_fin = np.median(sv_act[:, idx, -1]); a_fin = np.median(sa_act[:, idx, -1])
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Venta 2032", f"{int(v_fin):,} EUR", f"{((v_fin/last_v)-1)*100:.1f}%")
    k2.metric("Alquiler 2032", f"{a_fin:.1f} EUR", f"{((a_fin/ds_activo['alquiler'][idx, -1])-1)*100:.1f}%")
    k3.metric("Yield 2032", f"{(a_fin*12)/v_fin*100:.1f}%")
    k4.metric("Beta Venta (Set B)", f"{modelo_act['betas_vta'][idx]:.3f}")
    k5.metric("Beta Alquiler (Set B)", f"{modelo_act['betas_alq'][idx]:.3f}")

with tab2:
    pv0 = ds_activo['venta'][:, -1]
    n_yr = 2032 - ds_activo['ano_fin']
    cagrs = [((sv_act[:, i, -1].mean() / pv0[i]) ** (1/n_yr) - 1) * 100 for i in range(n_distritos)]
    vols  = [(sv_act[:, i, -1].std() / sv_act[:, i, -1].mean()) * 100 for i in range(n_distritos)]
    df_r = pd.DataFrame({'Distrito': distritos, 'CAGR (%)': cagrs, 'CV (%)': vols})
    fig_r = px.scatter(df_r, x='CV (%)', y='CAGR (%)', text='Distrito', color='CAGR (%)', color_continuous_scale='RdYlGn', size=[20]*10, title="Riesgo vs Retorno")
    fig_r.update_traces(textposition='top center')
    st.plotly_chart(fig_r, use_container_width=True)

with tab3:
    st.subheader("Parámetros Estructurales Extraídos del Set B (Ayuntamiento)")
    df_cal = pd.DataFrame({
        'Distrito': distritos,
        'Beta Venta (vs IPV)': modelo_act['betas_vta'].round(3),
        'Beta Alq (vs Sintético)': modelo_act['betas_alq'].round(3),
        'Volatilidad Base %': (modelo_act['vol_cal']*100).round(2),
        'Drift Venta (Local) %/a': (modelo_act['mu_hist_v']*100).round(2),
    })
    st.dataframe(df_cal.set_index('Distrito'), use_container_width=True)
    
    c_reg1, c_reg2 = st.columns(2)
    c_reg1.markdown("**Regresiones OLS (Boom 2015-2019)**")
    c_reg1.code(f"kappa_boom: {modelo_act['reg_params']['kappa_boom']:.4f}\nC_boom:     {modelo_act['reg_params']['C_boom']:.4f}")
    c_reg2.markdown("**Regresiones OLS (Crisis 13-14, 20)**")
    c_reg2.code(f"kappa_crisis: {modelo_act['reg_params']['kappa_crisis']:.4f}\nC_crisis:     {modelo_act['reg_params']['C_crisis']:.4f}")

    fig_corr = px.imshow(modelo_act['corr'], x=distritos, y=distritos, color_continuous_scale='RdYlGn', zmin=-1, zmax=1, title="Correlación Espacial (Cholesky Set B)")
    st.plotly_chart(fig_corr, use_container_width=True)

# ============================================================================
# 7. GENERACIÓN PDF (V40 METHODOLOGY)
# ============================================================================
def clean_str(txt: str) -> str:
    repls = {'€':'EUR','á':'a','é':'e','í':'i','ó':'o','ú':'u','ñ':'n','²':'2'}
    for k,v in repls.items(): txt = txt.replace(k, v)
    return txt.encode('latin-1', errors='replace').decode('latin-1')

def generate_metodologia_v40(params, modelo) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Times','B',16)
    pdf.multi_cell(0,9,clean_str("EXPLICACION METODOLOGICA DEL MODELO V40.0\nARQUITECTURA HIBRIDA Y BETAS ENDOGENAS"), align='C')
    pdf.ln(5)
    
    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("1. LA NUEVA ECUACION GOBERNADORA (SDE)"),0,1)
    pdf.set_font('Times','',10)
    pdf.multi_cell(0,5.5,clean_str(
        "El modelo V40 implementa una separacion estricta en bloques estancos. La inercia "
        "(tendencia y autorregresion) se calibra sobre el set de datos seleccionado, mientras que los "
        "parametros estructurales (Betas, Volatilidad, Cholesky) emanan siempre del Set B (Ayuntamiento) "
        "por su profundidad historica.\n"
        "E[dP/P] = (mu_hist + lambda * VAR) + Beta * Phi(S_t, M_t) + (sigma * theta * L * Z) + Feedback"
    ))
    pdf.ln(3)

    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("2. SHOCK DE REGIMEN ASIMETRICO Y OLS"),0,1)
    pdf.set_font('Times','',10)
    pdf.multi_cell(0,5.5,clean_str(
        "La Beta ya no multiplica la tendencia natural. Modula exclusivamente la funcion macro Phi, "
        "la cual se asimetriza en crisis para evitar rebotes patologicos positivos si M_t > 0:\n"
        "Phi(Crisis) = min(M_t, 0) * kappa_crisis + max(M_t, 0) * 1.0 - C_crisis\n"
        "Los parametros kappa y C se extraen por Regresion OLS empirica sobre los anos reales de Boom "
        "(2015-2019) y Crisis (2013, 2014, 2020) del Set B, evitando la circularidad de datos CAPM."
    ))
    pdf.ln(3)

    pdf.set_font('Times','B',12)
    pdf.cell(0,8,clean_str("3. BETAS INDEPENDIENTES Y SINTETICO DE ALQUILER"),0,1)
    pdf.set_font('Times','',10)
    pdf.multi_cell(0,5.5,clean_str(
        "Se elimina el coeficiente arbitrario de alquiler. Beta_vta se calcula contra el Indice IPV "
        "del INE, y Beta_alq se calcula cruzando los retornos distritales contra un Indice Sintetico "
        "de Alquiler (media simple de los 10 distritos del Set B). Ambas se truncan al intervalo "
        "de seguridad [0.5, 1.5] para proteger el motor Monte Carlo."
    ))
    return pdf.output(dest='S').encode('latin-1')

st.markdown("---")
if st.button("Descargar Metodología V40 (PDF)"):
    pdf_bytes = generate_metodologia_v40(params_act, modelo_act)
    st.download_button("Descargar", pdf_bytes, "BCN_Model_V40_Metodologia.pdf", "application/pdf")





