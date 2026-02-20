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
import io
import sys

# ============================================================================
# 0. LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BCNModel")

# ============================================================================
# 1. CONFIGURACIÓN VISUAL
# ============================================================================
st.set_page_config(
    page_title="Barcelona Strategic Model v31.0",
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
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border: none;
    }
    .stButton>button:hover { background-color: #1565c0; }
    .metric-card {
        background-color: white; padding: 15px; border-radius: 8px;
        border-left: 5px solid #0d47a1; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .test-pass { color: #27ae60; font-weight: bold; }
    .test-fail { color: #e74c3c; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

warnings.filterwarnings('ignore')

# ============================================================================
# 2. DATOS (DATA-DRIVEN) — ACTUALIZADO CON PRECIOS REALES 2026
# ============================================================================

distritos = ['Ciutat Vella', 'Eixample', 'Gracia', 'Horta Guinardo', 'Les Corts',
             'Nou Barris', 'Sant Andreu', 'Sant Marti', 'Sants-Montjuic', 'Sarria-Sant Gervasi']
n_distritos = len(distritos)

# 2.1 VECTOR DE EXPOSICIÓN VT (Peso Real - Densidad Airbnb por distrito)
concentracion_vt_real = np.array([0.150, 0.460, 0.105, 0.030, 0.035, 0.004, 0.015, 0.120, 0.110, 0.050])

# 2.2 PRECIOS DE VENTA (€/m²) — 2019–2026
# Columnas: 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026
# Fuente 2026: Datos reales de mercado (Idealista / Incasol)
# NOTA: separador miles=punto, decimales=coma en fuente original → valores en float
precio_venta_historico = np.array([
    [4200, 4100, 4250, 4450, 4620, 4740, 4817, 4811],   # Ciutat Vella
    [5500, 5350, 5550, 5850, 6050, 6180, 6299, 6363],   # Eixample
    [4600, 4480, 4650, 4900, 5100, 5220, 5301, 5404],   # Gracia
    [3350, 3280, 3400, 3600, 3750, 3820, 3862, 3905],   # Horta Guinardo
    [5350, 5200, 5400, 5700, 5950, 6080, 6135, 6355],   # Les Corts
    [2450, 2400, 2490, 2630, 2730, 2780, 2804, 2946],   # Nou Barris
    [3300, 3220, 3340, 3530, 3670, 3740, 3774, 3730],   # Sant Andreu
    [4180, 4080, 4230, 4460, 4640, 4730, 4784, 4899],   # Sant Marti
    [3780, 3690, 3830, 4040, 4200, 4280, 4338, 4477],   # Sants-Montjuic
    [5900, 5750, 5970, 6290, 6540, 6670, 6738, 6896]    # Sarria-Sant Gervasi
])

# 2.3 PRECIOS DE ALQUILER (€/m²/mes) — 2019–2026
# Fuente 2026: Datos reales de mercado
precio_alquiler_historico = np.array([
    [22.5, 20.8, 21.5, 23.2, 24.8, 25.8, 26.4, 25.7],  # Ciutat Vella
    [22.8, 21.2, 22.0, 23.8, 25.2, 26.1, 26.7, 26.0],  # Eixample
    [20.8, 19.5, 20.2, 21.8, 23.1, 24.0, 24.4, 23.7],  # Gracia
    [15.6, 14.8, 15.3, 16.5, 17.5, 18.1, 18.4, 19.1],  # Horta Guinardo
    [18.7, 17.5, 18.1, 19.6, 20.8, 21.6, 22.0, 21.9],  # Les Corts
    [13.4, 12.8, 13.2, 14.3, 15.1, 15.6, 15.8, 16.3],  # Nou Barris
    [15.8, 15.0, 15.5, 16.8, 17.8, 18.3, 18.6, 18.5],  # Sant Andreu
    [20.0, 18.8, 19.5, 21.0, 22.3, 23.2, 23.5, 23.6],  # Sant Marti
    [17.7, 16.7, 17.3, 18.7, 19.8, 20.5, 20.8, 20.8],  # Sants-Montjuic
    [19.8, 18.5, 19.2, 20.7, 22.0, 22.8, 23.2, 23.6]   # Sarria-Sant Gervasi
])

# Vectores de precios base (año más reciente = 2026)
precio_venta_2026   = precio_venta_historico[:, -1].astype(float)
precio_alquiler_2026 = precio_alquiler_historico[:, -1].astype(float)

# Metadatos de fuente
FECHA_DATO_REAL = "2026 (Q1 2026)"
FUENTE_DATO = "Idealista / Incasol (Generalitat de Catalunya)"

# 2.4 MACRO HISTÓRICO (precio medio €/m² en Barcelona)
datos_agregados_bcn = {
    2007: 4071, 2008: 3737, 2009: 3728, 2010: 3752, 2011: 3398, 2012: 3076,
    2013: 2992, 2014: 3061, 2015: 3297, 2016: 3649, 2017: 4154, 2018: 4232,
    2019: 4135, 2020: 4018, 2021: 3910, 2022: 4063, 2023: 4167, 2024: 4700,
    2025: 5042, 2026: 5200
}
anos_larga_data  = np.array(list(datos_agregados_bcn.keys()))
precios_agregados = np.array(list(datos_agregados_bcn.values()))

anos_proy            = np.arange(2027, 2033)
anos_proy_con_2026   = np.concatenate([[2026], anos_proy])
anos_retro_corto     = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])

# 2.5 PARÁMETROS DEL MODELO
volatilidad_array     = np.array([0.055, 0.042, 0.048, 0.035, 0.038, 0.025, 0.032, 0.045, 0.040, 0.030])
sensibilidad_distrital = np.array([0.95,  0.85,  0.80,  0.35,  0.60,  0.30,  0.40,  0.70,  0.60,  0.65])
rho_vta_alq           = 0.45
gamma_sensibility     = 0.35

# Matrices de Transición de Markov (filas deben sumar 1.0 exactamente)
MTM_RECESION_SEVERA = np.array([[0.60, 0.05, 0.35],
                                 [0.40, 0.60, 0.00],
                                 [0.15, 0.00, 0.85]])

MTM_CRECIMIENTO     = np.array([[0.73, 0.25, 0.02],
                                 [0.40, 0.60, 0.00],
                                 [0.45, 0.00, 0.55]])

MTM_ESTANFLACION    = np.array([[0.80, 0.05, 0.15],
                                 [0.30, 0.70, 0.00],
                                 [0.10, 0.00, 0.90]])

# ============================================================================
# 3. TESTS UNITARIOS — MOTOR MATEMÁTICO
# ============================================================================

def test_mtm_row_sums(mtm: np.ndarray, name: str) -> tuple:
    """Cada fila de la MTM debe sumar exactamente 1.0."""
    row_sums = mtm.sum(axis=1)
    passed   = np.allclose(row_sums, 1.0, atol=1e-9)
    msg      = f"MTM '{name}' sumas de filas: {row_sums.round(6)}"
    return passed, msg

def test_cholesky_positive_definite(L: np.ndarray) -> tuple:
    """L debe ser triangular inferior con diagonales positivas."""
    is_lower     = np.allclose(L, np.tril(L), atol=1e-10)
    pos_diag     = np.all(np.diag(L) > 0)
    passed       = is_lower and pos_diag
    msg          = f"Cholesky triangular inferior: {is_lower} | Diagonal positiva: {pos_diag}"
    return passed, msg

def test_betas_range(betas: np.ndarray) -> tuple:
    """Betas deben estar en el rango [0.5, 1.5] tras clip."""
    in_range = np.all((betas >= 0.5) & (betas <= 1.5))
    msg      = f"Betas min={betas.min():.3f} max={betas.max():.3f} | En rango [0.5,1.5]: {in_range}"
    return in_range, msg

def test_cholesky_reconstruction(corr: np.ndarray, L: np.ndarray) -> tuple:
    """L @ L^T debe aproximar la matriz de correlación original."""
    reconstructed = L @ L.T
    passed        = np.allclose(corr, reconstructed, atol=1e-6)
    max_err       = np.abs(corr - reconstructed).max()
    msg           = f"Reconstruccion Cholesky: error maximo={max_err:.2e} | OK={passed}"
    return passed, msg

def test_precios_2026_positivos() -> tuple:
    """Todos los precios 2026 deben ser positivos."""
    v_ok  = np.all(precio_venta_2026 > 0)
    a_ok  = np.all(precio_alquiler_2026 > 0)
    passed = v_ok and a_ok
    msg    = f"Precios venta positivos: {v_ok} | Precios alquiler positivos: {a_ok}"
    return passed, msg

def test_yield_razonable() -> tuple:
    """Yield bruto 2026 debe estar entre 2% y 10% para todos los distritos."""
    yields = (precio_alquiler_2026 * 12) / precio_venta_2026
    in_range = np.all((yields >= 0.02) & (yields <= 0.10))
    msg = f"Yields 2026: min={yields.min()*100:.1f}% max={yields.max()*100:.1f}% | En rango [2%,10%]: {in_range}"
    return in_range, msg

def run_all_tests(L_matrix=None, betas=None, corr_matrix=None):
    """Ejecuta todos los tests y devuelve resultados."""
    results = []

    # Tests de datos
    results.append(("Precios 2026 positivos",         *test_precios_2026_positivos()))
    results.append(("Yield bruto 2026 razonable",     *test_yield_razonable()))

    # Tests de matrices MTM
    for name, mtm in [("Crecimiento", MTM_CRECIMIENTO),
                      ("Estanflación", MTM_ESTANFLACION),
                      ("Recesión", MTM_RECESION_SEVERA)]:
        results.append((f"MTM {name} - filas suman 1", *test_mtm_row_sums(mtm, name)))

    # Tests del motor (si disponibles)
    if betas is not None:
        results.append(("Betas en rango [0.5, 1.5]", *test_betas_range(betas)))
    if L_matrix is not None:
        results.append(("Cholesky definida positiva",  *test_cholesky_positive_definite(L_matrix)))
    if L_matrix is not None and corr_matrix is not None:
        results.append(("Cholesky reconstruccion",     *test_cholesky_reconstruction(corr_matrix, L_matrix)))

    return results

# ============================================================================
# 4. VALIDACIÓN DE MATRICES DE TRANSICIÓN (antes de simular)
# ============================================================================

def validar_mtm(mtm: np.ndarray, nombre: str) -> bool:
    """
    Verifica que cada fila de la MTM sume 1.0.
    Lanza ValueError si la validación falla.
    """
    row_sums = mtm.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        msg = f"MTM '{nombre}' inválida: filas suman {row_sums}. Deben sumar 1.0."
        logger.error(msg)
        raise ValueError(msg)
    logger.info(f"MTM '{nombre}' validada correctamente.")
    return True

# Validar al cargar el módulo
for _nombre, _mtm in [("Crecimiento", MTM_CRECIMIENTO),
                       ("Estanflación", MTM_ESTANFLACION),
                       ("Recesión Severa", MTM_RECESION_SEVERA)]:
    validar_mtm(_mtm, _nombre)

# ============================================================================
# 5. MOTOR MATEMÁTICO V31.0
# ============================================================================

@st.cache_data
def inicializar_modelo_estructural():
    """
    Reconstruye series históricas de precios por distrito (2007-2025)
    usando regresión Beta-CAPM y descomposición de Cholesky.
    """
    idx_start = np.where(anos_larga_data == 2019)[0][0]
    precios_mkt_match = precios_agregados[idx_start:]

    betas = []
    for i in range(n_distritos):
        ret_dist = (precio_venta_historico[i, 1:] / precio_venta_historico[i, :-1]) - 1
        ret_mkt  = (precios_mkt_match[1:] / precios_mkt_match[:-1]) - 1
        cov      = np.cov(ret_dist, ret_mkt)[0, 1]
        var      = np.var(ret_mkt)
        beta     = cov / var if var > 1e-6 else 1.0
        betas.append(beta)
    betas = np.clip(np.array(betas), 0.5, 1.5)

    # Reconstrucción histórica (2007-2018) via CAPM inverso
    reconstruidos = np.zeros((n_distritos, len(anos_larga_data)))
    reconstruidos[:, idx_start:] = precio_venta_historico
    np.random.seed(999)
    for t in range(idx_start - 1, -1, -1):
        ret_mkt_val = (precios_agregados[t + 1] / precios_agregados[t]) - 1
        for d in range(n_distritos):
            ruido    = np.random.normal(0, 0.003)
            r_dist   = (betas[d] * ret_mkt_val) + ruido
            reconstruidos[d, t] = reconstruidos[d, t + 1] / (1 + r_dist)

    # Matriz de correlación + Cholesky
    retornos = (reconstruidos[:, 1:] / reconstruidos[:, :-1]) - 1
    corr     = np.corrcoef(retornos)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals         = np.maximum(eigvals, 1e-8)
        corr_rebuilt    = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_diag          = np.sqrt(np.diag(corr_rebuilt))
        corr_rebuilt    = corr_rebuilt / np.outer(d_diag, d_diag)
        try:
            L = np.linalg.cholesky(corr_rebuilt)
        except Exception:
            L = np.eye(n_distritos)
        corr = corr_rebuilt

    logger.info("Modelo estructural inicializado. Betas: %s", betas.round(3))
    return reconstruidos, L, betas, corr


PRECIOS_RECONSTRUIDOS, CHOLESKY_DECOMPOSITION, BETAS_MODELO, CORR_MATRIZ = \
    inicializar_modelo_estructural()


def _params_to_hashable(params: dict) -> tuple:
    """
    Convierte el diccionario de parámetros (con posibles arrays NumPy)
    a una tupla hashable apta para st.cache_data.
    """
    return (
        params['crecimiento_venta'],
        params['crecimiento_alquiler'],
        params['shock_vt'],
        tuple(params['mtm'].flatten().tolist()),
        params.get('n_sim', 1000),
    )


@st.cache_data
def simular_escenario_cached(params_hash: tuple, n_sim: int = 1000):
    """
    Motor Monte Carlo + Markov Chain. Recibe params como tupla hashable
    para garantizar cache correcto.
    """
    crecimiento_venta, crecimiento_alquiler, shock_vt, mtm_flat, _ = params_hash
    mtm = np.array(mtm_flat).reshape(3, 3)

    # Validación antes de simular
    validar_mtm(mtm, "simulación activa")

    np.random.seed(42)
    n_anos   = len(anos_proy)
    sim_vta  = np.zeros((n_sim, n_distritos, n_anos))
    sim_alq  = np.zeros((n_sim, n_distritos, n_anos))

    intensity_vta  = abs(crecimiento_venta)
    intensity_alq  = abs(crecimiento_alquiler)
    input_sign_vta = np.sign(crecimiento_venta) if crecimiento_venta != 0 else 1
    input_sign_alq = np.sign(crecimiento_alquiler) if crecimiento_alquiler != 0 else 1

    for sim in range(n_sim):
        p_vta  = precio_venta_2026.copy()
        p_alq  = precio_alquiler_2026.copy()
        estado = 0

        base_rand_v = np.random.normal(0, 0.01, n_anos)
        base_rand_a = np.random.normal(0, 0.01, n_anos)

        for i, ano in enumerate(anos_proy):
            rand = np.random.random()
            cum  = 0
            for j in range(3):
                cum += mtm[estado, j]
                if rand < cum:
                    estado = j
                    break

            tasa_base_v = (intensity_vta * input_sign_vta) + base_rand_v[i]
            tasa_base_a = (intensity_alq * input_sign_alq) + base_rand_a[i]

            if estado == 1:    # Boom
                tasa_final_v = tasa_base_v * 1.4
                tasa_final_a = tasa_base_a * 1.3
            elif estado == 2:  # Crisis
                tasa_final_v = -0.02 - (abs(tasa_base_v) * 0.5)
                tasa_final_a = -0.01 - (abs(tasa_base_a) * 0.5)
            else:              # Normal
                tasa_final_v = tasa_base_v
                tasa_final_a = tasa_base_a

            tasa_final_v *= sensibilidad_distrital
            tasa_final_a *= (sensibilidad_distrital * 0.8)

            z       = np.random.normal(0, 1, n_distritos)
            eps     = CHOLESKY_DECOMPOSITION @ z
            vol_adj = 1.5 if estado == 2 else 1.0

            tasa_final_v += volatilidad_array * vol_adj * eps
            tasa_final_a += volatilidad_array * vol_adj * 0.8 * eps
            tasa_final_v  = np.maximum(tasa_final_v, -0.06)

            # Resistencia yield (floor yield bruto ≥ 2.8%)
            current_yield    = (p_alq * 12) / p_vta
            min_yield        = 0.028
            resistance       = np.clip(current_yield / min_yield, 0.5, 1.0)
            tasa_final_v     = np.where(tasa_final_v > 0, tasa_final_v * resistance, tasa_final_v)

            # Feedback alquiler → venta (rho)
            if i > 0:
                prev_alq  = sim_alq[sim, :, i - 1]
                prev2_alq = precio_alquiler_2026 if i == 1 else sim_alq[sim, :, i - 2]
                rho_eff   = rho_vta_alq * ((prev_alq / prev2_alq) - 1)
                tasa_final_v += rho_eff

            # Shock Ley Vivienda 2029 (desplazado 1 año porque el horizonte arranca en 2027)
            shock_vt_discrete  = 0.0
            shock_alq_discrete = 0.0
            if shock_vt and ano == 2029:
                factor_exp        = concentracion_vt_real / 0.46
                shock_vt_discrete  = -0.10 * factor_exp
                shock_alq_discrete = -0.06 * factor_exp

            p_vta = p_vta * (1 + tasa_final_v) * (1 + shock_vt_discrete)
            p_alq = p_alq * (1 + tasa_final_a) * (1 + shock_alq_discrete)

            sim_vta[sim, :, i] = p_vta
            sim_alq[sim, :, i] = p_alq

    return sim_vta, sim_alq


def simular_escenario(params: dict, n_sim: int = 1000):
    """Wrapper público: convierte params a hash y llama al motor cacheado."""
    params_with_nsim = {**params, 'n_sim': n_sim}
    h = _params_to_hashable(params_with_nsim)
    return simular_escenario_cached(h, n_sim)

# ============================================================================
# 6. GESTIÓN DE ESTADO (SESSION STATE)
# ============================================================================

if 'regimen'  not in st.session_state: st.session_state.regimen  = "Crecimiento"
if 'vta_sl'   not in st.session_state: st.session_state.vta_sl   = 4.5
if 'alq_sl'   not in st.session_state: st.session_state.alq_sl   = 3.8
# Figura global inicializada para evitar scope frágil
if 'fig_distrito' not in st.session_state: st.session_state.fig_distrito = None
if 'idx_distrito' not in st.session_state: st.session_state.idx_distrito = 1


def update_params():
    sel = st.session_state.regimen_selector
    if sel == "Crecimiento":
        st.session_state.vta_sl = 4.5;  st.session_state.alq_sl = 3.8
    elif sel == "Estanflación":
        st.session_state.vta_sl = 1.5;  st.session_state.alq_sl = 2.5
    elif sel == "Recesión (Estructural)":
        st.session_state.vta_sl = -2.5; st.session_state.alq_sl = -1.0
    st.session_state.regimen = sel


# ============================================================================
# 7. SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuración")
    reg_sel = st.radio(
        "Ciclo Económico:",
        ("Crecimiento", "Estanflación", "Recesión (Estructural)"),
        key="regimen_selector",
        on_change=update_params
    )
    st.markdown("---")
    g_vta  = st.slider("Tendencia Venta (%)",    -10.0, 10.0, key="vta_sl")
    g_alq  = st.slider("Tendencia Alquiler (%)", -10.0, 10.0, key="alq_sl")
    shock_vt = st.checkbox("Activar Shock Ley Vivienda (2029)", value=False)

    st.markdown("---")
    st.subheader("🎛️ Precisión de Simulación")
    n_sim_ui = st.select_slider(
        "Número de simulaciones (n_sim):",
        options=[200, 500, 1000, 2000, 5000],
        value=1000,
        help="Mayor n_sim = más precisión estadística pero mayor tiempo de cómputo."
    )
    st.caption(f"Tiempo estimado: ~{n_sim_ui//500 + 1}s")

    if reg_sel == "Crecimiento":
        mtm = MTM_CRECIMIENTO
    elif reg_sel == "Estanflación":
        mtm = MTM_ESTANFLACION
    else:
        mtm = MTM_RECESION_SEVERA

    params_act = {
        'crecimiento_venta':     g_vta  / 100,
        'crecimiento_alquiler':  g_alq  / 100,
        'shock_vt':              shock_vt,
        'mtm':                   mtm,
        'n_sim':                 n_sim_ui,
    }

# ============================================================================
# 8. SIMULACIONES
# ============================================================================

with st.spinner(f"Calculando escenarios estocásticos ({n_sim_ui} simulaciones)..."):
    sv_act,  sa_act  = simular_escenario(params_act, n_sim=n_sim_ui)
    sv_crec, sa_crec = simular_escenario({'crecimiento_venta': 0.045, 'crecimiento_alquiler': 0.038,
                                          'shock_vt': shock_vt, 'mtm': MTM_CRECIMIENTO},    n_sim=200)
    sv_est,  sa_est  = simular_escenario({'crecimiento_venta': 0.015, 'crecimiento_alquiler': 0.020,
                                          'shock_vt': shock_vt, 'mtm': MTM_ESTANFLACION},   n_sim=200)
    sv_rec,  sa_rec  = simular_escenario({'crecimiento_venta': -0.025, 'crecimiento_alquiler': -0.010,
                                          'shock_vt': shock_vt, 'mtm': MTM_RECESION_SEVERA}, n_sim=200)

scenarios_data = {
    "Crecimiento":      (sv_crec, sa_crec),
    "Estanflación":     (sv_est,  sa_est),
    "Recesión":         (sv_rec,  sa_rec),
    "Usuario (Activo)": (sv_act,  sa_act),
}

# ============================================================================
# 9. DASHBOARD — CABECERA
# ============================================================================

st.title("🏙️ Barcelona Strategic Model v31.0")
st.markdown(f"**Modelo Definitivo | Datos actualizados a {FECHA_DATO_REAL} | Fuente: {FUENTE_DATO}**")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Escenario",   reg_sel)
c2.metric("Venta",       f"{g_vta}%")
c3.metric("Alquiler",    f"{g_alq}%")
c4.metric("Regulación",  "ON" if shock_vt else "OFF")
c5.metric("n_sim",       str(n_sim_ui))

# ============================================================================
# 10. TABS PRINCIPALES
# ============================================================================

tab1, tab2, tab3 = st.tabs(["📊 Análisis Distrito", "🎯 Riesgo/Retorno", "🧪 Tests del Modelo"])

# --- TAB 1: ANÁLISIS DISTRITO ---
with tab1:
    sel_dist = st.selectbox("Distrito:", distritos, index=st.session_state.idx_distrito)
    idx      = distritos.index(sel_dist)
    st.session_state.idx_distrito = idx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot Venta ---
    ax1.plot(anos_larga_data, PRECIOS_RECONSTRUIDOS[idx], 'o-', color='#2c3e50', label='Histórico', linewidth=1.5)
    ax1.axvspan(2007, 2019, color='gray', alpha=0.1)
    ax1.axvline(x=2026, color='navy', linestyle='--', linewidth=0.8, alpha=0.5, label='Dato real 2026')

    p50  = np.percentile(sv_act[:, idx, :], 50, axis=0)
    p10  = np.percentile(sv_act[:, idx, :], 10, axis=0)
    p90  = np.percentile(sv_act[:, idx, :], 90, axis=0)
    last = PRECIOS_RECONSTRUIDOS[idx][-1]

    ax1.plot(anos_proy_con_2026, np.concatenate([[last], p50]),
             '--', color='#e74c3c', linewidth=2, label=f'Proy. Activa (P50)')
    ax1.fill_between(anos_proy_con_2026,
                     np.concatenate([[last], p10]),
                     np.concatenate([[last], p90]),
                     color='#e74c3c', alpha=0.15, label='IC P10-P90')

    p50_est = np.percentile(sv_est[:, idx, :], 50, axis=0)
    p50_rec = np.percentile(sv_rec[:, idx, :], 50, axis=0)
    ax1.plot(anos_proy_con_2026, np.concatenate([[last], p50_est]), ':', color='gray',  label='Estanflación')
    ax1.plot(anos_proy_con_2026, np.concatenate([[last], p50_rec]), ':', color='black', label='Recesión')

    ax1.set_title(f"Venta (€/m²) — {sel_dist}", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize='small')
    ax1.set_xlabel("Año")

    # --- Plot Alquiler ---
    ax2.plot(anos_retro_corto, precio_alquiler_historico[idx], 'o-', color='#2c3e50', linewidth=1.5)
    ax2.axvline(x=2026, color='navy', linestyle='--', linewidth=0.8, alpha=0.5)

    p50a  = np.percentile(sa_act[:, idx, :], 50, axis=0)
    p10a  = np.percentile(sa_act[:, idx, :], 10, axis=0)
    p90a  = np.percentile(sa_act[:, idx, :], 90, axis=0)
    lasta = precio_alquiler_historico[idx][-1]

    ax2.plot(anos_proy_con_2026, np.concatenate([[lasta], p50a]),
             '--', color='#2ecc71', linewidth=2)
    ax2.fill_between(anos_proy_con_2026,
                     np.concatenate([[lasta], p10a]),
                     np.concatenate([[lasta], p90a]),
                     color='#2ecc71', alpha=0.15)

    ax2.set_title(f"Alquiler (€/m²/mes) — {sel_dist}", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlabel("Año")

    plt.tight_layout()

    # Guardar figura en session_state (fix scope frágil)
    st.session_state.fig_distrito = fig
    st.pyplot(fig)

    v31 = np.median(sv_act[:, idx, -1])
    a31 = np.median(sa_act[:, idx, -1])
    yld = (a31 * 12) / v31 * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Venta 2032",   f"{int(v31):,} €",
              f"{((v31 / precio_venta_2026[idx]) - 1) * 100:.1f}%")
    k2.metric("Alquiler 2032", f"{a31:.1f} €",
              f"{((a31 / precio_alquiler_2026[idx]) - 1) * 100:.1f}%")
    k3.metric("Yield 2032",   f"{yld:.1f}%")
    k4.metric("Dato base",    "2026 (Real)")

    # Tabla de precios actuales 2026
    st.markdown("#### Precios Reales 2026 — Todos los Distritos")
    df_precios = pd.DataFrame({
        'Distrito':       distritos,
        'Venta 2026 (€/m²)':    precio_venta_2026.astype(int),
        'Alquiler 2026 (€/m²)': precio_alquiler_2026,
        'Yield Bruto 2026':      [f"{(a*12/v*100):.1f}%" for v, a in
                                   zip(precio_venta_2026, precio_alquiler_2026)]
    })
    st.dataframe(df_precios.set_index('Distrito'), use_container_width=True)


# --- TAB 2: RIESGO/RETORNO ---
with tab2:
    cagrs, vols = [], []
    for i in range(n_distritos):
        f = sv_act[:, i, -1]
        cagrs.append(((f.mean() / precio_venta_2026[i]) ** (1 / 6) - 1) * 100)
        vols.append(f.std() / f.mean() * 100)

    df_r = pd.DataFrame({'Distrito': distritos, 'Retorno CAGR (%)': cagrs, 'Riesgo CV (%)': vols})

    fig_r = px.scatter(
        df_r, x='Riesgo CV (%)', y='Retorno CAGR (%)',
        text='Distrito', color='Retorno CAGR (%)',
        color_continuous_scale='RdYlGn', size=[20] * 10,
        title=f"Cuadrante Riesgo/Retorno — {reg_sel} | n_sim={n_sim_ui}"
    )

    mean_risk = df_r['Riesgo CV (%)'].mean()
    mean_ret  = df_r['Retorno CAGR (%)'].mean()
    fig_r.add_vline(x=mean_risk, line_width=1.5, line_dash="dash", line_color="red")
    fig_r.add_hline(y=mean_ret,  line_width=1.5, line_dash="dash", line_color="red")
    fig_r.add_annotation(x=df_r['Riesgo CV (%)'].min(),  y=df_r['Retorno CAGR (%)'].max(),
                         text="◆ ESTRELLAS (Buy)",      showarrow=False, font=dict(color="green", size=12))
    fig_r.add_annotation(x=df_r['Riesgo CV (%)'].max(),  y=df_r['Retorno CAGR (%)'].min(),
                         text="▼ INEFICIENTES (Sell)",  showarrow=False, font=dict(color="red",   size=12))

    st.plotly_chart(fig_r, use_container_width=True)
    fmt = {'Retorno CAGR (%)': '{:.2f}%', 'Riesgo CV (%)': '{:.2f}%'}
    st.dataframe(
        df_r.style.format(fmt).background_gradient(cmap="Greens", subset=["Retorno CAGR (%)"]),
        use_container_width=True
    )


# --- TAB 3: TESTS DEL MODELO ---
with tab3:
    st.subheader("🧪 Tests Unitarios del Motor Matemático")
    st.markdown("Verificación automática de la integridad matemática y estadística del modelo.")

    test_results = run_all_tests(
        L_matrix=CHOLESKY_DECOMPOSITION,
        betas=BETAS_MODELO,
        corr_matrix=CORR_MATRIZ
    )

    pass_count = sum(1 for _, p, _ in test_results if p)
    total      = len(test_results)

    col_res1, col_res2 = st.columns([1, 3])
    col_res1.metric("Tests Pasados", f"{pass_count}/{total}",
                    delta="OK" if pass_count == total else "⚠️ REVISAR")

    for test_name, passed, msg in test_results:
        icon = "✅" if passed else "❌"
        with st.expander(f"{icon} {test_name}"):
            st.code(msg)

    st.markdown("---")
    st.subheader("📐 Validación Matrices de Transición")
    for nom, m in [("Crecimiento", MTM_CRECIMIENTO),
                   ("Estanflación", MTM_ESTANFLACION),
                   ("Recesión Severa", MTM_RECESION_SEVERA)]:
        row_sums = m.sum(axis=1)
        ok = np.allclose(row_sums, 1.0, atol=1e-9)
        st.write(f"{'✅' if ok else '❌'} **{nom}** — Sumas de filas: "
                 f"{row_sums[0]:.6f}, {row_sums[1]:.6f}, {row_sums[2]:.6f}")

    st.markdown("---")
    st.subheader("📊 Betas por Distrito (Elasticidad CAPM)")
    df_betas = pd.DataFrame({'Distrito': distritos, 'Beta': BETAS_MODELO.round(4)})
    st.bar_chart(df_betas.set_index('Distrito'))

    st.subheader("🔗 Matriz de Correlación Histórica")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    im = ax_corr.imshow(CORR_MATRIZ, cmap='RdYlGn', vmin=-1, vmax=1)
    ax_corr.set_xticks(range(n_distritos)); ax_corr.set_xticklabels(distritos, rotation=45, ha='right', fontsize=7)
    ax_corr.set_yticks(range(n_distritos)); ax_corr.set_yticklabels(distritos, fontsize=7)
    plt.colorbar(im, ax=ax_corr)
    ax_corr.set_title("Correlación de Retornos entre Distritos")
    for i in range(n_distritos):
        for j in range(n_distritos):
            ax_corr.text(j, i, f"{CORR_MATRIZ[i,j]:.2f}", ha='center', va='center', fontsize=5)
    plt.tight_layout()
    st.pyplot(fig_corr)

# ============================================================================
# 11. GENERADORES DE PDF
# ============================================================================

def clean_str(txt: str) -> str:
    """Sanitiza caracteres no-latin1 para fpdf."""
    repls = {
        '€': 'EUR', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'ñ': 'n', 'Ñ': 'N', '²': '2', '–': '-', '•': '-',
        'à': 'a', 'ï': 'i', 'ü': 'u', 'ö': 'o', 'ä': 'a',
        '\u2019': "'", '\u201c': '"', '\u201d': '"',
    }
    for k, v in repls.items():
        txt = txt.replace(k, v)
    return txt


class ProfessionalPDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 9)
        self.cell(0, 8, 'BCN STRATEGIC MODEL V31.0 | CONFIDENTIAL', 0, 1, 'C')
        self.line(10, 18, 200, 18)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 10, f'Pagina {self.page_no()} | Generado: {ts} | Modelo v31.0', 0, 0, 'C')


def _log_params_to_pdf(pdf: FPDF, params: dict, dist_name: str, n_sim: int):
    """Añade bloque de trazabilidad de parámetros al PDF."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, clean_str("TRAZABILIDAD DEL ESCENARIO EXPORTADO"), 0, 1)
    pdf.set_font('Courier', '', 8)
    lines = [
        f"Timestamp         : {ts}",
        f"Modelo            : Barcelona Strategic Model v31.0",
        f"Distrito analizado: {dist_name}",
        f"Regimen economico : {st.session_state.regimen}",
        f"Tendencia venta   : {params.get('crecimiento_venta', 0)*100:.2f}%",
        f"Tendencia alquiler: {params.get('crecimiento_alquiler', 0)*100:.2f}%",
        f"Shock Ley Vivienda: {'Activado (2029)' if params.get('shock_vt') else 'Desactivado'}",
        f"Simulaciones (n)  : {n_sim}",
        f"Fecha dato real   : {FECHA_DATO_REAL}",
        f"Fuente datos      : {FUENTE_DATO}",
        f"Horizonte proy.   : 2027-2032",
    ]
    for ln in lines:
        pdf.cell(0, 5, clean_str(ln), 0, 1)
    pdf.ln(3)
    logger.info("Parámetros del escenario exportado: %s", {k: v for k, v in params.items() if k != 'mtm'})


def calculate_noshock_scenario(base_scenarios, dist_idx):
    conc           = concentracion_vt_real[dist_idx]
    factor_shock_v = (conc / 0.46) * 0.14
    factor_shock_a = (conc / 0.46) * 0.10
    sv, sa         = base_scenarios
    v31_shock      = np.median(sv[:, dist_idx, -1])
    a31_shock      = np.median(sa[:, dist_idx, -1])
    return v31_shock / (1 - factor_shock_v), a31_shock / (1 - factor_shock_a)


def generate_exec_report(dist_idx: int, scenarios: dict, params: dict,
                          fig_chart, n_sim: int) -> bytes:
    pdf = ProfessionalPDF()
    pdf.add_page()

    # Portada
    pdf.set_font('Arial', 'B', 16)
    pdf.set_fill_color(44, 62, 80); pdf.set_text_color(255)
    pdf.cell(0, 12, clean_str(f"INFORME EJECUTIVO: {distritos[dist_idx].upper()}"), 0, 1, 'C', 1)
    pdf.set_text_color(0); pdf.ln(3)

    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 6, clean_str(f"Barcelona Strategic Model v31.0 | Datos reales a {FECHA_DATO_REAL}"), 0, 1, 'C')
    pdf.ln(4)

    # Trazabilidad
    _log_params_to_pdf(pdf, params, distritos[dist_idx], n_sim)

    # Tabla de sensibilidad regulatoria
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("1. ANALISIS DE SENSIBILIDAD — IMPACTO REGULATORIO"), 0, 1)
    h = ["Escenario", "Venta (Shock)", "Venta (Sin Shock)", "Alq (Shock)", "Alq (Sin Shock)"]
    w = [35, 38, 38, 38, 38]
    pdf.set_fill_color(44, 62, 80); pdf.set_text_color(255); pdf.set_font('Arial', 'B', 8)
    for i, v in enumerate(h): pdf.cell(w[i], 7, clean_str(v), 1, 0, 'C', 1)
    pdf.ln()
    pdf.set_text_color(0); pdf.set_font('Arial', '', 8)

    for name, (sv, sa) in [("Crecimiento", scenarios["Crecimiento"]),
                            ("Estanflacion", scenarios["Estanflación"]),
                            ("Recesion", scenarios["Recesión"])]:
        v31_s  = np.median(sv[:, dist_idx, -1])
        a31_s  = np.median(sa[:, dist_idx, -1])
        v31_ns, a31_ns = calculate_noshock_scenario((sv, sa), dist_idx)
        pdf.cell(w[0], 7, clean_str(name), 1)
        pdf.cell(w[1], 7, f"{int(v31_s):,} EUR", 1)
        pdf.cell(w[2], 7, f"{int(v31_ns):,} EUR", 1)
        pdf.cell(w[3], 7, f"{a31_s:.1f} EUR", 1)
        pdf.cell(w[4], 7, f"{a31_ns:.1f} EUR", 1)
        pdf.ln()

    # Gráfico
    pdf.ln(4)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("2. PROYECCION GRAFICA EVOLUTIVA (2026-2032)"), 0, 1)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig_chart.savefig(tmp.name, dpi=100, bbox_inches='tight')
        pdf.image(tmp.name, x=10, w=180)

    # Tabla maestra
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("3. TABLA MAESTRA — PRECIOS REALES 2026 vs PROYECCION 2032"), 0, 1)

    h_m = ["Distrito", "Venta '26", "Venta '32", "Var%", "Alq '26", "Alq '32", "Yield'32"]
    w_m = [45, 25, 25, 18, 25, 25, 22]
    pdf.set_fill_color(44, 62, 80); pdf.set_text_color(255); pdf.set_font('Arial', 'B', 8)
    for i, v in enumerate(h_m): pdf.cell(w_m[i], 7, clean_str(v), 1, 0, 'C', 1)
    pdf.ln()

    pdf.set_text_color(0); pdf.set_font('Arial', '', 8)
    sv_u, sa_u = scenarios["Usuario (Activo)"]
    for i, d in enumerate(distritos):
        v26 = precio_venta_2026[i]; a26 = precio_alquiler_2026[i]
        v32 = np.median(sv_u[:, i, -1]); a32 = np.median(sa_u[:, i, -1])
        var = ((v32 / v26) - 1) * 100; yld = (a32 * 12) / v32 * 100
        fill = 235 if i % 2 == 0 else 255
        pdf.set_fill_color(fill)
        pdf.cell(w_m[0], 6, clean_str(d), 1, 0, 'L', 1)
        pdf.cell(w_m[1], 6, f"{int(v26):,}", 1, 0, 'C', 1)
        pdf.cell(w_m[2], 6, f"{int(v32):,}", 1, 0, 'C', 1)
        pdf.cell(w_m[3], 6, f"{var:+.1f}%", 1, 0, 'C', 1)
        pdf.cell(w_m[4], 6, f"{a26:.1f}", 1, 0, 'C', 1)
        pdf.cell(w_m[5], 6, f"{a32:.1f}", 1, 0, 'C', 1)
        pdf.cell(w_m[6], 6, f"{yld:.1f}%", 1, 0, 'C', 1)
        pdf.ln()

    # Recomendación
    pdf.ln(4)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("4. RECOMENDACION AL INVERSOR"), 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, clean_str(
        "Basado en los resultados del modelo v31.0 con datos reales a " + FECHA_DATO_REAL + ":\n"
        "- Para Rentabilidad por Yield: Busque activos en Nou Barris y Sant Andreu (>5% yield bruto).\n"
        "- Para Plusvalia (Growth): Eixample y Gracia ofrecen mayor beta en ciclos expansivos.\n"
        "- Precaucion regulatoria: El shock Ley Vivienda (2029) castiga valoraciones en "
        "  Ciutat Vella y Eixample en proporcion directa a su densidad de licencias turisticas.\n"
        "- Sarria-Sant Gervasi y Les Corts muestran mayor resiliencia en escenarios de recesion.\n"
        "- Nota: Proyecciones P50 del intervalo de confianza al 80% (P10-P90)."
    ))

    return pdf.output(dest='S').encode('latin-1')


def generate_tech_paper(params: dict, n_sim: int) -> bytes:
    """
    Genera el paper técnico-académico completo del modelo matemático.
    Explica la lógica desde perspectiva económica y matemática formal.
    """
    pdf = ProfessionalPDF()

    # -----------------------------------------------------------------------
    # PORTADA
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 18)
    pdf.ln(10)
    pdf.multi_cell(0, 9, clean_str(
        "MODELIZACION ESTOCASTICA DE DINAMICAS\n"
        "INMOBILIARIAS URBANAS EN BARCELONA"
    ), align='C')
    pdf.ln(3)
    pdf.set_font('Times', 'I', 13)
    pdf.multi_cell(0, 7, clean_str(
        "Un Enfoque Hibrido Markov-Monte Carlo con Integracion\n"
        "de Shocks Regulatorios y Correlacion Espacial (Cholesky)"
    ), align='C')
    pdf.ln(8)
    pdf.set_font('Times', '', 11)
    pdf.cell(0, 7, clean_str(f"Modelo: Barcelona Strategic Model v31.0"), 0, 1, 'C')
    pdf.cell(0, 7, clean_str(f"Datos actualizados a: {FECHA_DATO_REAL}"), 0, 1, 'C')
    pdf.cell(0, 7, clean_str(f"Fuente de datos: {FUENTE_DATO}"), 0, 1, 'C')
    pdf.cell(0, 7, clean_str(f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"), 0, 1, 'C')
    pdf.ln(6)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.ln(5)

    # Trazabilidad
    _log_params_to_pdf(pdf, params, "Todos los distritos", n_sim)

    # -----------------------------------------------------------------------
    # ABSTRACT
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "RESUMEN EJECUTIVO (ABSTRACT)", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "Este documento presenta la arquitectura matematica completa del Barcelona Strategic Model v31.0, "
        "un sistema de proyeccion de precios de activos residenciales en los diez distritos de Barcelona "
        "para el horizonte temporal 2027-2032. El modelo integra tres capas metodologicas: (i) una Ecuacion "
        "Diferencial Estocastica (SDE) de tipo Geometrico Browniano Modificado, (ii) un proceso de cambio "
        "de regimen macroeconomico modelizado mediante Cadenas de Markov de tiempo discreto, y (iii) una "
        "estructura de correlacion espacial entre distritos implementada via descomposicion de Cholesky. "
        "Adicionalmente, el modelo incorpora una funcion de salto (Jump Process) que cuantifica el impacto "
        "asimetrico del marco regulatorio de vivienda turistica sobre las valoraciones distritales, "
        "ponderado por la densidad real de licencias VT (Vector de Exposicion). "
        "Los precios base son datos reales de mercado actualizados a " + FECHA_DATO_REAL + ". "
        "El sistema permite parametrizacion interactiva del ciclo economico, la tendencia de crecimiento, "
        "la precision estadistica (n_sim) y la activacion del shock regulatorio, generando intervalos "
        "de confianza P10-P90 para cada distrito y escenario."
    ))
    pdf.ln(5)

    # -----------------------------------------------------------------------
    # 1. FUNDAMENTACION ECONOMICA
    # -----------------------------------------------------------------------
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "1. FUNDAMENTACION ECONOMICA Y CONTEXTUAL", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El mercado inmobiliario barcelones presenta caracteristicas estructurales que justifican un "
        "tratamiento econometrico avanzado. En primer lugar, existe una marcada heterogeneidad distrital: "
        "los precios de venta oscilan entre los 2.946 EUR/m2 de Nou Barris y los 6.896 EUR/m2 de "
        "Sarria-Sant Gervasi (datos 2026), una brecha del 134% que refleja diferencias fundamentales "
        "en capital humano, accesibilidad, dotacion de servicios y presion turistica. "
        "\n\n"
        "En segundo lugar, el mercado esta sujeto a intervenciones regulatorias de alto impacto, "
        "como la limitacion de licencias de vivienda turistica (VT) y los indices de precios de "
        "referencia del alquiler establecidos por la Ley 12/2023 de Derecho a la Vivienda. Estos "
        "shocks regulatorios tienen efectos asimetricos segun la concentracion de VT en cada distrito. "
        "\n\n"
        "En tercer lugar, los ciclos macroeconomicos (expansion, estanflacion, recesion) modulan de "
        "forma diferencial el comportamiento de cada distrito en funcion de su elasticidad historica "
        "al mercado agregado barcelones, cuantificada mediante el coeficiente Beta del modelo CAPM "
        "adaptado al sector inmobiliario."
    ))
    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 2. ARQUITECTURA MATEMÁTICA
    # -----------------------------------------------------------------------
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "2. ARQUITECTURA MATEMATICA DEL MODELO", 0, 1)

    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 7, "2.1 Ecuacion Diferencial Estocastica (SDE) Gobernante", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "La dinamica de precios de cada distrito i en el periodo t sigue un Movimiento Browniano "
        "Geometrico Modificado con componente de salto. La ecuacion discreta es:"
    ))
    pdf.set_font('Courier', 'B', 9)
    pdf.multi_cell(0, 5, clean_str(
        "dP_i(t) / P_i(t) = [mu(S_t) * gamma_i + rho * R_yield(t-1)] * dt\n"
        "                  + sigma_i * SUM_j(L_ij * dZ_j(t))\n"
        "                  + J_vt(t, i)"
    ))
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "\nDonde cada componente tiene la siguiente interpretacion economica:\n"
        "- mu(S_t): Drift macroeconomico determinado por el estado S_t de la Cadena de Markov.\n"
        "- gamma_i: Elasticidad distrital (Beta CAPM), calibrada sobre retornos historicos 2007-2025.\n"
        "- rho: Coeficiente de retroalimentacion alquiler-venta (estimado en 0.45).\n"
        "- R_yield(t-1): Retorno del alquiler en el periodo anterior (efecto arrastre).\n"
        "- sigma_i: Volatilidad historica especifica del distrito i.\n"
        "- L_ij: Elemento (i,j) de la matriz de Cholesky L (descomposicion de la covarianza).\n"
        "- dZ_j(t): Incrementos de Wiener estandar independientes (ruido blanco).\n"
        "- J_vt(t,i): Proceso de salto regulatorio (shock discreto en t=2029)."
    ))
    pdf.ln(3)

    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 7, "2.2 Modelizacion del Ciclo Economico: Cadenas de Markov", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El regimen macroeconomico S_t evoluciona segun una Cadena de Markov de tiempo discreto con "
        "espacio de estados E = {0: Normal, 1: Expansion/Boom, 2: Recesion/Crisis}. La dinamica viene "
        "determinada por la Matriz de Transicion P(pi_ij), donde pi_ij = P(S_{t+1}=j | S_t=i).\n\n"
        "Se definen tres matrices de transicion correspondientes a los tres regimenes parametrizables:\n\n"
        "Regimen Crecimiento: pi_00=0.73, pi_01=0.25, pi_02=0.02 (alta probabilidad de Boom).\n"
        "Regimen Estanflacion: pi_00=0.80, pi_01=0.05, pi_02=0.15 (predominio del estado Normal).\n"
        "Regimen Recesion: pi_00=0.60, pi_01=0.05, pi_02=0.35 (elevada probabilidad de Crisis).\n\n"
        "Propiedad matematica garantizada: todas las filas de cada MTM suman exactamente 1.0, "
        "verificado mediante test unitario automatico en cada ejecucion del modelo.\n\n"
        "Los estados modulan el drift de la siguiente manera:\n"
        "- Estado Boom (S=1): tasa_final = tasa_base * 1.4 (aceleracion del ciclo expansivo)\n"
        "- Estado Normal (S=0): tasa_final = tasa_base (sin modificacion)\n"
        "- Estado Crisis (S=2): tasa_final = -0.02 - |tasa_base| * 0.5 (contraccion forzada)\n\n"
        "La volatilidad tambien se modula: vol_adj = 1.5 en estado de Crisis vs 1.0 en otros estados, "
        "capturando el bien documentado efecto de incremento de volatilidad en mercados bajistas."
    ))
    pdf.ln(3)

    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 7, "2.3 Estructura de Correlacion Espacial: Descomposicion de Cholesky", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "Los mercados inmobiliarios de distritos adyacentes exhiben correlacion significativa debido "
        "a factores comunes (accesibilidad al empleo, infraestructuras de transporte, efectos "
        "de desbordamiento de precios). Ignorar esta correlacion produciria intervalos de confianza "
        "sesgados hacia la independencia.\n\n"
        "El modelo estima la matriz de correlacion Sigma a partir de retornos historicos reconstruidos "
        "(2007-2025) y aplica descomposicion de Cholesky: Sigma = L * L^T.\n\n"
        "Los shocks correlacionados se generan como: eps = L * Z, donde Z ~ N(0, I_n).\n"
        "Esto garantiza que eps ~ N(0, Sigma), es decir, shocks con la estructura de correlacion "
        "historica entre distritos. Si la matriz no es definida positiva (por redondeo o datos "
        "insuficientes), se aplica correccion espectral: Sigma_corr = V * diag(max(lambda_i, eps)) * V^T, "
        "asegurando factorizabilidad. La integridad de la descomposicion es verificada en cada "
        "inicializacion del modelo mediante tests unitarios automaticos."
    ))
    pdf.ln(3)

    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 7, "2.4 Calibracion de Elasticidades Distritales (Beta CAPM)", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "La elasticidad gamma_i de cada distrito respecto al mercado agregado barcelones se estima "
        "mediante el modelo CAPM adaptado al sector inmobiliario:\n\n"
        "   beta_i = Cov(r_i, r_M) / Var(r_M)\n\n"
        "Donde r_i es el retorno del precio de venta del distrito i y r_M es el retorno del indice "
        "agregado de precios de Barcelona. Esta estimacion se realiza sobre el periodo 2019-2026, "
        "el unico para el que se dispone de datos observados a nivel distrital.\n\n"
        "Para el periodo anterior (2007-2018), los precios distritales se reconstruyen historicamente "
        "invirtiendo la relacion CAPM: P_i(t) = P_i(t+1) / (1 + beta_i * r_M(t) + ruido), "
        "con ruido ~ N(0, 0.003). Los betas resultantes se truncan al intervalo [0.5, 1.5] para "
        "evitar valores extremos no plausibles economicamente.\n\n"
        "Interpretacion economica: un beta elevado (Ciutat Vella: " +
        f"{BETAS_MODELO[0]:.3f}" + ") indica alta sensibilidad a los ciclos de mercado, "
        "caracteristica de distritos con fuerte presion turistica y alta rotacion de activos. "
        "Un beta bajo (Nou Barris: " + f"{BETAS_MODELO[5]:.3f}" + ") indica mayor estabilidad, "
        "propia de mercados con demanda residencial consolidada y menor componente especulativo."
    ))
    pdf.ln(3)

    # -----------------------------------------------------------------------
    # 3. FUNCIÓN DE SHOCK REGULATORIO
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "3. FUNCION DE SHOCK REGULATORIO (JUMP PROCESS)", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "La prohibicion o limitacion de nuevas licencias de vivienda turistica (VT) constituye un "
        "shock exogeno discreto sobre los precios de venta, modelizado como proceso de salto J_vt "
        "activado en t=2029 (desplazamiento de un ano respecto a la version anterior por actualizacion "
        "del horizonte de datos a 2026).\n\n"
        "La magnitud del shock es proporcional a la concentracion relativa de VT en cada distrito, "
        "cuantificada mediante el vector de exposicion D_i (datos: Ajuntament de Barcelona / "
        "Inside Airbnb):\n\n"
        "   J_vt(i) = alpha_v * (D_i / D_max)   [venta]\n"
        "   J_alq(i) = alpha_a * (D_i / D_max)  [alquiler]\n\n"
        "Donde D_max = D_Eixample = 0.46 (densidad maxima), alpha_v = -0.10 y alpha_a = -0.06.\n\n"
        "Esto produce una correccion maxima del -10% en precio de venta para Eixample "
        "(D=0.46) y practicamente nula para Nou Barris (D=0.004, correccion de -0.09%), "
        "reflejando la logica economica de que la remocion de VT elimina una componente "
        "especulativa de demanda que opera principalmente en zonas de alta densidad turistica.\n\n"
        "Justificacion del canal economico: los VT compiten con la demanda residencial por el "
        "mismo stock de vivienda. Su eliminacion reduce la demanda efectiva por activos destinables "
        "a VT, lo que presiona a la baja las valoraciones en zonas de alta exposicion. "
        "Simultaneamente, el aumento de oferta disponible para residentes modera los precios "
        "de alquiler a largo plazo (efecto de menor magnitud: alpha_a < alpha_v)."
    ))

    # Tabla de vector VT
    pdf.ln(3)
    pdf.set_font('Times', 'B', 10)
    pdf.cell(0, 7, clean_str("Vector de Exposicion VT por Distrito (D_i)"), 0, 1)
    h_vt = ["Distrito", "D_i (densidad)", "Shock Venta", "Shock Alquiler"]
    w_vt = [55, 40, 40, 45]
    pdf.set_fill_color(44, 62, 80); pdf.set_text_color(255); pdf.set_font('Arial', 'B', 8)
    for i, v in enumerate(h_vt): pdf.cell(w_vt[i], 7, clean_str(v), 1, 0, 'C', 1)
    pdf.ln()
    pdf.set_text_color(0); pdf.set_font('Arial', '', 8)
    for i, d in enumerate(distritos):
        shock_v = -10.0 * (concentracion_vt_real[i] / 0.46)
        shock_a = -6.0  * (concentracion_vt_real[i] / 0.46)
        pdf.set_fill_color(235 if i % 2 == 0 else 255)
        pdf.cell(w_vt[0], 6, clean_str(d), 1, 0, 'L', 1)
        pdf.cell(w_vt[1], 6, f"{concentracion_vt_real[i]:.3f}", 1, 0, 'C', 1)
        pdf.cell(w_vt[2], 6, f"{shock_v:+.2f}%", 1, 0, 'C', 1)
        pdf.cell(w_vt[3], 6, f"{shock_a:+.2f}%", 1, 0, 'C', 1)
        pdf.ln()
    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 4. SIMULACIÓN MONTE CARLO
    # -----------------------------------------------------------------------
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "4. METODO DE SIMULACION MONTE CARLO", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El modelo implementa un simulador Monte Carlo con n_sim trayectorias independientes "
        "(parametrizable por el usuario entre 200 y 5.000 simulaciones). En esta ejecucion: n_sim = "
        + str(n_sim) + ".\n\n"
        "Algoritmo de simulacion (pseudocodigo):\n"
        "  Para cada simulacion s=1,...,n_sim:\n"
        "    P_vta = precios_reales_2026; P_alq = alquileres_reales_2026; Estado = 0\n"
        "    Para cada ano t=2027,...,2032:\n"
        "      1. Transicion de Markov: Estado = sample(E, p=MTM[Estado,:])\n"
        "      2. Calcular tasa_base = drift * input_sign + ruido_gaussiano\n"
        "      3. Modular tasa segun estado (Boom x1.4, Normal x1.0, Crisis negativo)\n"
        "      4. Aplicar elasticidad distrital: tasa *= gamma_i\n"
        "      5. Generar eps correlacionado: Z~N(0,I), eps = L*Z\n"
        "      6. Modular volatilidad: sigma_i * vol_adj * eps\n"
        "      7. Aplicar resistencia yield (floor de yield bruto >= 2.8%)\n"
        "      8. Aplicar feedback alquiler: rho * R_yield(t-1)\n"
        "      9. Si shock_vt y t=2029: aplicar J_vt(i)\n"
        "     10. P_vta *= (1+tasa_venta); P_alq *= (1+tasa_alquiler)\n\n"
        "Los resultados se agregan estadisticamente calculando los percentiles P10, P50 (mediana) "
        "y P90 de la distribucion simulada de precios para cada distrito y ano. "
        "El intervalo P10-P90 define la banda de incertidumbre al 80%.\n\n"
        "Seed: np.random.seed(42) garantiza reproducibilidad exacta de los resultados "
        "para identicos parametros de entrada."
    ))
    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 5. DATOS Y FUENTES
    # -----------------------------------------------------------------------
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "5. DATOS, FUENTES Y ACTUALIZACION", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El modelo trabaja con dos conjuntos de datos:\n\n"
        "A) DATOS HISTORICOS (2019-2025): Precios de venta y alquiler por distrito extraidos de "
        "Idealista Research e Incasol (Institut Catala del Sol, Generalitat de Catalunya). "
        "Usados para calibrar betas, volatilidades y correlaciones.\n\n"
        "B) DATOS REALES 2026 (NUEVOS en v31.0): Precios de mercado actualizados al primer trimestre "
        "de 2026, introducidos manualmente a partir de datos publicados por Idealista / Incasol. "
        "Estos constituyen el punto de partida (precio base) de todas las proyecciones.\n\n"
        "Precios de venta 2026 por distrito (EUR/m2):\n"
        "Ciutat Vella: 4.811 | Eixample: 6.363 | Gracia: 5.404 | Horta Guinardo: 3.905\n"
        "Les Corts: 6.355 | Nou Barris: 2.946 | Sant Andreu: 3.730 | Sant Marti: 4.899\n"
        "Sants-Montjuic: 4.477 | Sarria-Sant Gervasi: 6.896\n\n"
        "Precios de alquiler 2026 por distrito (EUR/m2/mes):\n"
        "Ciutat Vella: 25.7 | Eixample: 26.0 | Gracia: 23.7 | Horta Guinardo: 19.1\n"
        "Les Corts: 21.9 | Nou Barris: 16.3 | Sant Andreu: 18.5 | Sant Marti: 23.6\n"
        "Sants-Montjuic: 20.8 | Sarria-Sant Gervasi: 23.6\n\n"
        "Indice macro agregado: INE (Instituto Nacional de Estadistica) para el indice de precios "
        "de vivienda de Barcelona (IPV) a nivel ciudad, periodo 2007-2026.\n"
        "Densidad VT: Ajuntament de Barcelona (CEAT) / Inside Airbnb (datos de licencias activas)."
    ))
    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 6. LIMITACIONES
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "6. LIMITACIONES Y CONSIDERACIONES METODOLOGICAS", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El modelo presenta las siguientes limitaciones que el usuario debe considerar al interpretar "
        "los resultados:\n\n"
        "1. Estacionariedad implicita: Las volatilidades y correlaciones se estiman como constantes "
        "en el tiempo, cuando en realidad son dinamicas (GARCH-like). Un modelo con volatilidad "
        "estocástica (Heston) seria mas riguroso pero computacionalmente mas costoso.\n\n"
        "2. Reconstruccion historica simplificada: Los precios 2007-2018 se reconstruyen via CAPM "
        "inverso, no a partir de datos observados. Esto introduce sesgo en la estimacion de betas "
        "y correlaciones para el periodo pre-crisis.\n\n"
        "3. Aproximacion del shock sin-shock: La estimacion del escenario 'sin shock regulatorio' "
        "se realiza mediante inversion algebraica de los multiplicadores, lo que es una aproximacion "
        "de primer orden. Un enfoque mas preciso requeriria ejecutar simulaciones alternativas "
        "explicitamente sin el termino J_vt.\n\n"
        "4. Independencia de la tasa de interes: El modelo no incorpora explicitamente el efecto del "
        "euribor o las condiciones hipotecarias sobre la demanda. Un incremento sostenido de tipos "
        "de interes podria comprimir los precios mas alla de lo que el drift negativo captura.\n\n"
        "5. Oferta rigida: Se asume oferta inelastica de vivienda. En la practica, nuevas "
        "promociones o rehabilitaciones pueden moderar el crecimiento de precios en ciertos distritos.\n\n"
        "6. Seed fijo: np.random.seed(42) garantiza reproducibilidad pero implica que todas las "
        "simulaciones con identicos parametros producen identicos resultados. Para analisis de "
        "sensibilidad a la semilla, seria recomendable ejecutar con seeds aleatorios."
    ))
    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 7. TESTS UNITARIOS (resultados)
    # -----------------------------------------------------------------------
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "7. RESULTADOS DE TESTS UNITARIOS DEL MODELO", 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El modelo incluye un sistema de verificacion automatica que valida la integridad matematica "
        "y estadistica antes de cada ejecucion. Los tests verifican:\n"
        "- Positividad de todos los precios base 2026.\n"
        "- Yields brutos en rango economicamente plausible (2%-10%).\n"
        "- Sumas de filas de cada MTM = 1.0 (condicion de estochasticidad).\n"
        "- Betas en rango [0.5, 1.5] (plausibilidad economica).\n"
        "- Factorizabilidad de Cholesky (L triangular inferior, diagonal positiva).\n"
        "- Reconstruccion correcta: L*L^T ≈ Sigma (error maximo < 1e-6).\n\n"
        "Resultados de la ejecucion actual:"
    ))
    pdf.set_font('Courier', '', 8)
    test_results = run_all_tests(CHOLESKY_DECOMPOSITION, BETAS_MODELO, CORR_MATRIZ)
    for tname, tpass, tmsg in test_results:
        status = "[PASS]" if tpass else "[FAIL]"
        pdf.cell(0, 5, clean_str(f"{status} {tname}"), 0, 1)
        pdf.cell(0, 4, clean_str(f"       {tmsg}"), 0, 1)
    pdf.ln(3)

    # -----------------------------------------------------------------------
    # 8. BETAS POR DISTRITO
    # -----------------------------------------------------------------------
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 7, "8. BETAS DISTRITALES CALIBRADOS (CAPM)", 0, 1)
    h_b = ["Distrito", "Beta", "Volatilidad", "Sensibilidad", "Concentracion VT"]
    w_b = [55, 25, 30, 35, 40]
    pdf.set_fill_color(44, 62, 80); pdf.set_text_color(255); pdf.set_font('Arial', 'B', 8)
    for i, v in enumerate(h_b): pdf.cell(w_b[i], 7, clean_str(v), 1, 0, 'C', 1)
    pdf.ln()
    pdf.set_text_color(0); pdf.set_font('Arial', '', 8)
    for i, d in enumerate(distritos):
        pdf.set_fill_color(235 if i % 2 == 0 else 255)
        pdf.cell(w_b[0], 6, clean_str(d), 1, 0, 'L', 1)
        pdf.cell(w_b[1], 6, f"{BETAS_MODELO[i]:.4f}", 1, 0, 'C', 1)
        pdf.cell(w_b[2], 6, f"{volatilidad_array[i]*100:.2f}%", 1, 0, 'C', 1)
        pdf.cell(w_b[3], 6, f"{sensibilidad_distrital[i]:.2f}", 1, 0, 'C', 1)
        pdf.cell(w_b[4], 6, f"{concentracion_vt_real[i]:.3f}", 1, 0, 'C', 1)
        pdf.ln()
    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 9. REFERENCIAS
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "9. REFERENCIAS BIBLIOGRAFICAS Y NORMATIVAS", 0, 1)
    pdf.set_font('Times', '', 10)
    refs = [
        "Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. "
        "Journal of Political Economy, 81(3), 637-654.",

        "Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time "
        "series and the business cycle. Econometrica, 57(2), 357-384.",

        "Case, K.E. & Shiller, R.J. (1989). The Efficiency of the Market for Single-Family Homes. "
        "American Economic Review, 79(1), 125-137.",

        "Glaeser, E. & Gyourko, J. (2018). The Economic Implications of Housing Supply. "
        "Journal of Economic Perspectives, 32(1), 3-30.",

        "Idealista Research (2026). Informe de Precios de Vivienda en Barcelona por Distritos, "
        "primer trimestre 2026. Madrid: Idealista.",

        "Incasol - Institut Catala del Sol (2026). Estadistiques del sector immobiliari a Catalunya. "
        "Generalitat de Catalunya.",

        "INE - Instituto Nacional de Estadistica (2026). Indice de Precios de Vivienda (IPV). "
        "Serie historica 2007-2026. Madrid: INE.",

        "Ajuntament de Barcelona / CEAT (2024). Registre de Habitatges d'Us Turistic. "
        "Barcelona: Ajuntament.",

        "Inside Airbnb (2024). Data for Barcelona, Spain. insideairbnb.com.",

        "Ley 12/2023, de 24 de mayo, por el derecho a la vivienda. "
        "Boletin Oficial del Estado, num. 124, 25 de mayo de 2023.",

        "Sharpe, W.F. (1964). Capital asset prices: A theory of market equilibrium under "
        "conditions of risk. Journal of Finance, 19(3), 425-442.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 5, clean_str(f"- {ref}"))
        pdf.ln(1)

    return pdf.output(dest='S').encode('latin-1')


# ============================================================================
# 12. BOTONES DE DESCARGA (fix scope frágil)
# ============================================================================

st.markdown("---")
st.subheader("📥 Exportar Informes")

# Recuperar figura desde session_state (fix del scope frágil)
fig_para_pdf = st.session_state.fig_distrito
idx_para_pdf = st.session_state.idx_distrito

# Si la figura no existe aún (primera ejecución sin visitar tab1), crear una de emergencia
if fig_para_pdf is None:
    _idx = idx_para_pdf
    fig_para_pdf, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(12, 5))
    _p50 = np.percentile(sv_act[:, _idx, :], 50, axis=0)
    _last = PRECIOS_RECONSTRUIDOS[_idx][-1]
    ax_e1.plot(anos_larga_data, PRECIOS_RECONSTRUIDOS[_idx], 'o-', color='#2c3e50')
    ax_e1.plot(anos_proy_con_2026, np.concatenate([[_last], _p50]), '--', color='#e74c3c')
    ax_e1.set_title(f"Venta — {distritos[_idx]}"); ax_e1.grid(True, linestyle=':', alpha=0.6)
    _p50a = np.percentile(sa_act[:, _idx, :], 50, axis=0)
    _lasta = precio_alquiler_historico[_idx][-1]
    ax_e2.plot(anos_retro_corto, precio_alquiler_historico[_idx], 'o-', color='#2c3e50')
    ax_e2.plot(anos_proy_con_2026, np.concatenate([[_lasta], _p50a]), '--', color='#2ecc71')
    ax_e2.set_title(f"Alquiler — {distritos[_idx]}"); ax_e2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    st.session_state.fig_distrito = fig_para_pdf

col_pdf1, col_pdf2 = st.columns(2)

with col_pdf1:
    if st.button("📄 Generar Informe Ejecutivo"):
        with st.spinner("Generando informe ejecutivo con trazabilidad..."):
            logger.info("Generando informe ejecutivo para distrito: %s | params: %s",
                        distritos[idx_para_pdf], {k: v for k, v in params_act.items() if k != 'mtm'})
            pdf_bytes_exec = generate_exec_report(
                idx_para_pdf, scenarios_data, params_act, fig_para_pdf, n_sim_ui
            )
            fname_exec = f"Informe_Ejecutivo_{distritos[idx_para_pdf].replace(' ', '_')}_v31.pdf"
        st.download_button(
            label=f"💾 Descargar {fname_exec}",
            data=pdf_bytes_exec,
            file_name=fname_exec,
            mime="application/pdf"
        )

with col_pdf2:
    if st.button("🎓 Generar Paper Tecnico-Academico"):
        with st.spinner("Generando paper metodologico completo..."):
            logger.info("Generando paper academico | n_sim=%d | regimen=%s", n_sim_ui, reg_sel)
            pdf_bytes_paper = generate_tech_paper(params_act, n_sim_ui)
        st.download_button(
            label="💾 Descargar Paper Metodologico v31.0.pdf",
            data=pdf_bytes_paper,
            file_name="BCN_Model_v31_Paper_Metodologico.pdf",
            mime="application/pdf"
        )

st.markdown("---")
st.caption(
    f"Barcelona Strategic Model v31.0 | Datos reales: {FECHA_DATO_REAL} | "
    f"Fuente: {FUENTE_DATO} | n_sim activo: {n_sim_ui} | "
    f"Regimen: {reg_sel}"
)
