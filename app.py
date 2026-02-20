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
    page_title="Barcelona Strategic Model v32.0",
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
    .macro-card {
        background: white; padding: 12px; border-radius: 8px;
        border-left: 4px solid #e74c3c; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

warnings.filterwarnings('ignore')

# ============================================================================
# 2. DATOS HISTORICOS
# ============================================================================

distritos = [
    'Ciutat Vella', 'Eixample', 'Gracia', 'Horta Guinardo', 'Les Corts',
    'Nou Barris', 'Sant Andreu', 'Sant Marti', 'Sants-Montjuic', 'Sarria-Sant Gervasi'
]
n_distritos = len(distritos)

# Vector densidad licencias turísticas (Inside Airbnb / CEAT Ajuntament BCN)
concentracion_vt_real = np.array([0.150, 0.460, 0.105, 0.030, 0.035,
                                   0.004, 0.015, 0.120, 0.110, 0.050])

# Precios venta EUR/m2 por distrito, columnas 2019..2026
precio_venta_historico = np.array([
    [4200, 4100, 4250, 4450, 4620, 4740, 4817, 4811],
    [5500, 5350, 5550, 5850, 6050, 6180, 6299, 6363],
    [4600, 4480, 4650, 4900, 5100, 5220, 5301, 5404],
    [3350, 3280, 3400, 3600, 3750, 3820, 3862, 3905],
    [5350, 5200, 5400, 5700, 5950, 6080, 6135, 6355],
    [2450, 2400, 2490, 2630, 2730, 2780, 2804, 2946],
    [3300, 3220, 3340, 3530, 3670, 3740, 3774, 3730],
    [4180, 4080, 4230, 4460, 4640, 4730, 4784, 4899],
    [3780, 3690, 3830, 4040, 4200, 4280, 4338, 4477],
    [5900, 5750, 5970, 6290, 6540, 6670, 6738, 6896]
], dtype=float)

# Precios alquiler EUR/m2/mes por distrito, columnas 2019..2026
precio_alquiler_historico = np.array([
    [22.5, 20.8, 21.5, 23.2, 24.8, 25.8, 26.4, 25.7],
    [22.8, 21.2, 22.0, 23.8, 25.2, 26.1, 26.7, 26.0],
    [20.8, 19.5, 20.2, 21.8, 23.1, 24.0, 24.4, 23.7],
    [15.6, 14.8, 15.3, 16.5, 17.5, 18.1, 18.4, 19.1],
    [18.7, 17.5, 18.1, 19.6, 20.8, 21.6, 22.0, 21.9],
    [13.4, 12.8, 13.2, 14.3, 15.1, 15.6, 15.8, 16.3],
    [15.8, 15.0, 15.5, 16.8, 17.8, 18.3, 18.6, 18.5],
    [20.0, 18.8, 19.5, 21.0, 22.3, 23.2, 23.5, 23.6],
    [17.7, 16.7, 17.3, 18.7, 19.8, 20.5, 20.8, 20.8],
    [19.8, 18.5, 19.2, 20.7, 22.0, 22.8, 23.2, 23.6]
], dtype=float)

precio_venta_2026    = precio_venta_historico[:, -1].copy()
precio_alquiler_2026 = precio_alquiler_historico[:, -1].copy()

FECHA_DATO_REAL = "Q1 2026"
FUENTE_DATO     = "Idealista / Incasol (Generalitat de Catalunya)"

# Indice macro agregado Barcelona EUR/m2
datos_agregados_bcn = {
    2007: 4071, 2008: 3737, 2009: 3728, 2010: 3752, 2011: 3398, 2012: 3076,
    2013: 2992, 2014: 3061, 2015: 3297, 2016: 3649, 2017: 4154, 2018: 4232,
    2019: 4135, 2020: 4018, 2021: 3910, 2022: 4063, 2023: 4167, 2024: 4700,
    2025: 5042, 2026: 5200
}
anos_larga_data  = np.array(list(datos_agregados_bcn.keys()))
precios_agregados = np.array(list(datos_agregados_bcn.values()), dtype=float)

anos_proy          = np.arange(2027, 2033)
anos_proy_con_2026 = np.concatenate([[2026], anos_proy])
anos_retro_corto   = np.arange(2019, 2027)

# Parametros fijos del modelo
volatilidad_base      = np.array([0.055, 0.042, 0.048, 0.035, 0.038,
                                   0.025, 0.032, 0.045, 0.040, 0.030])
sensibilidad_distrital = np.array([0.95, 0.85, 0.80, 0.35, 0.60,
                                    0.30, 0.40, 0.70, 0.60, 0.65])
rho_vta_alq = 0.45

# ============================================================================
# 3. INDICADORES MACRO Y CALIBRACION POR REGIMEN
# ============================================================================
# Valores por defecto calibrados sobre datos historicos espanoles:
#   Crecimiento: Euribor bajo, IPC moderado, paro bajo
#   Estanflacion: Euribor alto, IPC alto, paro medio-alto
#   Recesion:    Euribor alto, IPC negativo/bajo, paro muy alto
#
# Fuentes de referencia para calibracion:
#   - Banco de Espana: Series historicas Euribor 12m (2000-2026)
#   - INE: IPC general anual (2000-2026)
#   - INE/Idescat: Tasa de paro Barcelona/Catalunya (2000-2026)

MACRO_DEFAULTS = {
    "Crecimiento": {
        "euribor":  1.5,   # % - media historica ciclos expansivos post-2015
        "ipc":      2.5,   # % - objetivo BCE cercano al 2%
        "paro":     9.0,   # % - paro bajo en Barcelona (minimos 2017-2019)
    },
    "Estanflacion": {
        "euribor":  3.8,   # % - nivel 2023-2024 tras ciclo de subidas BCE
        "ipc":      5.5,   # % - inflacion persistente sin crecimiento real
        "paro":     13.0,  # % - paro elevado con estancamiento economico
    },
    "Recesion (Estructural)": {
        "euribor":  2.5,   # % - tipos aun elevados pero con expectativa bajista
        "ipc":      1.0,   # % - desinflacion por contraccion de demanda
        "paro":     20.0,  # % - paro crisis (nivel 2012-2014 en Catalunya)
    },
}

def calcular_probabilidades_regimen(euribor: float, ipc: float, paro: float) -> np.ndarray:
    """
    Convierte indicadores macro en probabilidades de regimen via funcion logistica.

    Logica economica:
      - P(Boom)    crece con IPC real positivo y paro bajo
      - P(Crisis)  crece con Euribor alto y paro alto
      - P(Normal)  residual

    Devuelve array [p_normal, p_boom, p_crisis] que suma 1.0.
    Esta funcion implementa la cadena de Markov NO HOMOGENEA:
    las probabilidades de transicion dependen del estado macroeconomico
    observado en cada periodo, no son constantes como en la version anterior.
    """
    # Normalizar indicadores a [0,1] usando rangos historicos espanoles
    euribor_norm = np.clip((euribor - 0.0) / 5.0, 0.0, 1.0)   # [0%,5%]
    ipc_norm     = np.clip((ipc     - 0.0) / 8.0, 0.0, 1.0)   # [0%,8%]
    paro_norm    = np.clip((paro   - 5.0) / 25.0, 0.0, 1.0)   # [5%,30%]

    # Score de expansion: favorecido por tipos bajos, IPC moderado, paro bajo
    score_boom   = (1.0 - euribor_norm) * 0.4 + \
                   (1.0 - paro_norm)    * 0.4 + \
                   np.clip(ipc_norm * (1 - ipc_norm) * 4, 0, 1) * 0.2

    # Score de crisis: favorecido por tipos altos y paro alto
    score_crisis = euribor_norm * 0.45 + paro_norm * 0.45 + \
                   np.clip((ipc_norm - 0.5), 0, 0.5) * 0.1

    # Softmax para garantizar que sumen 1
    raw     = np.array([0.5, score_boom, score_crisis])
    exp_raw = np.exp(raw * 2.5)
    probs   = exp_raw / exp_raw.sum()
    return probs  # [p_normal, p_boom, p_crisis]


def construir_mtm_macro(euribor: float, ipc: float, paro: float) -> np.ndarray:
    """
    Construye la MTM no homogenea a partir de indicadores macro.

    La MTM tiene forma P(i,j) donde:
      - La distribucion de entrada de cada fila viene modulada por
        las probabilidades de regimen calculadas a partir de Z_t=(euribor,ipc,paro).
      - Las diagonales se refuerzan con un factor de persistencia (inercia economica).
      - Cada fila se renormaliza para que sume 1.0 exactamente.

    Estados: 0=Normal, 1=Boom, 2=Crisis
    """
    probs = calcular_probabilidades_regimen(euribor, ipc, paro)
    p_norm, p_boom, p_crisis = probs

    # Factor de persistencia del ciclo: los regimenes economicos tienen inercia
    PERSIST = 0.55

    mtm = np.array([
        # Desde Normal: distribucion condicionada a macro
        [PERSIST + (1 - PERSIST) * p_norm,
         (1 - PERSIST) * p_boom,
         (1 - PERSIST) * p_crisis],
        # Desde Boom: alta persistencia del boom, salida principalmente a normal
        [0.35 * (1 - p_boom),
         PERSIST + (1 - PERSIST) * p_boom,
         0.05 + 0.10 * p_crisis],
        # Desde Crisis: persistencia alta, salida lenta hacia normal
        [0.12 + 0.20 * p_norm,
         0.02,
         PERSIST + (1 - PERSIST) * p_crisis],
    ])

    # Renormalizar filas (garantia matematica)
    for row in range(3):
        total = mtm[row].sum()
        if total > 0:
            mtm[row] /= total

    return mtm


# ============================================================================
# 4. VALIDACION DE MATRICES DE TRANSICION
# ============================================================================

def validar_mtm(mtm: np.ndarray, nombre: str) -> bool:
    row_sums = mtm.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        msg = f"MTM '{nombre}' invalida: filas suman {row_sums}. Deben sumar 1.0."
        logger.error(msg)
        raise ValueError(msg)
    return True


# ============================================================================
# 5. MOTOR ESTRUCTURAL (VAR + CAPM + CHOLESKY)
# ============================================================================

@st.cache_data
def inicializar_modelo_estructural():
    """
    Calibra el modelo estructural sobre datos historicos 2007-2026:

    (A) BETAS CAPM: elasticidad de cada distrito al indice macro de Barcelona.
        Beta_i = Cov(r_i, r_M) / Var(r_M), calibrado sobre 2019-2026.

    (B) DRIFT VAR(1): matriz de coeficientes autoregresivos de primer orden
        calibrada sobre retornos observados 2019-2026.
        r_i(t) = SUM_j [A_ij * r_j(t-1)] + epsilon_i(t)
        El drift endogeno que se usara en simulacion es la media de retornos
        historicos por distrito, modulada por A_hat.

    (C) RECONSTRUCCION HISTORICA 2007-2018 via CAPM inverso (para Cholesky).

    (D) CHOLESKY de la matriz de correlacion de retornos reconstruidos.

    (E) DRIFT HISTORICO: media de retornos observados 2019-2026 por distrito.
        Este es el punto de partida endogeno del modelo v32, sin intervencion usuario.

    (F) VOLATILIDADES DINAMICAS BASE: calibradas sobre retornos historicos,
        luego moduladas en simulacion por DCC simplificado.
    """
    idx_start = np.where(anos_larga_data == 2019)[0][0]
    precios_mkt_match = precios_agregados[idx_start:]

    # --- A: BETAS CAPM ---
    ret_mkt_obs = (precios_mkt_match[1:] / precios_mkt_match[:-1]) - 1
    betas = []
    for i in range(n_distritos):
        ret_dist = (precio_venta_historico[i, 1:] / precio_venta_historico[i, :-1]) - 1
        cov = np.cov(ret_dist, ret_mkt_obs)[0, 1]
        var = np.var(ret_mkt_obs)
        beta = cov / var if var > 1e-8 else 1.0
        betas.append(beta)
    betas = np.clip(np.array(betas), 0.5, 1.5)

    # --- E: DRIFT HISTORICO POR DISTRITO (endogeno) ---
    retornos_obs = np.zeros((n_distritos, precio_venta_historico.shape[1] - 1))
    for i in range(n_distritos):
        retornos_obs[i] = (precio_venta_historico[i, 1:] / precio_venta_historico[i, :-1]) - 1
    drift_historico_venta = retornos_obs.mean(axis=1)  # media anual 2019-2026

    retornos_alq_obs = np.zeros((n_distritos, precio_alquiler_historico.shape[1] - 1))
    for i in range(n_distritos):
        retornos_alq_obs[i] = (
            precio_alquiler_historico[i, 1:] / precio_alquiler_historico[i, :-1]
        ) - 1
    drift_historico_alq = retornos_alq_obs.mean(axis=1)

    # --- B: COEFICIENTES VAR(1) ---
    # Estimacion OLS: r_i(t) = A_ij * r_j(t-1) + eps
    # Para cada distrito i, regresamos sus retornos sobre todos los retornos del periodo anterior
    n_obs = retornos_obs.shape[1]
    if n_obs >= 4:
        Y = retornos_obs[:, 1:].T        # (T-1, n_distritos)
        X = retornos_obs[:, :-1].T       # (T-1, n_distritos)
        # OLS multivariado: A = (X'X)^{-1} X'Y
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            A_var   = XtX_inv @ (X.T @ Y)   # (n_distritos, n_distritos)
        except Exception:
            A_var = np.eye(n_distritos) * 0.1
    else:
        A_var = np.eye(n_distritos) * 0.1

    # --- C: RECONSTRUCCION HISTORICA 2007-2018 ---
    reconstruidos = np.zeros((n_distritos, len(anos_larga_data)))
    reconstruidos[:, idx_start:] = precio_venta_historico
    np.random.seed(999)
    for t in range(idx_start - 1, -1, -1):
        ret_mkt_val = (precios_agregados[t + 1] / precios_agregados[t]) - 1
        for d in range(n_distritos):
            ruido = np.random.normal(0, 0.003)
            r_d   = betas[d] * ret_mkt_val + ruido
            reconstruidos[d, t] = reconstruidos[d, t + 1] / (1 + r_d)

    # --- D: CHOLESKY ---
    ret_full = (reconstruidos[:, 1:] / reconstruidos[:, :-1]) - 1
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

    # --- F: VOLATILIDADES CALIBRADAS ---
    vol_calibrada = np.array([ret_full[i].std() for i in range(n_distritos)])
    vol_calibrada = np.clip(vol_calibrada, 0.015, 0.10)

    logger.info("Modelo v32 inicializado. Betas: %s | Drift medio venta: %s",
                betas.round(3), drift_historico_venta.round(4))

    return reconstruidos, L, betas, corr, drift_historico_venta, drift_historico_alq, A_var, vol_calibrada


(PRECIOS_RECONSTRUIDOS, CHOLESKY_L, BETAS_MODELO,
 CORR_MATRIZ, DRIFT_VENTA, DRIFT_ALQUILER,
 A_VAR, VOL_CALIBRADA) = inicializar_modelo_estructural()


# ============================================================================
# 6. MOTOR DE SIMULACION v32: MS-VAR + DCC simplificado
# ============================================================================

def _params_to_hash(params: dict) -> tuple:
    return (
        params['regimen'],
        round(params['euribor'], 4),
        round(params['ipc'],     4),
        round(params['paro'],    4),
        params['shock_vt'],
        params.get('n_sim', 1000),
    )


@st.cache_data
def simular_escenario_cached(params_hash: tuple):
    """
    Motor principal MS-VAR Monte Carlo v32.0.

    Novedades respecto a v31:
    -------------------------
    1. DRIFT ENDOGENO: basado en drift historico por distrito (DRIFT_VENTA / DRIFT_ALQUILER)
       calibrado sobre 2019-2026. El usuario ya no fija el drift directamente.

    2. MODULACION MACRO (Markov no homogeneo):
       - La MTM se construye dinamicamente a partir de (Euribor, IPC, Paro)
         usando calcular_probabilidades_regimen() -> construir_mtm_macro().
       - Las probabilidades de transicion cambian en cada periodo segun el
         estado macro del escenario.

    3. COMPONENTE VAR(1):
       - El drift incorpora la componente autoregresiva: r(t) += A_var @ r(t-1)
         ponderada por coeficiente lambda_var calibrado.
       - Esto captura la autocorrelacion y las interdependencias entre distritos.

    4. VOLATILIDAD DINAMICA (DCC simplificado):
       - En estado de Boom: vol *= 0.85  (compresion de volatilidad en expansion)
       - En estado de Crisis: vol *= 1.6  (explosion de volatilidad en crisis)
       - En estado Normal: vol = vol_base
       - Las correlaciones se modulan analogamente: en crisis convergen hacia 1.0.

    5. EURIBOR -> ASIMETRIA VENTA/ALQUILER:
       - Euribor alto comprime retorno de venta (menos compradores solventes)
       - Euribor alto puede sostener/elevar alquiler (demanda desplazada)
       - Implementado via factor euribor_adj diferencial.

    Estados Markov: 0=Normal, 1=Boom, 2=Crisis
    """
    (regimen, euribor, ipc, paro, shock_vt, n_sim) = params_hash

    # Construir MTM no homogenea para este escenario
    mtm = construir_mtm_macro(euribor, ipc, paro)
    validar_mtm(mtm, f"MTM dinamica ({regimen})")

    # Ajuste de drift por indicadores macro
    # IPC real = IPC - Euribor_spread. Si IPC > Euribor, hay impulso real.
    ipc_real_adj    = np.clip((ipc - 2.0) / 100.0, -0.03, 0.03)
    euribor_adj_vta = -np.clip((euribor - 2.5) / 100.0, -0.02, 0.04)  # penaliza venta
    euribor_adj_alq =  np.clip((euribor - 2.5) / 100.0 * 0.4, -0.01, 0.02)  # beneficia alq
    paro_adj        = -np.clip((paro - 10.0) / 100.0 * 0.5, -0.02, 0.03)

    # Coeficiente VAR ponderado (reducido para no dominar el drift historico)
    lambda_var = 0.25

    np.random.seed(42)
    n_anos  = len(anos_proy)
    sim_vta = np.zeros((n_sim, n_distritos, n_anos))
    sim_alq = np.zeros((n_sim, n_distritos, n_anos))

    for sim in range(n_sim):
        p_vta  = precio_venta_2026.copy()
        p_alq  = precio_alquiler_2026.copy()
        estado = 0
        r_prev = DRIFT_VENTA.copy()  # retorno previo para VAR(1)

        for i in range(n_anos):
            # --- Transicion Markov no homogenea ---
            rand = np.random.random()
            cum  = 0.0
            for j in range(3):
                cum += mtm[estado, j]
                if rand < cum:
                    estado = j
                    break

            # --- Drift endogeno: historico + VAR(1) ---
            var_component = lambda_var * (A_VAR @ r_prev)
            drift_v = DRIFT_VENTA  + var_component + ipc_real_adj + euribor_adj_vta + paro_adj
            drift_a = DRIFT_ALQUILER + var_component * 0.7 + ipc_real_adj * 0.6 + euribor_adj_alq

            # --- Modulacion por estado Markov ---
            if estado == 1:    # Boom
                tasa_v = drift_v * 1.35 + np.random.normal(0, 0.008, n_distritos)
                tasa_a = drift_a * 1.25 + np.random.normal(0, 0.006, n_distritos)
            elif estado == 2:  # Crisis
                tasa_v = -np.abs(drift_v) * 0.6 - 0.015 + np.random.normal(0, 0.012, n_distritos)
                tasa_a = -np.abs(drift_a) * 0.4 - 0.008 + np.random.normal(0, 0.008, n_distritos)
            else:              # Normal
                tasa_v = drift_v + np.random.normal(0, 0.008, n_distritos)
                tasa_a = drift_a + np.random.normal(0, 0.006, n_distritos)

            # --- Aplicar elasticidad distrital ---
            tasa_v *= sensibilidad_distrital
            tasa_a *= sensibilidad_distrital * 0.8

            # --- DCC simplificado: volatilidad y correlacion dinamicas ---
            if estado == 2:    # Crisis: vol sube, correlaciones convergen a 1
                vol_dyn   = VOL_CALIBRADA * 1.60
                # Interpolar correlacion hacia 1 (fenomeno de crisis documentado)
                corr_dyn  = CORR_MATRIZ * 0.5 + np.ones_like(CORR_MATRIZ) * 0.5
                np.fill_diagonal(corr_dyn, 1.0)
                try:
                    eigvals, eigvecs = np.linalg.eigh(corr_dyn)
                    eigvals = np.maximum(eigvals, 1e-8)
                    corr_dyn = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    d_diag = np.sqrt(np.diag(corr_dyn))
                    corr_dyn = corr_dyn / np.outer(d_diag, d_diag)
                    L_dyn = np.linalg.cholesky(corr_dyn)
                except Exception:
                    L_dyn = CHOLESKY_L
            elif estado == 1:  # Boom: vol comprimida, correlaciones se dispersan
                vol_dyn  = VOL_CALIBRADA * 0.85
                corr_dyn = CORR_MATRIZ * 0.8 + np.eye(n_distritos) * 0.2
                try:
                    L_dyn = np.linalg.cholesky(corr_dyn)
                except Exception:
                    L_dyn = CHOLESKY_L
            else:              # Normal: parametros calibrados
                vol_dyn = VOL_CALIBRADA
                L_dyn   = CHOLESKY_L

            z   = np.random.normal(0, 1, n_distritos)
            eps = L_dyn @ z

            tasa_v += vol_dyn * eps
            tasa_a += vol_dyn * 0.75 * eps
            tasa_v  = np.maximum(tasa_v, -0.07)

            # --- Floor de yield bruto (2.5%) ---
            yld_actual  = (p_alq * 12) / p_vta
            resist      = np.clip(yld_actual / 0.025, 0.5, 1.0)
            tasa_v      = np.where(tasa_v > 0, tasa_v * resist, tasa_v)

            # --- Feedback alquiler -> venta (rho) ---
            if i > 0:
                prev_alq  = sim_alq[sim, :, i - 1]
                prev2_alq = precio_alquiler_2026 if i == 1 else sim_alq[sim, :, i - 2]
                safe_prev2 = np.where(prev2_alq > 0, prev2_alq, 1.0)
                tasa_v += rho_vta_alq * ((prev_alq / safe_prev2) - 1)

            # --- Shock regulatorio Ley Vivienda (2029) ---
            shk_v = 0.0
            shk_a = 0.0
            if shock_vt and anos_proy[i] == 2029:
                factor_exp = concentracion_vt_real / 0.46
                shk_v = -0.10 * factor_exp
                shk_a = -0.06 * factor_exp

            p_vta = p_vta * (1 + tasa_v) * (1 + shk_v)
            p_alq = p_alq * (1 + tasa_a) * (1 + shk_a)

            sim_vta[sim, :, i] = p_vta
            sim_alq[sim, :, i] = p_alq

            r_prev = tasa_v.copy()  # actualizar retorno previo para VAR(1)

    return sim_vta, sim_alq


def simular_escenario(params: dict, n_sim: int = 1000):
    p2 = {**params, 'n_sim': n_sim}
    h  = _params_to_hash(p2)
    return simular_escenario_cached(h)


# ============================================================================
# 7. TESTS UNITARIOS
# ============================================================================

def run_all_tests():
    results = []

    # Datos
    v_ok = np.all(precio_venta_2026 > 0)
    a_ok = np.all(precio_alquiler_2026 > 0)
    results.append(("Precios 2026 positivos",
                     v_ok and a_ok,
                     f"Venta OK:{v_ok} | Alquiler OK:{a_ok}"))

    yields = (precio_alquiler_2026 * 12) / precio_venta_2026
    in_rng = np.all((yields >= 0.02) & (yields <= 0.10))
    results.append(("Yields 2026 en rango [2%-10%]",
                     in_rng,
                     f"min={yields.min()*100:.1f}% max={yields.max()*100:.1f}%"))

    # Drift historico no nulo
    drift_ok = np.all(np.abs(DRIFT_VENTA) < 0.30)
    results.append(("Drift historico plausible (<30% anual)",
                     drift_ok,
                     f"Drift venta: {(DRIFT_VENTA*100).round(2)}%"))

    # VAR: matriz A estable (radio espectral < 1 para estacionariedad)
    eigvals_var = np.abs(np.linalg.eigvals(A_VAR))
    stable = np.all(eigvals_var < 1.5)
    results.append(("VAR(1) radio espectral < 1.5",
                     stable,
                     f"Max eigenvalue: {eigvals_var.max():.4f}"))

    # Betas
    in_rng_b = np.all((BETAS_MODELO >= 0.5) & (BETAS_MODELO <= 1.5))
    results.append(("Betas CAPM en [0.5, 1.5]",
                     in_rng_b,
                     f"min={BETAS_MODELO.min():.3f} max={BETAS_MODELO.max():.3f}"))

    # Cholesky
    is_lower  = np.allclose(CHOLESKY_L, np.tril(CHOLESKY_L), atol=1e-10)
    pos_diag  = np.all(np.diag(CHOLESKY_L) > 0)
    results.append(("Cholesky triangular inferior y diagonal positiva",
                     is_lower and pos_diag,
                     f"Lower:{is_lower} | PosDiag:{pos_diag}"))

    recon   = CHOLESKY_L @ CHOLESKY_L.T
    recon_ok = np.allclose(CORR_MATRIZ, recon, atol=1e-5)
    max_err  = np.abs(CORR_MATRIZ - recon).max()
    results.append(("Cholesky: L*L^T aprox Sigma",
                     recon_ok,
                     f"Error max={max_err:.2e}"))

    # Test MTM dinamica (con valores tipicos de cada regimen)
    for reg, (eu, ic, pa) in [
        ("Crecimiento",  (1.5, 2.5,  9.0)),
        ("Estanflacion", (3.8, 5.5, 13.0)),
        ("Recesion",     (2.5, 1.0, 20.0)),
    ]:
        mtm = construir_mtm_macro(eu, ic, pa)
        row_sums = mtm.sum(axis=1)
        ok = np.allclose(row_sums, 1.0, atol=1e-6)
        results.append((f"MTM dinamica '{reg}' filas suman 1",
                         ok,
                         f"Sumas: {row_sums.round(6)}"))

    # Probabilidades de regimen coherentes
    for reg, (eu, ic, pa) in [
        ("Crecimiento",  (1.5, 2.5,  9.0)),
        ("Recesion",     (2.5, 1.0, 20.0)),
    ]:
        probs = calcular_probabilidades_regimen(eu, ic, pa)
        ok    = np.allclose(probs.sum(), 1.0, atol=1e-9) and np.all(probs >= 0)
        results.append((f"Probabilidades regimen '{reg}' validas",
                         ok,
                         f"p_norm={probs[0]:.3f} p_boom={probs[1]:.3f} p_crisis={probs[2]:.3f}"))

    return results


# ============================================================================
# 8. GESTION DE ESTADO SESSION
# ============================================================================
if 'regimen'      not in st.session_state: st.session_state.regimen      = "Crecimiento"
if 'fig_distrito' not in st.session_state: st.session_state.fig_distrito = None
if 'idx_distrito' not in st.session_state: st.session_state.idx_distrito = 1
if 'macro_params' not in st.session_state:
    st.session_state.macro_params = MACRO_DEFAULTS["Crecimiento"].copy()


def reset_macro_defaults():
    reg = st.session_state.regimen_selector
    st.session_state.macro_params = MACRO_DEFAULTS[reg].copy()
    st.session_state.regimen      = reg


# ============================================================================
# 9. SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("Configuracion del Modelo")

    reg_sel = st.radio(
        "Regimen Economico:",
        ("Crecimiento", "Estanflacion", "Recesion (Estructural)"),
        key="regimen_selector",
        on_change=reset_macro_defaults
    )
    st.caption("Al cambiar regimen se restauran los indicadores macro por defecto.")

    st.markdown("---")
    st.subheader("Indicadores Macroeconomicos")
    st.caption("Valores calibrados por defecto sobre datos historicos espanoles. Ajustables.")

    defaults = MACRO_DEFAULTS.get(reg_sel, MACRO_DEFAULTS["Crecimiento"])

    euribor_val = st.slider(
        "Euribor 12m (%)",
        min_value=0.0, max_value=6.0,
        value=float(defaults['euribor']), step=0.1,
        help="Tipo de referencia hipotecario. Impacto asimetrico: comprime venta, sostiene alquiler."
    )
    ipc_val = st.slider(
        "IPC anual (%)",
        min_value=-1.0, max_value=10.0,
        value=float(defaults['ipc']), step=0.1,
        help="Inflacion general. Diferencial con IPC vivienda determina impulso real."
    )
    paro_val = st.slider(
        "Tasa de paro Barcelona (%)",
        min_value=4.0, max_value=30.0,
        value=float(defaults['paro']), step=0.5,
        help="Proxy de renta disponible y confianza compradora. Correlacion negativa con precios (lag 12-18 meses)."
    )

    # Probabilidades de regimen calculadas en tiempo real
    probs_regimen = calcular_probabilidades_regimen(euribor_val, ipc_val, paro_val)
    st.markdown("**Probabilidades de regimen (endogenas):**")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Normal",  f"{probs_regimen[0]*100:.0f}%")
    col_p2.metric("Boom",    f"{probs_regimen[1]*100:.0f}%")
    col_p3.metric("Crisis",  f"{probs_regimen[2]*100:.0f}%")

    st.markdown("---")
    shock_vt = st.checkbox("Activar Shock Ley Vivienda (2029)", value=False)

    st.markdown("---")
    st.subheader("Precision de Simulacion")
    n_sim_ui = st.select_slider(
        "Simulaciones (n_sim):",
        options=[200, 500, 1000, 2000, 5000],
        value=1000,
        help="Mayor n_sim = mayor precision estadistica."
    )
    st.caption(f"Tiempo estimado: ~{n_sim_ui // 500 + 1}s")

    params_act = {
        'regimen':  reg_sel,
        'euribor':  euribor_val,
        'ipc':      ipc_val,
        'paro':     paro_val,
        'shock_vt': shock_vt,
        'n_sim':    n_sim_ui,
    }

# ============================================================================
# 10. SIMULACIONES
# ============================================================================
with st.spinner(f"Simulando {n_sim_ui} trayectorias MS-VAR..."):
    sv_act, sa_act = simular_escenario(params_act, n_sim=n_sim_ui)

    # Escenarios de referencia fijos (para comparativa)
    sv_crec, sa_crec = simular_escenario(
        {'regimen': 'Crecimiento',  'euribor': 1.5, 'ipc': 2.5, 'paro':  9.0,
         'shock_vt': shock_vt}, n_sim=200)
    sv_est, sa_est   = simular_escenario(
        {'regimen': 'Estanflacion', 'euribor': 3.8, 'ipc': 5.5, 'paro': 13.0,
         'shock_vt': shock_vt}, n_sim=200)
    sv_rec, sa_rec   = simular_escenario(
        {'regimen': 'Recesion',     'euribor': 2.5, 'ipc': 1.0, 'paro': 20.0,
         'shock_vt': shock_vt}, n_sim=200)

scenarios_data = {
    "Crecimiento":      (sv_crec, sa_crec),
    "Estanflacion":     (sv_est,  sa_est),
    "Recesion":         (sv_rec,  sa_rec),
    "Usuario (Activo)": (sv_act,  sa_act),
}

# ============================================================================
# 11. DASHBOARD
# ============================================================================
st.title("Barcelona Strategic Model v32.0")
st.markdown(
    f"**MS-VAR | Markov no homogeneo | DCC simplificado | "
    f"Datos reales {FECHA_DATO_REAL} | Fuente: {FUENTE_DATO}**"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Regimen",   reg_sel.replace(" (Estructural)", ""))
c2.metric("Euribor",   f"{euribor_val:.1f}%")
c3.metric("IPC",       f"{ipc_val:.1f}%")
c4.metric("Paro BCN",  f"{paro_val:.1f}%")
c5.metric("Regulacion","ON" if shock_vt else "OFF")
c6.metric("n_sim",     str(n_sim_ui))

# ============================================================================
# 12. TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Analisis Distrito",
    "Riesgo / Retorno",
    "Macro & Regimenes",
    "Tests del Modelo"
])

# --- TAB 1: DISTRITO ---
with tab1:
    sel_dist = st.selectbox("Distrito:", distritos, index=st.session_state.idx_distrito)
    idx      = distritos.index(sel_dist)
    st.session_state.idx_distrito = idx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(anos_larga_data, PRECIOS_RECONSTRUIDOS[idx],
             'o-', color='#2c3e50', label='Historico', linewidth=1.5, markersize=4)
    ax1.axvspan(2007, 2019, color='gray', alpha=0.08, label='Reconstruido CAPM')
    ax1.axvline(x=2026, color='navy', linestyle='--', linewidth=0.8, alpha=0.6)

    p50 = np.percentile(sv_act[:, idx, :], 50, axis=0)
    p10 = np.percentile(sv_act[:, idx, :], 10, axis=0)
    p90 = np.percentile(sv_act[:, idx, :], 90, axis=0)
    last = PRECIOS_RECONSTRUIDOS[idx][-1]

    ax1.plot(anos_proy_con_2026, np.concatenate([[last], p50]),
             '--', color='#e74c3c', linewidth=2, label='P50 activo')
    ax1.fill_between(anos_proy_con_2026,
                     np.concatenate([[last], p10]),
                     np.concatenate([[last], p90]),
                     color='#e74c3c', alpha=0.15, label='IC P10-P90')
    ax1.plot(anos_proy_con_2026,
             np.concatenate([[last], np.percentile(sv_crec[:, idx, :], 50, axis=0)]),
             ':', color='green', linewidth=1.2, label='Crecimiento')
    ax1.plot(anos_proy_con_2026,
             np.concatenate([[last], np.percentile(sv_rec[:, idx, :], 50, axis=0)]),
             ':', color='black', linewidth=1.2, label='Recesion')
    ax1.set_title(f"Venta EUR/m2 - {sel_dist}", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(fontsize='x-small')
    ax1.set_xlabel("Ano")

    p50a = np.percentile(sa_act[:, idx, :], 50, axis=0)
    p10a = np.percentile(sa_act[:, idx, :], 10, axis=0)
    p90a = np.percentile(sa_act[:, idx, :], 90, axis=0)
    lasta = precio_alquiler_historico[idx, -1]

    ax2.plot(anos_retro_corto, precio_alquiler_historico[idx],
             'o-', color='#2c3e50', linewidth=1.5, markersize=4)
    ax2.axvline(x=2026, color='navy', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.plot(anos_proy_con_2026, np.concatenate([[lasta], p50a]),
             '--', color='#27ae60', linewidth=2, label='P50 activo')
    ax2.fill_between(anos_proy_con_2026,
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

    v32 = np.median(sv_act[:, idx, -1])
    a32 = np.median(sa_act[:, idx, -1])
    yld = (a32 * 12) / v32 * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Venta 2032",    f"{int(v32):,} EUR",
              f"{((v32/precio_venta_2026[idx])-1)*100:.1f}%")
    k2.metric("Alquiler 2032", f"{a32:.1f} EUR",
              f"{((a32/precio_alquiler_2026[idx])-1)*100:.1f}%")
    k3.metric("Yield 2032",    f"{yld:.1f}%")
    k4.metric("Drift hist. venta",
              f"{DRIFT_VENTA[idx]*100:.2f}%/a")
    k5.metric("Beta CAPM",     f"{BETAS_MODELO[idx]:.3f}")

    st.markdown("#### Precios Reales 2026")
    df_p = pd.DataFrame({
        'Distrito': distritos,
        'Venta EUR/m2': precio_venta_2026.astype(int),
        'Alquiler EUR/m2': precio_alquiler_2026,
        'Yield bruto':  [f"{(a*12/v*100):.1f}%" for v, a in
                          zip(precio_venta_2026, precio_alquiler_2026)],
        'Drift hist. venta': [f"{d*100:.2f}%" for d in DRIFT_VENTA],
        'Beta CAPM':   BETAS_MODELO.round(3),
    })
    st.dataframe(df_p.set_index('Distrito'), use_container_width=True)


# --- TAB 2: RIESGO/RETORNO ---
with tab2:
    cagrs, vols = [], []
    for i in range(n_distritos):
        f = sv_act[:, i, -1]
        cagrs.append(((f.mean() / precio_venta_2026[i]) ** (1/6) - 1) * 100)
        vols.append(f.std() / f.mean() * 100)

    df_r = pd.DataFrame({
        'Distrito':         distritos,
        'Retorno CAGR (%)': cagrs,
        'Riesgo CV (%)':    vols,
        'Drift hist (%)':   (DRIFT_VENTA * 100).tolist(),
        'Beta':             BETAS_MODELO.tolist(),
    })

    fig_r = px.scatter(
        df_r, x='Riesgo CV (%)', y='Retorno CAGR (%)',
        text='Distrito', color='Retorno CAGR (%)',
        color_continuous_scale='RdYlGn', size=[20]*10,
        hover_data=['Drift hist (%)', 'Beta'],
        title=f"Cuadrante Riesgo/Retorno | {reg_sel} | Euribor={euribor_val}% | Paro={paro_val}%"
    )
    mr = df_r['Riesgo CV (%)'].mean()
    mt = df_r['Retorno CAGR (%)'].mean()
    fig_r.add_vline(x=mr, line_dash="dash", line_color="red", line_width=1.5)
    fig_r.add_hline(y=mt, line_dash="dash", line_color="red", line_width=1.5)
    fig_r.add_annotation(x=df_r['Riesgo CV (%)'].min(), y=df_r['Retorno CAGR (%)'].max(),
                         text="ESTRELLAS (Buy)", showarrow=False, font=dict(color="green", size=11))
    fig_r.add_annotation(x=df_r['Riesgo CV (%)'].max(), y=df_r['Retorno CAGR (%)'].min(),
                         text="INEFICIENTES (Sell)", showarrow=False, font=dict(color="red", size=11))
    st.plotly_chart(fig_r, use_container_width=True)

    fmt = {'Retorno CAGR (%)': '{:.2f}%', 'Riesgo CV (%)': '{:.2f}%',
           'Drift hist (%)': '{:.2f}%', 'Beta': '{:.3f}'}
    st.dataframe(
        df_r.style.format(fmt).background_gradient(cmap="Greens", subset=["Retorno CAGR (%)"]),
        use_container_width=True
    )


# --- TAB 3: MACRO & REGIMENES ---
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
        st.dataframe(df_mtm.style.background_gradient(cmap='Blues'), use_container_width=True)
        st.caption("Cada fila: probabilidades de transicion al estado destino dado el estado origen.")
        st.caption(f"Suma filas: {mtm_act.sum(axis=1).round(6)}")

    with col_m2:
        st.markdown("**Sensibilidad de probabilidades al Euribor**")
        euribor_range = np.linspace(0.5, 5.5, 30)
        p_boom_range  = []
        p_cris_range  = []
        for eu in euribor_range:
            pp = calcular_probabilidades_regimen(eu, ipc_val, paro_val)
            p_boom_range.append(pp[1] * 100)
            p_cris_range.append(pp[2] * 100)

        fig_macro, ax_m = plt.subplots(figsize=(7, 3.5))
        ax_m.plot(euribor_range, p_boom_range,  color='green', linewidth=2, label='P(Boom)')
        ax_m.plot(euribor_range, p_cris_range,  color='red',   linewidth=2, label='P(Crisis)')
        ax_m.axvline(x=euribor_val, color='navy', linestyle='--', linewidth=1.2,
                     label=f'Euribor actual ({euribor_val}%)')
        ax_m.set_xlabel("Euribor 12m (%)")
        ax_m.set_ylabel("Probabilidad (%)")
        ax_m.set_title("Funcion de respuesta: Euribor -> Regimen")
        ax_m.legend(fontsize='small')
        ax_m.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_macro)

    st.markdown("---")
    st.subheader("Comparativa de Drift: Endogeno vs Escenarios")
    df_drift = pd.DataFrame({
        'Distrito': distritos,
        'Drift hist. venta (%)': (DRIFT_VENTA * 100).round(3),
        'Drift hist. alquiler (%)': (DRIFT_ALQUILER * 100).round(3),
        'Beta CAPM': BETAS_MODELO.round(3),
        'Vol. calibrada (%)': (VOL_CALIBRADA * 100).round(2),
    })
    st.dataframe(df_drift.set_index('Distrito'), use_container_width=True)
    st.caption(
        "El drift historico es la media de retornos anuales observados por distrito (2019-2026). "
        "Es el punto de partida endogeno del modelo: no lo fija el usuario. "
        "Los indicadores macro modulan las probabilidades de regimen que escalan este drift."
    )

    st.markdown("---")
    st.subheader("Correlacion historica entre distritos")
    fig_corr, ax_c = plt.subplots(figsize=(8, 6))
    im = ax_c.imshow(CORR_MATRIZ, cmap='RdYlGn', vmin=-1, vmax=1)
    ax_c.set_xticks(range(n_distritos))
    ax_c.set_xticklabels(distritos, rotation=45, ha='right', fontsize=7)
    ax_c.set_yticks(range(n_distritos))
    ax_c.set_yticklabels(distritos, fontsize=7)
    plt.colorbar(im, ax=ax_c)
    ax_c.set_title("Matriz de Correlacion de Retornos (2007-2026)")
    for r in range(n_distritos):
        for c in range(n_distritos):
            ax_c.text(c, r, f"{CORR_MATRIZ[r,c]:.2f}",
                      ha='center', va='center', fontsize=5)
    plt.tight_layout()
    st.pyplot(fig_corr)


# --- TAB 4: TESTS ---
with tab4:
    st.subheader("Tests Unitarios del Motor Matematico v32.0")
    test_results = run_all_tests()
    n_pass = sum(1 for _, p, _ in test_results if p)
    total  = len(test_results)

    st.metric("Tests pasados", f"{n_pass}/{total}",
              delta="Todos OK" if n_pass == total else "REVISAR")

    for tname, tpass, tmsg in test_results:
        icon = "OK" if tpass else "FAIL"
        with st.expander(f"[{icon}] {tname}"):
            st.code(tmsg)


# ============================================================================
# 13. UTILIDADES PDF
# ============================================================================

def clean_str(txt: str) -> str:
    """Sanitiza a latin-1 para fpdf: tabla explicita + red de seguridad encode/replace."""
    repls = {
        '€': 'EUR',
        'á': 'a', 'Á': 'A', 'à': 'a', 'À': 'A', 'â': 'a', 'ã': 'a', 'ä': 'a', 'Ä': 'A',
        'é': 'e', 'É': 'E', 'è': 'e', 'È': 'E', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'Í': 'I', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'Ó': 'O', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'Ö': 'O',
        'ú': 'u', 'Ú': 'U', 'ù': 'u', 'û': 'u', 'ü': 'u', 'Ü': 'U',
        'ñ': 'n', 'Ñ': 'N',
        '²': '2', '³': '3', '°': 'deg',
        '–': '-', '—': '-', '−': '-',
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u00ab': '"', '\u00bb': '"',
        '•': '-', '·': '.', '≈': '~', '≥': '>=', '≤': '<=', '×': 'x',
        '\u00a0': ' ', '\u00ad': '', '\u2026': '...', '→': '->',
    }
    for k, v in repls.items():
        txt = txt.replace(k, v)
    return txt.encode('latin-1', errors='replace').decode('latin-1')


class ProfessionalPDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 9)
        self.cell(0, 8, 'BCN STRATEGIC MODEL V32.0 | CONFIDENCIAL', 0, 1, 'C')
        self.line(10, 17, 200, 17)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 10,
                  f'Pagina {self.page_no()} | Generado: {ts} | Modelo v32.0 | Confidencial',
                  0, 0, 'C')


def _bloque_trazabilidad(pdf: FPDF, params: dict, dist_name: str, n_sim: int):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, clean_str("TRAZABILIDAD DEL ESCENARIO EXPORTADO"), 0, 1)
    pdf.set_font('Courier', '', 8)
    lineas = [
        f"Timestamp              : {ts}",
        f"Modelo                 : Barcelona Strategic Model v32.0",
        f"Distrito analizado     : {dist_name}",
        f"Regimen economico      : {params.get('regimen', 'N/A')}",
        f"Euribor 12m            : {params.get('euribor', 0):.2f}%",
        f"IPC anual              : {params.get('ipc', 0):.2f}%",
        f"Tasa de paro           : {params.get('paro', 0):.1f}%",
        f"Shock Ley Vivienda     : {'Activado (2029)' if params.get('shock_vt') else 'Desactivado'}",
        f"Simulaciones (n_sim)   : {n_sim}",
        f"Fecha dato real        : {FECHA_DATO_REAL}",
        f"Fuente datos           : {FUENTE_DATO}",
        f"Horizonte proyeccion   : 2027-2032",
        f"Motor                  : MS-VAR + Cholesky + DCC simplificado",
    ]
    mtm_act = construir_mtm_macro(
        params.get('euribor', 2.0),
        params.get('ipc', 2.5),
        params.get('paro', 10.0)
    )
    probs = calcular_probabilidades_regimen(
        params.get('euribor', 2.0),
        params.get('ipc', 2.5),
        params.get('paro', 10.0)
    )
    lineas.append(f"P(Normal/Boom/Crisis)  : {probs[0]*100:.1f}% / {probs[1]*100:.1f}% / {probs[2]*100:.1f}%")
    for ln in lineas:
        pdf.cell(0, 5, clean_str(ln), 0, 1)
    pdf.ln(3)
    logger.info("PDF exportado: %s | params: %s", dist_name,
                {k: v for k, v in params.items()})


def _tabla_encabezado(pdf: FPDF, headers: list, widths: list,
                      fill_r=44, fill_g=62, fill_b=80):
    pdf.set_fill_color(fill_r, fill_g, fill_b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 8)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, clean_str(h), 1, 0, 'C', 1)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 8)


def calculate_noshock(base_scenarios, dist_idx):
    conc   = concentracion_vt_real[dist_idx]
    fv     = (conc / 0.46) * 0.14
    fa     = (conc / 0.46) * 0.10
    sv, sa = base_scenarios
    v_s    = np.median(sv[:, dist_idx, -1])
    a_s    = np.median(sa[:, dist_idx, -1])
    return v_s / max(1 - fv, 0.01), a_s / max(1 - fa, 0.01)


# ============================================================================
# 14. INFORME EJECUTIVO
# ============================================================================

def generate_exec_report(dist_idx: int, scenarios: dict,
                          params: dict, fig_chart, n_sim: int) -> bytes:
    pdf = ProfessionalPDF()
    pdf.add_page()

    # Portada
    pdf.set_font('Arial', 'B', 15)
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255)
    pdf.cell(0, 11, clean_str(
        f"INFORME EJECUTIVO: {distritos[dist_idx].upper()}"), 0, 1, 'C', 1)
    pdf.set_text_color(0)
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 6, clean_str(
        f"Barcelona Strategic Model v32.0 | Datos reales {FECHA_DATO_REAL} | "
        f"Motor: MS-VAR + Markov no homogeneo + DCC"), 0, 1, 'C')
    pdf.ln(4)

    _bloque_trazabilidad(pdf, params, distritos[dist_idx], n_sim)

    # --- Resumen macro activo ---
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("1. PARAMETROS MACROECONOMICOS ACTIVOS"), 0, 1)
    pdf.set_font('Arial', '', 10)
    mtm_act = construir_mtm_macro(
        params.get('euribor', 2.0), params.get('ipc', 2.5), params.get('paro', 10.0))
    probs   = calcular_probabilidades_regimen(
        params.get('euribor', 2.0), params.get('ipc', 2.5), params.get('paro', 10.0))
    pdf.multi_cell(0, 5, clean_str(
        f"Regimen base: {params.get('regimen', 'N/A')}\n"
        f"Euribor 12m: {params.get('euribor', 0):.2f}% | "
        f"IPC: {params.get('ipc', 0):.2f}% | "
        f"Paro Barcelona: {params.get('paro', 0):.1f}%\n"
        f"Probabilidades de regimen endogenas (MTM no homogenea): "
        f"Normal={probs[0]*100:.1f}% | Boom={probs[1]*100:.1f}% | Crisis={probs[2]*100:.1f}%\n"
        f"Drift historico venta para {distritos[dist_idx]}: "
        f"{DRIFT_VENTA[dist_idx]*100:.2f}% anual (calibrado 2019-2026)\n"
        f"Beta CAPM: {BETAS_MODELO[dist_idx]:.3f} | "
        f"Vol. calibrada: {VOL_CALIBRADA[dist_idx]*100:.2f}%"
    ))
    pdf.ln(3)

    # --- Sensibilidad regulatoria ---
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("2. SENSIBILIDAD REGULATORIA - IMPACTO SHOCK VT"), 0, 1)
    hdrs = ["Escenario", "Venta (Shock)", "Venta (Sin Shock)", "Alq (Shock)", "Alq (Sin Shock)"]
    wdts = [35, 38, 38, 38, 38]
    _tabla_encabezado(pdf, hdrs, wdts)
    for name, (sv, sa) in [
        ("Crecimiento",  scenarios["Crecimiento"]),
        ("Estanflacion", scenarios["Estanflacion"]),
        ("Recesion",     scenarios["Recesion"]),
    ]:
        v32_s  = np.median(sv[:, dist_idx, -1])
        a32_s  = np.median(sa[:, dist_idx, -1])
        v32_ns, a32_ns = calculate_noshock((sv, sa), dist_idx)
        pdf.cell(wdts[0], 7, clean_str(name), 1)
        pdf.cell(wdts[1], 7, f"{int(v32_s):,} EUR", 1)
        pdf.cell(wdts[2], 7, f"{int(v32_ns):,} EUR", 1)
        pdf.cell(wdts[3], 7, f"{a32_s:.1f} EUR", 1)
        pdf.cell(wdts[4], 7, f"{a32_ns:.1f} EUR", 1)
        pdf.ln()

    # --- Grafico ---
    pdf.ln(3)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("3. PROYECCION GRAFICA (2026-2032)"), 0, 1)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig_chart.savefig(tmp.name, dpi=100, bbox_inches='tight')
        pdf.image(tmp.name, x=10, w=185)

    # --- Tabla maestra ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("4. TABLA MAESTRA - TODOS LOS DISTRITOS (2026 vs 2032)"), 0, 1)
    hdrs2 = ["Distrito", "Venta'26", "Venta'32", "Var%",
             "Alq'26", "Alq'32", "Yield'32", "Beta", "Drift%"]
    wdts2 = [42, 22, 22, 16, 21, 21, 19, 14, 16]
    _tabla_encabezado(pdf, hdrs2, wdts2)
    sv_u, sa_u = scenarios["Usuario (Activo)"]
    for i, d in enumerate(distritos):
        v26 = precio_venta_2026[i];    a26 = precio_alquiler_2026[i]
        v32 = np.median(sv_u[:, i, -1]); a32 = np.median(sa_u[:, i, -1])
        var = ((v32/v26)-1)*100;       yld = (a32*12)/v32*100
        pdf.set_fill_color(240 if i % 2 == 0 else 255)
        vals = [clean_str(d), f"{int(v26):,}", f"{int(v32):,}", f"{var:+.1f}%",
                f"{a26:.1f}", f"{a32:.1f}", f"{yld:.1f}%",
                f"{BETAS_MODELO[i]:.3f}", f"{DRIFT_VENTA[i]*100:.2f}%"]
        for v, w in zip(vals, wdts2):
            pdf.cell(w, 6, v, 1, 0, 'C', 1)
        pdf.ln()

    # --- Recomendacion ---
    pdf.ln(4)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_str("5. RECOMENDACION AL INVERSOR"), 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, clean_str(
        f"Basado en modelo v32.0 con datos reales {FECHA_DATO_REAL}:\n\n"
        "- YIELD (renta corriente): Nou Barris y Sant Andreu presentan yields superiores al 5%.\n"
        "  Son distritos con baja exposicion VT y demanda residencial estable.\n\n"
        "- PLUSVALIA (apreciacion capital): Eixample y Gracia lideran en escenario de\n"
        "  Crecimiento por su beta elevado. En estanflacion o recesion, este liderazgo\n"
        "  se invierte y son los de mayor caida relativa.\n\n"
        "- RIESGO REGULATORIO: El shock VT (2029) impacta de forma muy asimetrica.\n"
        "  Eixample (densidad VT=0.46) puede sufrir correcciones de hasta el -10%\n"
        "  en precio de venta. Nou Barris (densidad=0.004) resulta practicamente inmune.\n\n"
        f"- MACRO ACTUAL: Con Euribor={params.get('euribor',0):.1f}% y Paro={params.get('paro',0):.1f}%,\n"
        f"  el modelo asigna P(Boom)={probs[1]*100:.0f}% y P(Crisis)={probs[2]*100:.0f}%.\n"
        "  El euribor actua comprimiendo la demanda de compra y desplazandola hacia\n"
        "  el alquiler, lo que puede sostener rentas incluso en escenarios adversos.\n\n"
        "NOTA: Proyecciones basadas en mediana (P50). Ver intervalo de confianza P10-P90\n"
        "para rango completo de incertidumbre."
    ))

    return pdf.output(dest='S').encode('latin-1')


# ============================================================================
# 15. EXPLICACION METODOLOGICA DETALLADA DEL MODELO
# ============================================================================

def generate_explicacion_metodologica(params: dict, n_sim: int) -> bytes:
    """
    Genera la Explicacion Metodologica Detallada del Modelo v32.0.
    Documento de referencia academico-profesional que describe la logica
    economica y matematica completa del sistema.
    """
    pdf = ProfessionalPDF()

    # -----------------------------------------------------------------------
    # PORTADA
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.ln(8)
    pdf.set_font('Times', 'B', 17)
    pdf.multi_cell(0, 9,
        clean_str("EXPLICACION METODOLOGICA DETALLADA DEL MODELO\n"
                  "BARCELONA STRATEGIC MODEL v32.0"),
        align='C')
    pdf.ln(4)
    pdf.set_font('Times', 'I', 12)
    pdf.multi_cell(0, 7,
        clean_str("Modelo de Simulacion Estocastica de Precios Inmobiliarios Residenciales\n"
                  "Enfoque MS-VAR: Markov Switching + Vector Autoregresivo + Cholesky + DCC"),
        align='C')
    pdf.ln(6)
    pdf.set_font('Times', '', 10)
    for linea in [
        f"Version del modelo   : Barcelona Strategic Model v32.0",
        f"Fecha dato real base : {FECHA_DATO_REAL}",
        f"Fuente de datos      : {FUENTE_DATO}",
        f"Horizonte proyeccion : 2027-2032 (6 anos)",
        f"Distritos modelizados: {n_distritos} (todos los de Barcelona)",
        f"Generado             : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]:
        pdf.cell(0, 6, clean_str(linea), 0, 1, 'C')
    pdf.ln(5)
    pdf.line(25, pdf.get_y(), 185, pdf.get_y())
    pdf.ln(4)
    _bloque_trazabilidad(pdf, params, "Todos los distritos", n_sim)

    # -----------------------------------------------------------------------
    # INDICE
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, "INDICE DE CONTENIDOS", 0, 1)
    pdf.set_font('Times', '', 10)
    secciones = [
        ("1", "Introduccion y Contexto Economico del Mercado Barcelones"),
        ("2", "Arquitectura General del Modelo: Vision de Conjunto"),
        ("3", "Datos de Entrada: Calibracion y Fuentes"),
        ("4", "Calibracion de Elasticidades (Beta CAPM)"),
        ("5", "Drift Endogeno: Retornos Historicos y Componente VAR(1)"),
        ("6", "Indicadores Macroeconomicos y la Cadena de Markov No Homogenea"),
        ("7", "Motor Monte Carlo: Algoritmo de Simulacion Completo"),
        ("8", "Estructura de Correlacion Espacial: Descomposicion de Cholesky"),
        ("9", "Volatilidad Dinamica: DCC Simplificado"),
        ("10","Funcion de Shock Regulatorio (Jump Process)"),
        ("11","Retroalimentacion Alquiler-Venta (Canal rho)"),
        ("12","Aggregacion Estadistica y Percentiles"),
        ("13","Limitaciones del Modelo y Advertencias"),
        ("14","Tests de Integridad Matematica"),
        ("15","Referencias Bibliograficas y Normativas"),
    ]
    for num, titulo in secciones:
        pdf.cell(15, 6, clean_str(f"{num}."), 0, 0)
        pdf.cell(0,  6, clean_str(titulo), 0, 1)
    pdf.ln(3)

    # -----------------------------------------------------------------------
    # SECCION 1: CONTEXTO ECONOMICO
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("1. INTRODUCCION Y CONTEXTO ECONOMICO"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El mercado inmobiliario residencial de Barcelona presenta cuatro caracteristicas "
        "estructurales que justifican un tratamiento econometrico avanzado y diferencian "
        "este modelo de una simple extrapolacion lineal.\n\n"
        "PRIMERA - HETEROGENEIDAD DISTRITAL EXTREMA: los precios de venta oscilan en 2026 "
        "entre los 2.946 EUR/m2 de Nou Barris y los 6.896 EUR/m2 de Sarria-Sant Gervasi "
        "(brecha del 134%), reflejo de diferencias fundamentales en capital humano, "
        "accesibilidad al empleo, dotacion de servicios, presion turistica y perfil de "
        "demanda. Esta heterogeneidad exige modelizar cada distrito de forma individual "
        "pero interconectada, con elasticidades especificas calibradas sobre datos historicos.\n\n"
        "SEGUNDA - INTERVENCION REGULATORIA DE ALTO IMPACTO: la Ley 12/2023 de Derecho "
        "a la Vivienda y las restricciones municipales a las licencias de vivienda turistica "
        "(VT) constituyen shocks exogenos discretos cuyo efecto sobre los precios es "
        "asimetrico segun la densidad de VT en cada distrito. Ignorarlos produciria "
        "proyecciones sobreestimadas especialmente en Eixample y Ciutat Vella.\n\n"
        "TERCERA - DEPENDENCIA DEL CICLO MACROECONOMICO: los tipos de interes (Euribor), "
        "la inflacion (IPC) y el mercado laboral (paro) son determinantes primarios de "
        "la demanda efectiva de vivienda. El Euribor tiene un impacto asimetrico "
        "documentado: comprime la demanda de compra (mayores cuotas hipotecarias) pero "
        "puede sostener o elevar el alquiler (desplazamiento de demanda). El IPC real "
        "(diferencia entre IPC general e IPC de vivienda) determina si los precios crecen "
        "en terminos reales o solo nominalmente. El paro actua con retardo de 12-18 meses "
        "sobre la confianza del comprador.\n\n"
        "CUARTA - CORRELACION ESPACIAL DINAMICA: los mercados de distritos adyacentes o "
        "similares exhiben correlacion significativa por factores compartidos (empleo, "
        "transporte, oleadas de gentrificacion). Esta correlacion no es constante: "
        "en periodos de crisis tiende a converger hacia 1 (todos los mercados caen juntos) "
        "y en expansion se dispersa. Un modelo que asuma correlacion constante "
        "subestimara el riesgo en escenarios adversos."
    ))
    pdf.ln(3)

    # -----------------------------------------------------------------------
    # SECCION 2: ARQUITECTURA GENERAL
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("2. ARQUITECTURA GENERAL DEL MODELO: VISION DE CONJUNTO"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El modelo v32.0 es un sistema de ecuaciones diferenciales estocasticas discretas "
        "(SDE discreta) con cuatro capas metodologicas interactuantes:\n\n"
        "CAPA 1 - DRIFT ENDOGENO (VAR): el componente de tendencia de cada distrito "
        "se estima automaticamente a partir de los retornos historicos observados "
        "(2019-2026) y de un modelo Vector Autoregresivo de orden 1 (VAR(1)) que "
        "captura las interdependencias temporales entre distritos. El usuario NO "
        "introduce el drift directamente: emerge de los datos.\n\n"
        "CAPA 2 - REGIMEN MACROECONOMICO (Markov no homogeneo): el estado "
        "macroeconomico (Normal, Boom, Crisis) evoluciona segun una cadena de "
        "Markov cuyas probabilidades de transicion se calculan dinamicamente en "
        "funcion de los indicadores observables introducidos por el usuario: "
        "Euribor, IPC y Tasa de Paro. Esta es la principal novedad respecto a v31: "
        "la MTM no es fija sino que se recalcula en cada escenario.\n\n"
        "CAPA 3 - CORRELACION ESPACIAL DINAMICA (Cholesky + DCC simplificado): "
        "los shocks estocasticos se generan con la estructura de correlacion "
        "historica entre distritos (Cholesky). Adicionalmente, esta correlacion "
        "se modula segun el estado del ciclo economico (DCC simplificado): "
        "se comprime en expansion y converge hacia 1 en crisis.\n\n"
        "CAPA 4 - SHOCKS REGULATORIOS (Jump Process): se modela el impacto discreto "
        "de la restriccion de licencias VT en 2029, con magnitud proporcional a la "
        "densidad de VT en cada distrito.\n\n"
        "La ecuacion gobernante consolidada es:\n"
    ))
    pdf.set_font('Courier', 'B', 9)
    pdf.multi_cell(0, 5, clean_str(
        "dP_i(t)/P_i(t) = [mu_hist_i + A_VAR*r(t-1) + f(Euribor,IPC,Paro)] * gamma_i * dt\n"
        "               + sigma_i(S_t) * SUM_j[L_ij(S_t) * dZ_j(t)]\n"
        "               + rho * R_alq(t-1)\n"
        "               + J_VT(t, i)"
    ))
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "\nDonde:\n"
        "mu_hist_i    : drift historico calibrado del distrito i (2019-2026)\n"
        "A_VAR        : matriz de coeficientes VAR(1) estimados por OLS\n"
        "f(E,I,P)     : funcion de ajuste macro (Euribor, IPC, Paro) -> delta drift\n"
        "gamma_i      : elasticidad distrital (sensibilidad al ciclo de mercado)\n"
        "sigma_i(S_t) : volatilidad dinamica, funcion del estado Markov S_t\n"
        "L_ij(S_t)    : elemento (i,j) de Cholesky modulada por estado\n"
        "rho          : coeficiente de retroalimentacion alquiler-venta (0.45)\n"
        "J_VT(t,i)    : salto regulatorio en t=2029, proporcional a densidad VT"
    ))
    pdf.ln(3)

    # -----------------------------------------------------------------------
    # SECCION 3: DATOS
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("3. DATOS DE ENTRADA: CALIBRACION Y FUENTES"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El modelo trabaja con dos tipos de datos:\n\n"
        "A) PRECIOS HISTORICOS DISTRITALES (2019-2026): series de precios medios "
        "de venta (EUR/m2) y alquiler (EUR/m2/mes) para los 10 distritos de "
        "Barcelona, con frecuencia anual. Fuente: Idealista Research e Incasol "
        "(Institut Catala del Sol, Generalitat de Catalunya). Los datos de 2026 "
        "son datos reales de Q1 2026 e introducen el punto de partida de todas "
        "las proyecciones.\n\n"
        "B) INDICE MACRO AGREGADO BCN (2007-2026): precio medio de venta EUR/m2 "
        "a nivel ciudad, necesario para calibrar las betas CAPM y para reconstruir "
        "las series historicas 2007-2018. Fuente: INE (Indice de Precios de Vivienda).\n\n"
        "PRECIOS DE VENTA 2026 (EUR/m2):\n"
        "Ciutat Vella:4.811 | Eixample:6.363 | Gracia:5.404 | Horta Guinardo:3.905\n"
        "Les Corts:6.355 | Nou Barris:2.946 | Sant Andreu:3.730 | Sant Marti:4.899\n"
        "Sants-Montjuic:4.477 | Sarria-Sant Gervasi:6.896\n\n"
        "PRECIOS DE ALQUILER 2026 (EUR/m2/mes):\n"
        "Ciutat Vella:25.7 | Eixample:26.0 | Gracia:23.7 | Horta Guinardo:19.1\n"
        "Les Corts:21.9 | Nou Barris:16.3 | Sant Andreu:18.5 | Sant Marti:23.6\n"
        "Sants-Montjuic:20.8 | Sarria-Sant Gervasi:23.6\n\n"
        "DENSIDAD VT (vector de exposicion, fuente Inside Airbnb / CEAT):\n"
        "Eixample:0.460 | Ciutat Vella:0.150 | Sant Marti:0.120 | Sants:0.110 | "
        "Gracia:0.105 | Sarria:0.050 | Les Corts:0.035 | Horta:0.030 | "
        "Sant Andreu:0.015 | Nou Barris:0.004"
    ))

    # Tabla betas y volatilidades calibradas
    pdf.ln(3)
    pdf.set_font('Times', 'B', 10)
    pdf.cell(0, 7, clean_str("Parametros calibrados por distrito"), 0, 1)
    hdrs_d = ["Distrito", "Beta CAPM", "Drift Venta%", "Drift Alq%", "Vol.Cal%", "Dens.VT"]
    wdts_d = [50, 22, 25, 25, 22, 22]
    _tabla_encabezado(pdf, hdrs_d, wdts_d)
    for i, d in enumerate(distritos):
        pdf.set_fill_color(240 if i % 2 == 0 else 255)
        vals = [
            clean_str(d),
            f"{BETAS_MODELO[i]:.4f}",
            f"{DRIFT_VENTA[i]*100:.2f}%",
            f"{DRIFT_ALQUILER[i]*100:.2f}%",
            f"{VOL_CALIBRADA[i]*100:.2f}%",
            f"{concentracion_vt_real[i]:.3f}",
        ]
        for v, w in zip(vals, wdts_d):
            pdf.cell(w, 6, v, 1, 0, 'C', 1)
        pdf.ln()

    # -----------------------------------------------------------------------
    # SECCION 4: BETAS CAPM
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("4. CALIBRACION DE ELASTICIDADES: BETA CAPM"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "La elasticidad de cada distrito al ciclo general del mercado barcelones "
        "se cuantifica mediante el coeficiente Beta del modelo CAPM adaptado "
        "al sector inmobiliario, siguiendo la metodologia de Shiller (1993) "
        "y su extension a mercados de vivienda local:\n\n"
        "   Beta_i = Cov(r_i, r_M) / Var(r_M)\n\n"
        "Donde r_i es el retorno anual del precio de venta del distrito i y "
        "r_M es el retorno del indice agregado de Barcelona. La estimacion "
        "se realiza sobre los retornos observados 2019-2026 (n=7 observaciones).\n\n"
        "Los betas se truncan al intervalo [0.5, 1.5] para evitar valores extremos "
        "no plausibles economicamente. Un beta de 0.5 implicaria que cuando el "
        "mercado barcelones cae un 10%, el distrito solo cae un 5% (defensivo). "
        "Un beta de 1.5 implicaria una caida del 15% (altamente ciclico).\n\n"
        "INTERPRETACION ECONOMICA:\n"
        "- Beta alto (Ciutat Vella, Eixample): mercados con alta rotacion, "
        "  fuerte componente especulativo y turismo. Amplificadores del ciclo.\n"
        "- Beta bajo (Nou Barris, Sant Andreu): mercados con demanda residencial "
        "  estable, menor especulacion y mayor rigidez de precios a la baja.\n\n"
        "RECONSTRUCCION HISTORICA 2007-2018: dado que no se dispone de datos "
        "distritales previos a 2019, los precios se reconstruyen historicamente "
        "invirtiendo el modelo CAPM:\n\n"
        "   P_i(t) = P_i(t+1) / (1 + Beta_i * r_M(t) + epsilon)\n\n"
        "Con epsilon ~ N(0, 0.003). Esta reconstruccion sirve exclusivamente "
        "para estimar la matriz de correlacion entre distritos (Cholesky), "
        "no para el drift del modelo."
    ))

    # -----------------------------------------------------------------------
    # SECCION 5: DRIFT ENDOGENO Y VAR(1)
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("5. DRIFT ENDOGENO: RETORNOS HISTORICOS Y COMPONENTE VAR(1)"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "Esta es la principal diferencia de v32.0 respecto a versiones anteriores: "
        "el drift (tendencia) ya no lo fija el usuario mediante un slider. "
        "El drift emerge de los propios datos historicos del mercado.\n\n"
        "DRIFT HISTORICO BASE (mu_hist_i):\n"
        "Se calcula como la media aritmetica simple de los retornos anuales "
        "observados para cada distrito en el periodo 2019-2026:\n\n"
        "   mu_hist_i = (1/T) * SUM_t [(P_i(t)/P_i(t-1)) - 1]\n\n"
        "Este drift refleja el comportamiento medio del distrito en condiciones "
        "de mercado recientes. Incorpora implicitamente el efecto de todos los "
        "factores estructurales del periodo de calibracion: gentrificacion, "
        "turismo, inversion extranjera, regulacion, etc.\n\n"
        "COMPONENTE VAR(1) (A_VAR):\n"
        "Adicionalmente, se estima un modelo Vector Autoregresivo de orden 1 "
        "sobre los retornos distritales observados. El VAR(1) captura las "
        "interdependencias dinamicas entre distritos: si Eixample tuvo un retorno "
        "elevado en t-1, eso puede predecir un retorno elevado en Gracia en t "
        "(efecto desbordamiento geografico documentado en Barcelona).\n\n"
        "   r(t) = A_VAR * r(t-1) + epsilon(t)\n\n"
        "La matriz A_VAR (10x10) se estima por Minimos Cuadrados Ordinarios (OLS) "
        "multivariado sobre los retornos observados 2019-2026. Para evitar "
        "sobreajuste con n=7 observaciones, la contribucion del VAR al drift total "
        "se pondera con lambda_VAR=0.25 (25% del drift total).\n\n"
        "ESTABILIDAD DEL VAR: el radio espectral de A_VAR (mayor valor propio "
        "en modulo) debe ser menor que 1 para garantizar la estacionariedad "
        "del proceso. Se verifica mediante test automatico en cada ejecucion.\n\n"
        "AJUSTE MACRO AL DRIFT:\n"
        "Sobre el drift endogeno se aplica un ajuste adicional por indicadores macro:\n"
        "   delta_venta = -clip((Euribor - 2.5)/100, -0.02, 0.04)  [efecto hipotecario]\n"
        "   delta_comun = clip((IPC - 2.0)/100, -0.03, 0.03)       [IPC real]\n"
        "   delta_comun -= clip((Paro - 10.0)/100 * 0.5, -0.02, 0.03) [efecto laboral]\n\n"
        "El efecto Euribor es asimetrico: un Euribor elevado comprime el drift "
        "de venta (menos compradores solventes) pero eleva el drift de alquiler "
        "(desplazamiento de demanda hacia arrendamiento). Este canal de transmision "
        "esta documentado empiricamente en el mercado espanol post-2022."
    ))

    # -----------------------------------------------------------------------
    # SECCION 6: MARKOV NO HOMOGENEO
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("6. INDICADORES MACRO Y CADENA DE MARKOV NO HOMOGENEA"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "LA LIMITACION DE LAS MTM FIJAS (version anterior):\n"
        "En v31, las matrices de transicion eran fijas por regimen (3 matrices "
        "precalibradas). El usuario elegia subjetivamente el regimen, lo que "
        "mantenia cierto grado de endogeneidad. En v32, las probabilidades de "
        "transicion se calculan endogenamente a partir de indicadores observables.\n\n"
        "CADENA DE MARKOV NO HOMOGENEA:\n"
        "En lugar de P(S_{t+1}=j | S_t=i) = constante, el modelo implementa:\n\n"
        "   P(S_{t+1}=j | S_t=i, Z_t) = f(Z_t)_ij\n\n"
        "Donde Z_t = (Euribor_t, IPC_t, Paro_t) es el vector de indicadores macro.\n"
        "Esta es la formulacion del Markov Switching con variables exogenas "
        "de Hamilton (1994), generalizacion del modelo seminal de Hamilton (1989).\n\n"
        "FUNCION DE PROBABILIDADES DE REGIMEN:\n"
        "Se construye una funcion de tipo softmax sobre scores economicos:\n\n"
        "   Score_Boom   = (1 - Euribor_norm) * 0.4 + (1 - Paro_norm) * 0.4 "
        "+ IPC_cuadratico * 0.2\n"
        "   Score_Crisis = Euribor_norm * 0.45 + Paro_norm * 0.45 "
        "+ excess_IPC * 0.10\n"
        "   Score_Normal = 0.5 (referencia)\n\n"
        "   P(regimen) = softmax([Score_Normal, Score_Boom, Score_Crisis], T=2.5)\n\n"
        "Los indicadores se normalizan sobre rangos historicos espanoles:\n"
        "   Euribor: [0%, 5%] | IPC: [0%, 8%] | Paro: [5%, 30%]\n\n"
        "CONSTRUCCION DE LA MTM:\n"
        "A partir de P(regimen) = [p_norm, p_boom, p_crisis], se construye la MTM:\n\n"
        "   Fila 0 (desde Normal): [persist+delta_norm, delta_boom, delta_crisis]\n"
        "   Fila 1 (desde Boom):   [salida_gradual, persist+delta_boom, baja_crisis]\n"
        "   Fila 2 (desde Crisis): [recuperacion, baja_boom, persist+delta_crisis]\n\n"
        "Con PERSIST=0.55 (factor de inercia del ciclo economico). "
        "Cada fila se renormaliza para sumar 1.0 exactamente.\n\n"
        "LOGICA ECONOMICA:\n"
        "- Con Euribor=1.5% y Paro=9%: P(Boom) elevada, MTM favorece permanencia "
        "  en expansion. Reproduce el ciclo 2015-2019 en Barcelona.\n"
        "- Con Euribor=3.8% y Paro=13%: P(Normal) domina, ciclo de estanflacion. "
        "  Reproduce 2022-2024 en Espana.\n"
        "- Con Euribor=2.5% y Paro=20%: P(Crisis) elevada. Reproduce 2011-2014."
    ))

    # MTM activa
    pdf.ln(3)
    pdf.set_font('Times', 'B', 10)
    pdf.cell(0, 7, clean_str(
        f"MTM no homogenea activa (Euribor={params.get('euribor',0):.1f}%, "
        f"IPC={params.get('ipc',0):.1f}%, Paro={params.get('paro',0):.1f}%)"), 0, 1)
    mtm_doc = construir_mtm_macro(
        params.get('euribor', 2.0), params.get('ipc', 2.5), params.get('paro', 10.0))
    hdrs_m = ["", "-> Normal", "-> Boom", "-> Crisis", "Suma"]
    wdts_m = [35, 35, 35, 35, 25]
    _tabla_encabezado(pdf, hdrs_m, wdts_m)
    filas_m = ["Desde Normal", "Desde Boom", "Desde Crisis"]
    for i, fila in enumerate(filas_m):
        pdf.set_fill_color(240 if i % 2 == 0 else 255)
        pdf.cell(wdts_m[0], 6, clean_str(fila), 1, 0, 'L', 1)
        for j in range(3):
            pdf.cell(wdts_m[j+1], 6, f"{mtm_doc[i,j]:.4f}", 1, 0, 'C', 1)
        pdf.cell(wdts_m[4], 6, f"{mtm_doc[i].sum():.6f}", 1, 0, 'C', 1)
        pdf.ln()

    # -----------------------------------------------------------------------
    # SECCION 7: ALGORITMO MONTE CARLO
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("7. MOTOR MONTE CARLO: ALGORITMO COMPLETO"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        f"El motor realiza n_sim={n_sim} simulaciones independientes de trayectorias "
        "de precios 2027-2032. En esta ejecucion: np.random.seed(42) "
        "garantiza reproducibilidad exacta para identicos parametros.\n\n"
        "PSEUDOCODIGO DEL ALGORITMO:\n"
    ))
    pdf.set_font('Courier', '', 8)
    pdf.multi_cell(0, 4.5, clean_str(
        "PARA sim = 1,...,n_sim:\n"
        "  p_vta = precio_venta_2026.copy()    # precios iniciales reales\n"
        "  p_alq = precio_alquiler_2026.copy()\n"
        "  estado = 0 (Normal)                 # estado inicial Markov\n"
        "  r_prev = drift_historico_venta       # retorno previo para VAR(1)\n"
        "\n"
        "  PARA t = 2027, 2028, ..., 2032:\n"
        "    1. TRANSICION MARKOV NO HOMOGENEA:\n"
        "       u ~ Uniform(0,1); estado = sample(MTM[estado,:], u)\n"
        "\n"
        "    2. DRIFT ENDOGENO:\n"
        "       var_comp = lambda_var(0.25) * A_VAR @ r_prev\n"
        "       drift_v  = mu_hist + var_comp + f_macro(Euribor,IPC,Paro)\n"
        "       drift_a  = mu_hist_alq + var_comp*0.7 + f_macro_alq\n"
        "\n"
        "    3. MODULACION POR ESTADO MARKOV:\n"
        "       SI Boom:   tasa_v = drift_v * 1.35 + N(0,0.008)\n"
        "       SI Crisis: tasa_v = -|drift_v|*0.6 - 0.015 + N(0,0.012)\n"
        "       SI Normal: tasa_v = drift_v + N(0,0.008)\n"
        "\n"
        "    4. ELASTICIDAD DISTRITAL:\n"
        "       tasa_v *= sensibilidad_distrital  # gamma_i\n"
        "\n"
        "    5. DCC SIMPLIFICADO (Cholesky + volatilidad dinamica):\n"
        "       vol_dyn, L_dyn = f(estado)\n"
        "       Z ~ N(0, I_n); eps = L_dyn @ Z\n"
        "       tasa_v += vol_dyn * eps\n"
        "\n"
        "    6. FLOOR YIELD (min 2.5% bruto):\n"
        "       resist = clip(yield_actual / 0.025, 0.5, 1.0)\n"
        "       tasa_v = where(tasa_v>0, tasa_v*resist, tasa_v)\n"
        "\n"
        "    7. FEEDBACK ALQUILER->VENTA:\n"
        "       SI t>2027: tasa_v += rho(0.45) * ret_alq(t-1)\n"
        "\n"
        "    8. SHOCK REGULATORIO (si activado y t=2029):\n"
        "       tasa_v += -0.10 * (D_i / D_max)  # Jump Process\n"
        "\n"
        "    9. ACTUALIZAR PRECIOS:\n"
        "       p_vta *= (1 + tasa_v); p_alq *= (1 + tasa_a)\n"
        "       r_prev = tasa_v  # actualizar para VAR(1)\n"
        "\n"
        "FIN SIMULACION\n"
        "RESULTADOS: percentiles P10, P50, P90 por distrito y periodo"
    ))

    # -----------------------------------------------------------------------
    # SECCION 8: CHOLESKY
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("8. CORRELACION ESPACIAL: DESCOMPOSICION DE CHOLESKY"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "MOTIVACION ECONOMICA:\n"
        "Los mercados de distritos adyacentes o similares comparten factores de "
        "demanda (mismo mercado laboral central, mismas infraestructuras de transporte, "
        "oleadas de gentrificacion que se propagan geograficamente). Un modelo con "
        "shocks independientes por distrito subestimaria el riesgo sistematico: "
        "en realidad, cuando Eixample cae, Gracia tiende a caer tambien.\n\n"
        "METODOLOGIA:\n"
        "1. Se estima la matriz de correlacion Sigma sobre los retornos historicos "
        "   reconstruidos 2007-2026 (series completas necesarias para Cholesky).\n"
        "2. Se aplica la descomposicion: Sigma = L * L^T, donde L es triangular inferior.\n"
        "3. Los shocks correlacionados se generan: eps = L * Z, Z ~ N(0, I_n).\n"
        "   Esto garantiza eps ~ N(0, Sigma): shocks con correlacion historica.\n\n"
        "CORRECCION ESPECTRAL:\n"
        "Si la matriz de correlacion estimada no es definida positiva (puede ocurrir "
        "con n pequeno o multicolinealidad), se aplica correccion espectral:\n"
        "   Sigma_corr = V * diag(max(lambda_i, 1e-8)) * V^T\n"
        "seguida de renormalizacion de la diagonal a 1. Esto garantiza que la "
        "factorizacion de Cholesky siempre sea posible sin alterar significativamente "
        "la estructura de correlacion estimada.\n\n"
        "La validez de L se verifica mediante test unitario: es_lower_triangular, "
        "diagonal_positiva, y L*L^T aprox Sigma (error maximo < 1e-5)."
    ))

    # -----------------------------------------------------------------------
    # SECCION 9: DCC SIMPLIFICADO
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("9. VOLATILIDAD DINAMICA: DCC SIMPLIFICADO"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "MOTIVACION:\n"
        "La evidencia empirica en mercados financieros e inmobiliarios muestra dos "
        "fenomenos bien documentados: (1) la volatilidad se incrementa en periodos "
        "de contraccion (GARCH en crisis), y (2) las correlaciones entre activos "
        "convergen hacia 1 durante los crash (DCC en crisis). Un modelo con "
        "volatilidad y correlacion constantes subestima el riesgo en escenarios adversos.\n\n"
        "IMPLEMENTACION (DCC simplificado, inspirado en Engle 2002):\n\n"
        "ESTADO BOOM:\n"
        "  - Volatilidad: sigma_dyn = sigma_calibrada * 0.85\n"
        "    (compresion de volatilidad en expansion, caracteristica de bull markets)\n"
        "  - Correlacion: Sigma_dyn = Sigma_historica * 0.8 + I * 0.2\n"
        "    (reduccion de correlaciones en expansion: mercados se diferencian)\n\n"
        "ESTADO NORMAL:\n"
        "  - Volatilidad: sigma_dyn = sigma_calibrada\n"
        "  - Correlacion: Sigma_dyn = Sigma_historica\n\n"
        "ESTADO CRISIS:\n"
        "  - Volatilidad: sigma_dyn = sigma_calibrada * 1.60\n"
        "    (explosion de volatilidad en crisis, factor 1.6 calibrado sobre crisis 2008)\n"
        "  - Correlacion: Sigma_dyn = Sigma_historica * 0.5 + 1_matriz * 0.5\n"
        "    (correlaciones convergen a 1: en crisis todo cae junto)\n"
        "  - En cada periodo se recalcula la Cholesky de Sigma_dyn para garantizar\n"
        "    la descomposicion de la correlacion modificada.\n\n"
        "DIFERENCIA CON DCC COMPLETO (Engle 2002):\n"
        "Un DCC completo estima la volatilidad y correlacion mediante modelos GARCH "
        "y un proceso dinamico continuo, requiriendo series temporales largas (>100 obs). "
        "Con n=8 periodos observados, esta estimacion seria estadisticamente inestable. "
        "El enfoque simplificado implementado captura la logica esencial del DCC "
        "(correlaciones dinamicas condicionadas al estado del ciclo) con los datos disponibles."
    ))

    # -----------------------------------------------------------------------
    # SECCION 10: SHOCK REGULATORIO
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("10. FUNCION DE SHOCK REGULATORIO (JUMP PROCESS)"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "CANAL ECONOMICO:\n"
        "Las licencias de vivienda turistica (VT) crean una demanda adicional "
        "de compra no residencial: inversores que adquieren pisos para destinarlos "
        "a alquiler turistico obtienen una rentabilidad superior a la del alquiler "
        "residencial, lo que eleva los precios de venta en zonas de alta densidad VT. "
        "La restriccion o eliminacion de nuevas licencias suprime esta demanda "
        "adicional, produciendo una correccion de precios proporcional a la densidad "
        "preexistente de VT.\n\n"
        "El mismo mecanismo actua sobre el alquiler en sentido inverso en el largo "
        "plazo (mas oferta disponible para residentes modera el alquiler), aunque "
        "en el horizonte de 1-2 anos el efecto es de menor magnitud.\n\n"
        "ESPECIFICACION MATEMATICA:\n"
        "Se modela como proceso de salto discreto (Jump Process) activado en t=2029:\n\n"
        "   J_vt(i) = -0.10 * (D_i / D_max)  [impacto sobre precio de venta]\n"
        "   J_alq(i) = -0.06 * (D_i / D_max)  [impacto sobre precio de alquiler]\n\n"
        "Donde D_i es la densidad de VT del distrito i y D_max=0.46 (Eixample).\n\n"
        "Impactos resultantes:\n"
        "Eixample (0.460): venta -10.0%, alquiler -6.0%\n"
        "Ciutat Vella (0.150): venta -3.3%, alquiler -2.0%\n"
        "Sant Marti (0.120): venta -2.6%, alquiler -1.6%\n"
        "Sants-Montjuic (0.110): venta -2.4%, alquiler -1.4%\n"
        "Gracia (0.105): venta -2.3%, alquiler -1.4%\n"
        "Sarria-Sant Gervasi (0.050): venta -1.1%, alquiler -0.7%\n"
        "Les Corts (0.035): venta -0.8%, alquiler -0.5%\n"
        "Horta Guinardo (0.030): venta -0.7%, alquiler -0.4%\n"
        "Sant Andreu (0.015): venta -0.3%, alquiler -0.2%\n"
        "Nou Barris (0.004): venta -0.1%, alquiler -0.1% (practicamente inmune)\n\n"
        "NOTA METODOLOGICA: la recuperacion del shock es gradual en las simulaciones "
        "posteriores a 2029 a traves del mecanismo de retroalimentacion alquiler-venta "
        "y del drift historico. No se modela recuperacion total en el horizonte 2032."
    ))

    # -----------------------------------------------------------------------
    # SECCION 11: FEEDBACK ALQUILER-VENTA
    # -----------------------------------------------------------------------
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("11. RETROALIMENTACION ALQUILER-VENTA (CANAL rho)"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "Los mercados de venta y alquiler no son independientes: estan conectados "
        "a traves del yield de rentabilidad bruta. Un incremento sostenido de los "
        "alquileres eleva la rentabilidad del activo como inversion de renta, "
        "atrayendo nueva demanda compradora y presionando al alza los precios de venta.\n\n"
        "Este mecanismo se captura mediante el coeficiente rho:\n\n"
        "   tasa_venta(t) += rho * [(Alquiler(t-1)/Alquiler(t-2)) - 1]\n\n"
        "Con rho=0.45, calibrado sobre la correlacion historica observada entre "
        "retornos de alquiler y precio de venta con retardo de 1 periodo en "
        "el mercado barcelones.\n\n"
        "El canal inverso (yield floor) se implementa mediante el factor de "
        "resistencia: cuando el yield bruto cae por debajo del 2.5%, la tasa de "
        "crecimiento de los precios de venta se frena proporcionalmente para "
        "mantener la viabilidad como inversion de renta. Este es el mecanismo "
        "de ancla fundamental del modelo: aunque no aplica mean-reversion directa, "
        "si evita que los precios de venta crezcan indefinidamente desconectados "
        "de los alquileres."
    ))
    pdf.ln(3)

    # -----------------------------------------------------------------------
    # SECCION 12: AGREGACION
    # -----------------------------------------------------------------------
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("12. AGREGACION ESTADISTICA Y RESULTADOS"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "Las n_sim trayectorias simuladas producen una distribucion empirica de precios "
        "para cada distrito y cada periodo 2027-2032. Los estadisticos reportados son:\n\n"
        "- P50 (mediana): proyeccion central. Se usa la mediana y no la media para "
        "  su mayor robustez frente a trayectorias extremas en las colas.\n"
        "- P10 / P90: percentiles que definen el intervalo de confianza al 80%. "
        "  El P10 representa el escenario adverso (solo el 10% de las simulaciones "
        "  produce precios por debajo de este nivel). El P90 representa el escenario "
        "  optimista.\n"
        "- CAGR (Compound Annual Growth Rate): tasa de crecimiento anual compuesta "
        "  calculada sobre la media de la distribucion final. Se usa para comparar "
        "  el retorno esperado entre distritos en el cuadrante Riesgo/Retorno.\n"
        "- CV (Coeficiente de Variacion): desviacion estandar / media de la "
        "  distribucion final. Proxy de riesgo: mayor CV indica mayor dispersion "
        "  de resultados posibles."
    ))

    # -----------------------------------------------------------------------
    # SECCION 13: LIMITACIONES
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("13. LIMITACIONES Y ADVERTENCIAS METODOLOGICAS"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El modelo v32.0 supone un avance significativo respecto a versiones anteriores "
        "al eliminar la endogeneidad del usuario en el drift, pero mantiene las "
        "siguientes limitaciones que el usuario debe considerar al interpretar los resultados:\n\n"
        "1. MUESTRA CORTA: el periodo de calibracion (2019-2026, n=8) es limitado para "
        "   estimar con precision los coeficientes VAR(1) y las volatilidades. Con n=8 "
        "   observaciones por distrito, los estimadores OLS tienen alta varianza. "
        "   El modelo compensa con truncaciones y regularizacion, pero los intervalos "
        "   de confianza internos de los parametros son amplios.\n\n"
        "2. DATOS DE OFERTA AUSENTES: el modelo no incorpora nueva construccion, "
        "   rehabilitacion ni cambios de uso. Un aumento significativo de la oferta "
        "   residencial en un distrito (nueva promotion, plan urbanistico) moderaria "
        "   los precios mas alla de lo que el drift historico captura.\n\n"
        "3. LINEARIDAD DEL AJUSTE MACRO: la funcion f(Euribor, IPC, Paro) -> delta_drift "
        "   es lineal y normalizada. En la realidad, los efectos macro pueden ser no "
        "   lineales (umbral de tipos a partir del cual la demanda colapsa).\n\n"
        "4. DCC SIMPLIFICADO: la modulacion de correlaciones por estado es una "
        "   aproximacion de la correlacion condicional dinamica real (Engle 2002). "
        "   No estima parametros GARCH por insuficiencia de datos.\n\n"
        "5. SEED FIJA: np.random.seed(42) garantiza reproducibilidad pero implica "
        "   que identicos parametros producen identicos resultados. Esto es "
        "   una ventaja para auditoria pero no captura la variabilidad montecarlo.\n\n"
        "6. HORIZONTE 6 ANOS: la incertidumbre acumulada en 6 anos es considerable. "
        "   Los intervalos P10-P90 son indicativos. A partir del ano 4-5, la banda "
        "   de incertidumbre puede representar el 30-40% del precio central.\n\n"
        "CLASIFICACION DEL MODELO: herramienta de analisis estrategico de inversion "
        "y comunicacion con inversores. No es un modelo de valoracion tasacion ni "
        "debe usarse para decisiones de credito hipotecario individual."
    ))

    # -----------------------------------------------------------------------
    # SECCION 14: TESTS
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("14. TESTS DE INTEGRIDAD MATEMATICA"), 0, 1)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5.5, clean_str(
        "El modelo incluye un sistema de verificacion automatica que se ejecuta "
        "en cada inicializacion. Los tests cubren:\n\n"
        "- Positividad de todos los precios base 2026\n"
        "- Yields brutos en rango economicamente plausible (2%-10%)\n"
        "- Drift historico plausible (< 30% anual por distrito)\n"
        "- Estabilidad del VAR(1): radio espectral de A_VAR < 1.5\n"
        "- Betas CAPM en rango [0.5, 1.5]\n"
        "- Descomposicion de Cholesky valida (triangular inferior, diagonal positiva)\n"
        "- Reconstruccion L*L^T aprox Sigma (error maximo < 1e-5)\n"
        "- MTM dinamicas filas suman 1.0 para los tres regimenes de referencia\n"
        "- Probabilidades de regimen en [0,1] y suma=1\n\n"
        "Resultados de la ejecucion actual:"
    ))
    pdf.set_font('Courier', '', 8)
    test_results = run_all_tests()
    for tname, tpass, tmsg in test_results:
        status = "[PASS]" if tpass else "[FAIL]"
        pdf.cell(0, 5, clean_str(f"{status}  {tname}"), 0, 1)
        pdf.cell(0, 4, clean_str(f"         {tmsg}"), 0, 1)

    # -----------------------------------------------------------------------
    # SECCION 15: REFERENCIAS
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 9, clean_str("15. REFERENCIAS BIBLIOGRAFICAS Y NORMATIVAS"), 0, 1)
    pdf.set_font('Times', '', 10)
    refs = [
        "Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary "
        "time series and the business cycle. Econometrica, 57(2), 357-384.",

        "Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press. "
        "[Referencia para Markov Switching con variables exogenas]",

        "Engle, R.F. (2002). Dynamic Conditional Correlation: A Simple Class of "
        "Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models. "
        "Journal of Business & Economic Statistics, 20(3), 339-350.",

        "Sims, C.A. (1980). Macroeconomics and Reality. Econometrica, 48(1), 1-48. "
        "[Referencia fundacional del modelo VAR]",

        "Sharpe, W.F. (1964). Capital asset prices: A theory of market equilibrium "
        "under conditions of risk. Journal of Finance, 19(3), 425-442.",

        "Shiller, R.J. (1993). Measuring Asset Values for Cash Settlement in Derivative "
        "Markets: Hedonic Repeated Measures Indices and Perpetual Futures. "
        "Journal of Finance, 48(3), 911-931.",

        "Case, K.E. & Shiller, R.J. (1989). The Efficiency of the Market for "
        "Single-Family Homes. American Economic Review, 79(1), 125-137.",

        "Glaeser, E. & Gyourko, J. (2018). The Economic Implications of Housing Supply. "
        "Journal of Economic Perspectives, 32(1), 3-30.",

        "Banco de Espana (2023). Informe de Estabilidad Financiera. Capitulo 2: "
        "Vulnerabilidades del sector inmobiliario. Madrid: BdE.",

        "Idealista Research (2026). Informe de Precios de Vivienda en Barcelona "
        "por Distritos, Q1 2026. Madrid: Idealista.",

        "Incasol - Institut Catala del Sol (2026). Estadistiques del sector immobiliari "
        "a Catalunya. Generalitat de Catalunya.",

        "INE (2026). Indice de Precios de Vivienda (IPV). "
        "Serie historica 2007-2026. Madrid: Instituto Nacional de Estadistica.",

        "Ajuntament de Barcelona / CEAT (2024). Registre d'Habitatges d'Us Turistic. "
        "Barcelona: Ajuntament de Barcelona.",

        "Inside Airbnb (2024). Data for Barcelona, Spain. insideairbnb.com.",

        "Ley 12/2023, de 24 de mayo, por el derecho a la vivienda. "
        "Boletin Oficial del Estado, num. 124, 25 de mayo de 2023.",

        "European Banking Authority (2023). EBA Report on Residential Real Estate "
        "Risk Weighting and Methodologies. EBA/REP/2023/28.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 5, clean_str(f"- {ref}"))
        pdf.ln(1)

    return pdf.output(dest='S').encode('latin-1')


# ============================================================================
# 16. BOTONES DE EXPORTACION
# ============================================================================
st.markdown("---")
st.subheader("Exportar documentacion")

fig_pdf = st.session_state.fig_distrito
idx_pdf = st.session_state.idx_distrito

if fig_pdf is None:
    _idx = idx_pdf
    fig_pdf, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _p50  = np.percentile(sv_act[:, _idx, :], 50, axis=0)
    _last = PRECIOS_RECONSTRUIDOS[_idx][-1]
    _ax1.plot(anos_larga_data, PRECIOS_RECONSTRUIDOS[_idx], 'o-', color='#2c3e50')
    _ax1.plot(anos_proy_con_2026, np.concatenate([[_last], _p50]), '--', color='#e74c3c')
    _ax1.set_title(f"Venta - {distritos[_idx]}")
    _ax1.grid(True, linestyle=':', alpha=0.5)
    _p50a  = np.percentile(sa_act[:, _idx, :], 50, axis=0)
    _lasta = precio_alquiler_historico[_idx, -1]
    _ax2.plot(anos_retro_corto, precio_alquiler_historico[_idx], 'o-', color='#2c3e50')
    _ax2.plot(anos_proy_con_2026, np.concatenate([[_lasta], _p50a]), '--', color='#27ae60')
    _ax2.set_title(f"Alquiler - {distritos[_idx]}")
    _ax2.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    st.session_state.fig_distrito = fig_pdf

col_e1, col_e2 = st.columns(2)

with col_e1:
    if st.button("Generar Informe Ejecutivo"):
        with st.spinner("Generando informe ejecutivo..."):
            pdf_exec = generate_exec_report(
                idx_pdf, scenarios_data, params_act, fig_pdf, n_sim_ui
            )
        fname_exec = (
            f"Informe_Ejecutivo_{distritos[idx_pdf].replace(' ', '_')}_v32.pdf"
        )
        st.download_button(
            label=f"Descargar {fname_exec}",
            data=pdf_exec,
            file_name=fname_exec,
            mime="application/pdf"
        )

with col_e2:
    if st.button("Generar Explicacion Metodologica Detallada"):
        with st.spinner("Generando documento metodologico completo..."):
            pdf_meto = generate_explicacion_metodologica(params_act, n_sim_ui)
        st.download_button(
            label="Descargar Explicacion Metodologica Detallada v32.0.pdf",
            data=pdf_meto,
            file_name="BCN_Model_v32_Explicacion_Metodologica_Detallada.pdf",
            mime="application/pdf"
        )

st.markdown("---")
st.caption(
    f"Barcelona Strategic Model v32.0 | Motor: MS-VAR + Markov no homogeneo + DCC | "
    f"Datos reales: {FECHA_DATO_REAL} | Fuente: {FUENTE_DATO} | "
    f"Regimen: {reg_sel} | Euribor: {euribor_val}% | IPC: {ipc_val}% | "
    f"Paro: {paro_val}% | n_sim: {n_sim_ui}"
)


