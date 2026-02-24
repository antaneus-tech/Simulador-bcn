# 🏙️ Barcelona Strategic Model v41.0

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)
![License](https://img.shields.io/badge/License-Confidential-red.svg)

El **Barcelona Strategic Model v40.0** es un simulador econométrico avanzado basado en **Ecuaciones Diferenciales Estocásticas (SDE)** diseñado para proyectar la evolución del mercado inmobiliario residencial (venta y alquiler) en los 10 distritos de Barcelona hasta el año 2032.

Esta herramienta supera las proyecciones lineales tradicionales mediante la integración de Cadenas de Markov no homogéneas, correlación espacial dinámica (DCC), calibración de Betas empíricas independientes y simulación de Monte Carlo.

---

## 🚀 Novedades de la Versión 41.0 (Arquitectura Híbrida)

La V40.0 representa un rediseño matemático completo del motor estocástico:

1. **SDE Bifurcada:** Separación estricta entre la **Inercia Endógena** (el *drift* histórico y la inercia autorregresiva VAR/AR) y el **Ciclo Macro Modulado**.
2. **Betas Empíricas Independientes:** Eliminación de los multiplicadores arbitrarios. Se calculan elasticidades ($\beta$) reales cruzando los datos distritales contra el **IPV del INE** (para ventas) y un **Índice Sintético de Barcelona** (para alquiler).
3. **Calibración OLS para Shocks de Régimen:** La función de impacto macroeconómico $\Phi(S_t, M_t)$ asume una asimetría matemática en escenarios de crisis (solo amplifica *shocks* negativos) y sus multiplicadores se extraen empíricamente mediante regresión OLS sobre ventanas históricas reales (2013-2020).
4. **Arquitectura Híbrida de Datos:** Los parámetros estructurales profundos ($\beta$, Cholesky, Volatilidad Base) se derivan **siempre** de la extensa serie histórica de transacciones del Ayuntamiento (Set B), mientras que la inercia a corto plazo respeta el set de datos que el usuario desea analizar (ej. Portales Inmobiliarios).

---

## 🧮 La Ecuación Gobernadora (SDE)

El motor de simulación principal gobierna las tasas de crecimiento mediante la siguiente estructura de bloques estancos:

```text
E[dP_i(t)/P_i(t)] = 
    [ μ_hist_i + λ * A_var_i(t) ]        <-- Bloque 1: Inercia Endógena (Set Seleccionado)
  + [ β_vta_i * Φ(S_t, M_t) ]            <-- Bloque 2: Ciclo Modulado (OLS Empírico)
  + [ σ_base_i * θ(S_t) * Σ L_ij * Z_j ] <-- Bloque 3: Estocástico (DCC + Cholesky)
  + [ ρ * ret_alq(t-1) ]                 <-- Bloque 4: Feedback del Yield
  + J_VT(t, i, año_shock)                <-- Bloque 5: Shock Regulatorio (Jump Process)
⚙️ Instalación y Ejecución Local1. Clonar el repositorioBashgit clone [https://github.com/tu-usuario/barcelona-strategic-model.git](https://github.com/tu-usuario/barcelona-strategic-model.git)
cd barcelona-strategic-model
2. Instalar dependenciasSe recomienda utilizar un entorno virtual (venv o conda).Bashpip install -r requirements.txt
Asegúrate de que el archivo requirements.txt incluya fpdf2 (para la correcta generación de PDFs con codificación UTF-8).3. Ejecutar la aplicaciónEl modelo utiliza Streamlit para la interfaz web.Bashstreamlit run barcelona_model_v41.py
(Sustituye el nombre del archivo si lo has renombrado a app.py o main.py).📊 Estructura de la Interfaz (UI)El panel lateral permite configurar el Régimen Económico, los Indicadores Macro (Euríbor, IPC, Paro), el Año del Shock Regulatorio VT, y la fuente de datos base.El panel principal se divide en 5 pestañas interactivas renderizadas con Plotly:📍 Análisis Distrito: Proyección temporal interactiva (Fan Chart P10-P50-P90) del precio de venta y alquiler por distrito.⚖️ Riesgo / Retorno: Cuadrante estratégico cruzando la Volatilidad (CV) frente a la Tasa de Crecimiento Anual Compuesta (CAGR).🔬 Macro & Φ: Visualización de las probabilidades de la Matriz de Transición de Markov y la curva de la función asimétrica de Crisis $\Phi$.🧮 Parámetros v40: Tablas con las $\beta$ empíricas calculadas, matriz heatmap de correlación de Cholesky y componentes de inercia.✅ Tests de Integridad: Batería de tests unitarios automáticos que validan el rango de las volatilidades, los autovalores de la matriz Ridge y la positividad geométrica.📄 Generación de Informes PDFEl modelo permite exportar la inteligencia generada de dos formas nativas desde la barra inferior de la aplicación:Informe Ejecutivo V40 (por distrito): Documento gerencial con los KPIs a 2032, una renderización estática (matplotlib en modo Agg) del gráfico de proyecciones, y una tabla de comparativa de rendimientos Portales vs. transacciones de Ayuntamiento.Explicación Metodológica Detallada: Un Whitepaper algorítmico de múltiples páginas generado dinámicamente con los parámetros y seeds exactos de la sesión actual, diseñado para auditoría cuantitativa o presentación a comités de inversión.⚠️ Aviso Legal (Disclaimer)El Barcelona Strategic Model es una herramienta matemática diseñada exclusivamente para el análisis estratégico, la investigación macroeconómica y el estudio académico de escenarios de estrés. No constituye asesoramiento financiero, recomendación de inversión ni tasación inmobiliaria vinculante. Los autores no se hacen responsables de las decisiones económicas tomadas en base a las simulaciones (Monte Carlo) arrojadas por este software.

