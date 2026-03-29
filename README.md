# Informe Metodológico Exhaustivo: Barcelona Strategic Model v40.1

**Fecha de Análisis:** 29 de Marzo de 2026
**Objetivo:** Desglose técnico y matemático completo de las Ecuaciones Diferenciales Estocásticas (SDE), procesos de decisión de Markov y formulaciones empíricas del modelo predictivo inmobiliario `app.py`.

---

## 1. Fundamentos del Entorno Macroeconómico ($M_t$)

El motor principal que dirige el ciclo de mercado se llama **Impulso Macro ($M_t$)**, calculado de forma diferente para el mercado de Venta y Alquiler debido a la naturaleza asimétrica de la demanda cruzada.

### A. Impulso Macro para Venta ($M_{t, vta}$)
Penaliza las subidas de los tipos de interés y el paro, pero asume una inflación contenida como proxy de salud transaccional:
$$M_{t, vta} = - \text{clip}\left(\frac{\text{Euribor} - 2.5}{100}, -0.02, 0.04\right) + \text{clip}\left(\frac{\text{IPC} - 2.0}{100}, -0.03, 0.03\right) - \text{clip}\left(\frac{\text{Paro} - 10.0}{100} \cdot 0.5, -0.02, 0.03\right)$$
*(Un $M_t > 0$ incentiva el ciclo, $M_t < 0$ lo deprime).*

### B. Impulso Macro para Alquiler ($M_{t, alq}$)
El Euribor tiene aquí un signo inverso. Tipos altos restringen el acceso crediticio a la venta, desviando estructuralmente esa demanda hacia el alquiler, lo que incrementa los precios por shock de demanda (crowding out):
$$M_{t, alq} = + \text{clip}\left(\frac{\text{Euribor} - 2.5}{100} \cdot 0.4, -0.01, 0.02\right) + \text{clip}\left(\frac{\text{IPC} - 2.0}{100} \cdot 0.6, -0.02, 0.02\right) - \text{clip}\left(\frac{\text{Paro} - 10.0}{100} \cdot 0.3, -0.01, 0.02\right)$$

---

## 2. Régimen Predictivo de Markov No Homogéneo

El algoritmo evalúa iterativamente 3 estados estocásticos discretos $S_t \in \{\text{0: Normal, 1: Boom, 2: Crisis}\}$. Primero calcula un "score" probabilístico ($U_{S}$) normalizando las tres variables en su recorrido histórico en España (Euribor $[0, 5]$, IPC $[0, 8]$, Paro $[5, 30]$):

$$E_{norm} = \text{clip}\left(\frac{\text{Euribor}}{5.0}, 0, 1\right)$$
$$I_{norm} = \text{clip}\left(\frac{\text{IPC}}{8.0}, 0, 1\right)$$
$$P_{norm} = \text{clip}\left(\frac{\text{Paro} - 5.0}{25.0}, 0, 1\right)$$

Se generan los puntajes (scores) brutos para cada régimen:
$$U_{norm} = 0.5$$
$$U_{boom} = (1 - E_{norm}) \cdot 0.4 + (1 - P_{norm}) \cdot 0.4 + \text{clip}\left( I_{norm}(1 - I_{norm}) \cdot 4, 0, 1\right) \cdot 0.2$$
$$U_{crisis} = E_{norm} \cdot 0.45 + P_{norm} \cdot 0.45 + \text{clip}(I_{norm} - 0.5, 0, 0.5) \cdot 0.1$$

Posteriormente, las probabilidades de estado $P(S)$ se consiguen a través de una función _Softmax_ aplacada:
$$P(S_i) = \frac{\exp(U_{S_i} \cdot 2.5)}{\sum_{j \in \{norm, boom, crisis\}} \exp(U_{S_j} \cdot 2.5)}$$

Estas probabilidades $P_{norm}, P_{boom}, P_{crisis}$ actúan sobre una **Matriz de Transición de Markov ($MTM$)** con persistencia base $PER = 0.55$:
$$
\mathbf{MTM} = 
\begin{bmatrix}
PER + (1-PER)P_{norm} & (1-PER)P_{boom} & (1-PER)P_{crisis} \\
0.35(1-P_{boom}) & PER + (1-PER)P_{boom} & 0.05 + 0.10 P_{crisis} \\
0.12 + 0.20 P_{norm} & 0.02 & PER + (1-PER)P_{crisis}
\end{bmatrix}
$$
*(Cada fila es re-normalizada obligatoriamente para que sume 1).*

---

## 3. Calibración Empírica del Shock $\Phi(S_t, M_t)$

El algoritmo realiza un análisis OLS empírico sobre las muestras de **Set B (Transacciones Reales y Oficiales del Ayuntamiento)** durante periodos pasados de Boom y Crisis, derivando el multiplicador de inercia empírica $\kappa$ y el diferencial constante $C$.
Durante Monte Carlo, el impulso de crecimiento macro que se traslada al precio se modela bajo un efecto multiplicativo y asimétrico:

$$
\Phi_v(S_t, M_{t, vta}) = 
\begin{cases} 
M_{t, vta} & \text{si } S_t = \text{Normal} \\
M_{t, vta} \cdot \kappa_{boom} + C_{boom} & \text{si } S_t = \text{Boom} \\
\min(M_{t, vta}, 0) \cdot \kappa_{crisis} + \max(M_{t, vta}, 0) \cdot 1.0 - C_{crisis} & \text{si } S_t = \text{Crisis} 
\end{cases}
$$

**Asimetría Convexa de "Crisis":** Como se observa, ante un estado de Crisis, todo "rebote momentáneo" macropositivo ($\max(M_t, 0) \cdot 1.0$) no goza del impulso agresivo $\kappa_{crisis}$, mientras que cada choque macro negativo ($\min(M_t, 0)$) sí se amplifica agudamente multiplicándose por $\kappa_{crisis}$ y restándole $C_{crisis}$, provocando drenajes bruscos durante el crash.

*Para alquiler*, el factor de amplificación de shock se atenúa algorítmicamente aplicándose multiplicadores más elásticos:
$$\Phi_{a} \to M_{t, alq} \cdot (\kappa \cdot 0.7) \pm (C \cdot 0.6)$$

---

## 4. Motor VAR(1) e Inercia Endógena

Las memorias cortas del comportamiento del mercado (inercia autorregresiva de precios locales de un año respecto al anterior $r_{t-1}$) resueltas en los 10 distritos se estiman globalmente a través de un **Vector Autorregresivo regularizado (Ridge):**

Dada matriz $X$ con retornos en $t$ y vector predictivo $Y$ con retornos en $t+1$:
$$A_{var} = \left( X^T X + \lambda_{var\_ridge} \cdot \mathbf{I} \right)^{-1} X^T Y$$
Donde $\lambda_{var\_ridge} = 0.5$.

Se limita en seguridad el radio espectral de los autovalores en $\mid \lambda_{max}(A_{var}) \mid \leq 0.85$ para evitar trayectorias estocásticas explosivas. 

---

## 5. Simulación Estocástica: Dynamic Conditional Correlation (SDE Monte Carlo)

### A. DCC - Adaptación a Régimen
Añadida la Matriz de Correlación Base ($\mathbf{Corr}_{base}$) de los 10 distritos originada desde 2007, y las volatilidades base ($\sigma_{base}$), el modelo DCC inyecta asimetrías de diversificación:
* **Boom:** La correlación disminuye, el ruido sube independientemente en zonas de moda, bajando la volatilidad general.
  $\sigma_{dyn} = \sigma_{base} \cdot 0.85$ 
  $\mathbf{Corr}_{dyn} = \mathbf{Corr}_{base} \cdot 0.8 + \mathbf{I} \cdot 0.2$
* **Crisis:** La correlación se acentúa severamente hacia 1.0 (caen todos a la vez), aumentando gravemente la dispersión.
  $\sigma_{dyn} = \sigma_{base} \cdot 1.60$
  $\mathbf{Corr}_{dyn} = \mathbf{Corr}_{base} \cdot 0.5 + J \cdot 0.5$ (siendo $J$ Matriz de unos)
* Y en Normal: $\sigma_{dyn} = \sigma_{base}, \mathbf{Corr}_{dyn} = \mathbf{Corr}_{base}$

Se deriva la descomposicion **Cholesky** dependiente del régimen para distribuir la innovación de ruido gaussiano múltiple Z:
$$L_{dyn} = \text{Cholesky}(\mathbf{Corr}_{dyn})$$

### B. Ecuación Estocástica de Venta Anual ($t$)
Para la Venta iterativa en cada $t$, para cada distrito $dist$:

$$
r_{vta, t}^{dist} = 
\underbrace{\mu_{vta}^{dist} + \lambda_{var} \left[ A_{var} \cdot \vec{r}_{prev} \right]^{dist}}_{\text{Inercia Endógena (Drift + MS-VAR)}} 
+ 
\underbrace{\beta_{vta}^{dist} \cdot \Phi_v(S_t, M_{t, vta})}_{\text{Ciclo Macro + Shock Régimen}} 
+ 
\underbrace{\sigma_{dyn}^{dist} \cdot \left[ L_{dyn} \cdot \vec{Z} \right]^{dist}}_{\text{Ruido Blanco Cíclico Incorrelacionado Espacial}}
+
\underbrace{\rho_{vta, alq} \cdot \text{clip}(r_{alq, t-1}^{dist}, -0.05, 0.05)}_{\text{Feedback Retorno Alquiler}}
+
J_{vta}
$$
  * *$\lambda_{var} = 0.15$ es la atenuación temporal del vector autorregresivo.*
  * $\vec{Z} \sim \mathcal{N}(0, 1)$
  * $\rho_{vta, alq} = 0.45$

### C. Estabilizaciones Fundamentales (Yield Floor & Vol Floor)
1. **Piso de Caída libre:**
   $$r_{vta, t}^{dist} = \max(r_{vta, t}^{dist}, -0.07)$$
2. **Floor Yield Ratio (Gravity):**
   Si la simulación empuja la Venta lejos de los fundamentos básicos provocando rentabilidades ínfimas de Yield, la función atenúa la tasa futura.
   $$Yield_{bruto, t} = \frac{Precio_{alq, t} \cdot 12}{Precio_{vta, t}}$$
   $$Resistencia_t = \text{clip}\left( \frac{Yield_{bruto, t}}{0.025}, 0.5, 1.0 \right)$$
   $$\text{Si } r_{vta, t}^{dist} > 0 \rightarrow r_{vta, t}^{dist} = r_{vta, t}^{dist} \cdot Resistencia_t$$
   *(Es decir, a medida que la vivienda renta menos del 2.5%, las subidas interanuales especulativas se bloquean gradualmente frenando hasta el 50% de revalorización).*

### D. Ecuación Estocástica de Alquiler ($t$)
$$
r_{alq, t}^{dist} = 
\max\left[ \mu_{alq}^{dist} + (\lambda_{var} \cdot 0.5) \cdot \left(\text{diag}(A_{var}) \cdot r_{acumulado, alq}^{dist} \right) + \beta_{alq}^{dist} \cdot \Phi_a + (\sigma_{dyn}^{dist} \cdot 0.6) \left[ L_{dyn} \cdot \vec{Z} \right]^{dist} + J_{alq}, -0.05 \right]
$$
*(Nota: El alquiler no es vectorial en el término cruzado de VAR, es diagonal al distrito local puro local, es significativamente menos sensible al régimen de volatilidad espacial $\cdot 0.6$, y no padece bloqueos de yield, reflejando su inercia viscosa de contratos plurianuales).*

### E. Shock Regulatorio Negativo Turístico (Jump Process: $J$)
En el año de entrada del decreto anti pisos turísticos (2028/2029), el modelo aplica una pérdida aditiva inmediata de mercado según la carga de saturación registrada (`concentración_vt_real` local). 
Si la concentración máxima en BCN es de $D_{max} = 0.46$ (como el de mayor carga proporcional de la ciudad), el salto brusco es perjudicial:
$$J_{vta} = -0.10 \cdot \left( \frac{\text{Concentración}_{vt}^{dist}}{D_{max}} \right)$$
$$J_{alq} = -0.06 \cdot \left( \frac{\text{Concentración}_{vt}^{dist}}{D_{max}} \right)$$

---

## 6. Procedimiento Retrospectivo Final (Recálculo Iterativo)
1. **Integración:** El precio final simula el *Growth Rate* determinándolo escalarmente:
   $$Precio_{t} = \max(Precio_{t-1} \cdot (1 + r_t), \text{Límite Inferior})$$
   (Tratado con límite min: Venta 500 €/m², Alquiler 5 €/m²).
2. **Distribución Confidencial P10, P50 (Mediana), P90**: Tras generar $n\_sim$ trayectorias matriciales, el vector 3D final `[simulación, distrito, tiempo]` se reduce en cuantiles y media móvil para reportarse en visualización final.


