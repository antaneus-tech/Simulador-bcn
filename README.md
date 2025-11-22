TECHNICAL METHODOLOGICAL REPORT
1. INTRODUCTION
This model projects real estate prices using a hybrid stochastic approach that combines Markov Chains
for economic regime shifts with spatially correlated Monte Carlo simulations.
2. MASTER PRICE EQUATION
The evolution of price P over time t for district i is defined as:
P(t) = P(t-1) * [1 + mu * M(state) * gamma(i) + rho * R_rent(t-1) + Shock(t)]
Where:
- mu: Base macroeconomic growth rate.
- M(state): Multiplier derived from the Markov Chain (Expansion, Recession, Stagflation).
- gamma(i): District-specific sensitivity (price elasticity).
- rho: Coupling coefficient (0.45) linking rental yield to sale price.
- Shock(t): Random component spatially correlated through Cholesky decomposition.
3. SPATIAL MODELING (CHOLESKY)
Not all districts behave independently. The model uses a historical Correlation Matrix
(2019-2025) to ensure that a shock in l'Eixample coherently affects Gracia or Ciutat Vella.
Mathematically: Correlated_Epsilon = L * Z, where L is the lower triangular Cholesky matrix and Z is a vector
of standard white noise.
4. PARAMETERS AND ASSUMPTIONS
A) Markov Regimes: Three probability transition matrices are defined (Growth, Recession, Stagflation).
B) VT Regulation: A negative impact of 5% on rents and a slowdown in sales in high-tourism
concentration areas is assumed from 2028 onwards if the law is enacted.
C) Time Horizon: Projections over 6 years (2026-2031) with annual steps.
