BCN MODEL V30.0 | MASTER REPORT
STOCHASTIC MODELING OF URBAN REAL ESTATE DYNAMICS
A Hybrid Markov-Monte Carlo Approach with Regulatory Shock Integration
ABSTRACT
This paper presents the mathematical architecture of the Barcelona Strategic Model v30.0. The 
system forecasts residential asset prices (2026-2031) using a Discretized Stochastic Differential 
Equation (SDE) driven by macroeconomic regime switching (Markov Chains) and spatially correlated 
shocks (Cholesky). Special attention is given to the 'Regulatory Shock Function', which models the 
asymmetrical impact of the 2028 Tourist License ban based on real district-level density data.

1. MATHEMATICAL SPECIFICATION (SDE)
The price dynamics follow a modified Geometric Brownian Motion with Jump-Diffusion components. The 
governing equation for district i at time t is:
dP_i(t) / P_i(t) = [ mu(S_t) * gamma_i + rho * R_yield(t) ] dt + [ sigma_i * sum(L_ij * dZ_j) ] + 
J_vt(t, i)
Where:
- mu(S_t): Macro drift determined by the active Markov state S_t (Expansion, Stagflation, 
Recession).
- gamma_i: District-specific elasticity (Beta) derived from CAPM reconstruction (2007-2018).
- L_ij: Lower triangular matrix from Cholesky decomposition of the covariance matrix Sigma.
- J_vt: Jump process representing the regulatory shock in 2028, proportional to tourist license 
density.

2. PARAMETER DERIVATION & CALIBRATION
A) MARKOV DRIFT (mu): Calibrated to ensure hierarchy: Expansion (+4.5%) > Stagflation (+1.5%) > 
Recession (-2.5%).
B) ELASTICITY (gamma): Calculated via historical regression of district returns against the 
aggregated Barcelona index during the 2008-2013 crisis.
C)  CHOLESKY  MATRIX  (L):  Spectral  decomposition  is  applied  to  the  historical  correlation  
matrix  to  guarantee  positive definiteness before Cholesky factorization.

3. REGULATORY IMPACT FUNCTION
The shock J_vt is modeled as an additive term in 2028. It is not uniform but follows the density 
vector D_i derived from 'Inside Airbnb' and municipal census data:
J_vt(i) = alpha * (Density_i / Max_Density_Eixample)
This results in a maximum correction of -10% for Eixample (Density=0.46) and negligible impact for 
Nou Barris (Density=0.004).
4. DATA SOURCES
Transactional Data: Idealista Data / Incasol (Generalitat de Catalunya). Macro Data: INE (National 
Statistics Institute).
Tourist Licenses: Ajuntament de Barcelona (CEAT) / Inside Airbnb.
Page 1 | Confidential

