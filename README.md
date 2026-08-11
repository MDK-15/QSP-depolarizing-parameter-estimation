# QSP-depolarizing-parameter-estimation
The goal of this work is to estimate decoherent noise parameters as efficiently as possible

We utilize the QSP framework to build circuits optimized to maximize sensitivity to the desired parameters. Concretely, we aim to maximize:
```math
\kappa(\theta, \Phi, d;\, \mathcal{I}_{p}) = \frac{L_{\mathcal{E}}(\theta, \Phi, d;\, \mathcal{I}_{p})}{d}.
```
where:
```math
L_{\mathcal{E}}(\theta, \Phi, d;\, \mathcal{I}_{p}) = \min_{p \in \mathcal{I}_{p}} \sqrt{ F_{p}(\theta, \Phi, d)}
```
```math
F_{p}(\theta, \Phi, d) = \frac{ \left( \partial_{p} f_{\mathcal{E}} \right)^{2} }{ f_{\mathcal{E}} \left( 1 - f_{\mathcal{E}} \right)}
```
```math
f_{\mathcal{E}}  (p) =  \operatorname{Tr}\left[ \left( \ket{0}\bra{0}_{A} \otimes I_{S} \right) \rho_{AS}^{\mathrm{out}}(\theta, p; \Phi, d) \right]
```
