# SCE Equations Reference

## Notation
- Dataset: $\mathcal{D} = \{(x_t, y_t)\}_{t=1}^n$
- Hierarchy levels: $k = 1,\dots,K$
- Neighborhood for level $k$: $\mathcal{N}_k(t)$
- Summarizer: $\mathcal{S}_k(\cdot)$ returns $\mathbb{R}^{d_k}$

## Context Features

**Eq. (1)** Context vector per level

$\phi^{(k)}(x_t) = \mathcal{S}_k(\{y_s : s \in \mathcal{N}_k(t)\})$

**Eq. (2)** Concatenated embedding

$\Phi(x_t) = [\phi^{(1)}(x_t), \dots, \phi^{(K)}(x_t)]$

## Relative Features

**Eq. (3)** Relative features

$ r_{k,z}(x_t) = \frac{y_t - \mu_k(x_t)}{\sigma_k(x_t) + \varepsilon}$

$ r_{k,ratio}(x_t) = \frac{y_t}{\mathrm{median}_k(x_t) + \varepsilon}$

## Leakage-Safe Cross-Fitting

**Eq. (4)** Out-of-fold context for training point $t \in \mathcal{I}_m$

$\phi^{(k)}_{cf}(x_t) = \mathcal{S}_k(\{y_s : s \in \mathcal{N}_k(t) \cap (\{1,\dots,n\}\setminus \mathcal{I}_m)\})$

## Algorithm 1 (SCE Construction)

1. For each level $k$, compute cross-fitted group summaries
2. Join summaries to dataset
3. Add relative features
4. Return augmented dataset $[x_t, \Phi(x_t), r_{k,z}, r_{k,ratio}]$
