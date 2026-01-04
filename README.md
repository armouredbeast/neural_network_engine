# Quant Neural Research Engine  
**Multi-Model Market Dynamics Study (Returns · Volatility · Regimes · Signals)**

---

## Overview

This project is a **from-scratch quantitative research engine** designed to study how market behavior emerges from **returns, volatility, regime changes, and risk-adjusted signals** — using both **first-principles neural networks** and **modern machine-learning models**.

The goal is **not prediction for trading**, but **structural understanding**:
- How price movements evolve
- How volatility clusters
- How regimes shift
- How signals should adapt to risk

This mirrors how **institutional quant desks** think about markets:  
as **systems**, not indicators.

---

## Motivation

Most quant implementations jump straight to:
- Black-box models
- Over-engineered pipelines
- Libraries without understanding

This project deliberately **does the opposite**.

### Core motivations:
1. Understand the mechanics of learning, not just use ML
2. Separate market components instead of fitting one monolithic model
3. Build intuition for:
   - Feature construction
   - Label alignment
   - Regime dependence
   - Risk-aware signal design

This reflects real-world quant research:

> *Returns are noisy. Volatility is structured. Regimes govern behavior.*

---

## Architecture (High Level)

The system is composed of **four conceptual models**, each answering a different question:

| Model | Question Answered |
|------|------------------|
| Model 1 | Where might returns move next? |
| Model 2 | How risky is that movement? |
| Model 3 | What market regime are we in? |
| Model 4 | How should a signal adapt to all of the above? |

Each model is implemented in **two styles**:
- **A-Series** → built from scratch (NumPy, math-first)
- **B-Series** → built using ML abstractions (production-style)

---

## Model Breakdown

### Model 1 — Return Dynamics

**Objective:** Learn directional return behavior from recent price history.

- Input: rolling return windows  
- Output: next-step return estimate  

Returns are the noisiest part of the market. Modeling them alone highlights the limits of prediction and the need for risk control.

- **1A:** Neural network from scratch (NumPy)
- **1B:** Random Forest regressor (non-linear baseline)

---

### Model 2 — Volatility Structure

**Objective:** Estimate realized volatility aligned with return features.

- Input: rolling windows  
- Output: forward realized volatility  

Volatility is more predictable than returns and drives position sizing, risk management, and execution decisions.

Key design choice:
- Volatility labels are **strictly aligned** with return features  
  (a common real-world quant pitfall)

---

### Model 3 — Regime Classification

**Objective:** Classify market state (e.g., low-vol vs high-vol).

- Input: historical feature vectors  
- Output: regime label  

Markets behave differently under different regimes.  
Strategies that ignore regimes fail in practice.

This layer governs **how signals should behave**, not just what they predict.

---

### Model 4 — Risk-Adjusted Signal Engine

**Objective:** Combine all previous outputs into an adaptive signal.

Signal logic:
- Normalize expected returns by volatility
- Modulate exposure based on regime

Resulting signal is:
- Direction-aware
- Risk-aware
- Regime-aware

This mirrors how **real trading systems** separate forecasting from decision-making.

---

## Data Flow (Conceptual)
Returns
↓
Rolling Features
↓
Model 1 → Expected Return
↓
Model 2 → Expected Volatility
↓
Model 3 → Market Regime
↓
Model 4 → Risk-Adjusted Signal

Each layer adds **structure**, not unnecessary complexity.

---

## Why This Matters (Interview Perspective)

This project demonstrates:

- Proper feature–label alignment
- Understanding of market microstructure concepts
- Separation of prediction, risk, and decision logic
- Ability to build models from first principles
- Awareness of regime-dependent behavior

It reflects **how quant desks actually reason**, not how tutorials are written.

---

## Design Philosophy

- Clarity over cleverness  
- Structure over shortcuts  
- Learning systems, not chasing alpha  

The engine is modular and extensible:
- Macro variables can be added
- EU / global market extensions are straightforward
- Outputs can be visualized using 3D or animation tools
- Serves as a foundation for deeper research

---

## What This Is Not

- Not a trading strategy  
- Not investment advice  
- Not an HFT system  
- Not optimized for PnL  

This is a **research framework** — the layer *before* trading.

---

## Future Extensions

Planned work includes:
- Macro-economic inputs (rates, FX, flows)
- Cross-market spillover modeling
- Sequence-based neural models
- Visualization-driven quantitative storytelling

---

## Closing Thought

> Markets are not random.  
> They are structured uncertainty.

This project is about learning that structure — one layer at a time.
