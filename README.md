# TMA4195 Climate Modelling Project

Code from a group project on simple climate models in **TMA4195 Mathematical Modelling** at NTNU (autumn 2024).
The project builds energy balance models of the Earth, from a two-layer surface/atmosphere model to a
latitude-resolved model with an ice cap. It also studies their equilibria and stability numerically.

The repository holds the code and figures, which were shared here for grading. The written report itself is not included.
Contributors: Axel Friberg Hagen, [EliasKRekkedal](https://github.com/EliasKRekkedal) and [MaxPfisterer](https://github.com/MaxPfisterer).

## Models and methods

**Two-layer radiative balance (`matmodOppg3.py`).**
The model tracks shortwave and longwave energy flows between the sun, atmosphere, clouds and the Earth's surface.
- The constant shortwave fluxes come from a $4 \times 4$ linear system.
- The steady-state balance for the surface and atmosphere temperatures $(T_E, T_A)$ is a pair of nonlinear equations with $T^4$ radiation terms. They are solved with Newton's method using an analytic inverse Jacobian, with and without cloud cover.
- A central-difference estimate of $\partial T / \partial \alpha$ gives the sensitivity to one of the heat-exchange parameters.

**Latitudinal energy balance model (`Question10.py`).**
With $x = \sin(\text{latitude})$, the steady-state temperature $T(x)$ satisfies

$$
D \frac{d}{dx}\left[(1 - x^2)\frac{dT}{dx}\right] - (A_{out} + B_{out} T) + Q\, S(x)\, a(x, x_s) = 0,
$$

where:
- $S(x) = 1 + S_2 P_2(x)$ is the normalised insolation.
- $a(x, x_s)$ is the absorbed fraction of sunlight, which drops at the ice line $x_s$.
- $D$ is the heat-transport coefficient.

The operator is discretised with central finite differences on $[0, 1]$, giving a tridiagonal matrix. Its eigenvalues are inspected to assess stability.

**Linear stability (`exercise12.ipynb`).**
Small perturbations satisfy $C_a\, \partial_t \delta T = D\, \partial_x[(1-x^2)\partial_x \delta T] - B_{out}\, \delta T$.
The notebook discretises this, computes the eigenvalue spectrum for $N = 10, 20, 50, 100, 500$ grid points and plots the leading eigenmodes, the sensitivity to $D$ and $B_{out}$, and how a perturbation decays over time.

## Results

- In the saved notebook output, every eigenvalue of the discretised perturbation operator has a negative real part for all tested grid sizes. The equilibrium is therefore linearly stable. For $N = 50$, no eigenvalue has a nonzero imaginary part, so there are no oscillatory modes.
- Two result figures are included in the repository:

| Insolation profile $S(x)$ | Temperature vs. latitude |
|---|---|
| ![S(x) approximations](Question7.png) | ![Temperature as a function of latitude](Question10.png) |

`Question7.png` compares the Legendre approximation of $S(x)$ (unscaled and scaled) with a geometric estimate.
`Question10.png` shows the steady-state temperature profile with the given $D$ and with a tuned $D$. With the tuned $D$, the temperature at the ice-cap boundary is 0 °C.

## Repository structure

```
matmodOppg3.py     Two-layer surface/atmosphere model, Newton solver, sensitivity
Question10.py      Finite-difference latitudinal EBM and eigenvalue stability check
exercise12.ipynb   Linear stability analysis, eigenmodes and perturbation decay
Question7.png      Figure: insolation distribution S(x)
Question10.png     Figure: temperature vs. latitude, given vs. tuned D
```

## Running

Requires Python 3 with `numpy`, `scipy` and `matplotlib` (plus `jupyter` for the notebook).

```bash
pip install numpy scipy matplotlib jupyter
python matmodOppg3.py      # prints equilibrium temperatures and a sensitivity estimate
python Question10.py       # plots eigenvalue spectra
jupyter notebook exercise12.ipynb
```

The scripts are exploratory course code. Parameter values are hard-coded, and some parts, such as the steady-state solve in `Question10.py`, are commented out.
The code that produced the two PNG figures is not fully included.
