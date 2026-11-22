```@meta
# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
# SPDX-License-Identifier: MIT OR CC-BY-4.0
```

# [Background](@id QGG-Background)

This page derives the mathematical theory behind gradient generators. The central result is that propagating a gradient vector under a gradient generator yields the derivative of the propagated state with respect to the controls, exactly, and without any assumption on the time step. We prove this first for a single time step with a constant dynamical generator, then reinterpret the construction as forward-mode automatic differentiation with dual numbers, and finally extend it to time-dependent generators and to arbitrary control parameters, which is the foundation for methods like GOAT [MachnesPRL2018](@cite).

**Contents**

```@contents
Pages=["background.md"]
Depth=2:3
```

## [Setting and notation](@id Background-Setting)

We consider an equation of motion of the form

```math
\begin{equation}\label{eq:tdse}
\ii \frac{\partial}{\partial t} |Ψ(t)⟩ = \hat{H}(t) |Ψ(t)⟩
\end{equation}
```

with ``\hbar = 1``, where the dynamical generator ``\hat{H}(t) = \hat{H}(\{ϵ_l(t)\})`` is a Hamiltonian or a Liouvillian (in which case ``|Ψ⟩`` is a vectorized density matrix) of dimension ``N \times N`` that depends on the real-valued [control functions](@extref QuantumControl :label:`Control-Function`) (or "controls") ``ϵ_1(t), \dots, ϵ_L(t)``. We do not assume that ``\hat{H}`` is Hermitian, and none of the results on this page depend on it. The [control derivatives](@extref QuantumControl :label:`Control-Derivative`) are

```math
\begin{equation}\label{eq:mu}
\hat{μ}_l = \frac{\partial \hat{H}}{\partial ϵ_l}\,.
\end{equation}
```

For linear controls, ``\hat{H} = \hat{H}_0 + \sum_l ϵ_l(t) \hat{H}_l``, the control derivatives are the static [control operators](@extref QuantumControl :label:`Control-Operator`), ``\hat{μ}_l = \hat{H}_l``. For non-linear controls, they depend on the values of the controls.

In a piecewise-constant propagation on a time grid ``t_0 = 0, t_1, \dots, t_{N_T} = T``, the time evolution of a single time step ``dt_n = t_n - t_{n-1}`` is ``\hat{U}_n = \exp[-\ii \hat{H}_n dt_n]``, with the generator ``\hat{H}_n`` evaluated for the values ``ϵ_{nl}`` of the controls on the ``n``'th interval. In GRAPE, these values (the [pulses](@extref QuantumControl :label:`Pulse`)) are the control parameters that are optimized. Gradient-based optimal control methods like GRAPE [KhanejaJMR2005](@cite) require the derivative ``\partial \hat{U}_n / \partial ϵ_{nl}`` applied to a state, see the [Background of the GRAPE documentation](@extref GRAPE :label:`GRAPE-Background`). The accuracy of this derivative is critical for convergence, in particular in combination with quasi-Newton methods [FouquieresJMR2011](@cite).

The gradient generator ``G = G[\hat{H}]`` is the block operator

```math
\begin{equation}\label{eq:gradgen}
G[\hat{H}] = \begin{pmatrix}
    \hat{H}  &  0       &  \dots   &  0        &  \hat{μ}_1 \\
    0        &  \hat{H} &  \dots   &  0        &  \hat{μ}_2 \\
    \vdots   &          &  \ddots  &           &  \vdots      \\
    0        &  0       &  \dots   &  \hat{H}  &  \hat{μ}_L \\
    0        &  0       &  \dots   &  0        &  \hat{H}
\end{pmatrix}\,,
\end{equation}
```

which acts on gradient vectors ``Ψ`` of dimension ``N(L+1)`` as

```math
\begin{equation}\label{eq:gradgen-action}
G Ψ
= G \begin{pmatrix} |Ψ^\prime_1⟩ \\ \vdots \\ |Ψ^\prime_L⟩ \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix} \hat{H} |Ψ^\prime_1⟩ + \hat{μ}_1 |Ψ⟩ \\ \vdots \\ \hat{H} |Ψ^\prime_L⟩ + \hat{μ}_L |Ψ⟩ \\ \hat{H} |Ψ⟩ \end{pmatrix}\,.
\end{equation}
```

The gradient vector ``Ψ`` (without a ket) contains the gradient states ``|Ψ^\prime_l⟩`` in addition to the state ``|Ψ⟩``. The claim we will prove is that

```math
\begin{equation}\label{eq:main}
e^{-\ii G dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix}
    \frac{\partial}{\partial ϵ_1} e^{-\ii \hat{H} dt} |Ψ⟩ \\
    \vdots \\
    \frac{\partial}{\partial ϵ_L} e^{-\ii \hat{H} dt} |Ψ⟩ \\
    e^{-\ii \hat{H} dt} |Ψ⟩
\end{pmatrix}\,.
\end{equation}
```

Any method that can propagate a state under ``\hat{H}`` can propagate a gradient vector under ``G``. In the code, a [`GradgenOperator`](@ref) implements Eq. \eqref{eq:gradgen-action} block by block, without ever forming the matrix in Eq. \eqref{eq:gradgen}, and a [`GradVector`](@ref) holds ``|Ψ⟩`` in `Ψ.state` and ``|Ψ^\prime_l⟩`` in `Ψ.grad_states[l]`.


## [Reduction to a single control](@id Background-Reduction)

By Eq. \eqref{eq:gradgen-action}, the ``l``'th gradient state is driven only by itself and by the state ``|Ψ⟩``, and the state evolves under ``\hat{H}`` alone. Hence, in any propagation under ``G``, the pair ``(|Ψ^\prime_l⟩, |Ψ⟩)`` evolves exactly as it would under the single-control gradient generator

```math
\begin{equation}\label{eq:gradgen-single}
G_l = \begin{pmatrix} \hat{H} & \hat{μ}_l \\ 0 & \hat{H} \end{pmatrix}
\end{equation}
```

of dimension ``2N \times 2N``. This holds for arbitrary initial gradient states and for any kind of propagation: a single time step, a product of time steps, a polynomial propagator, or the solution of the equation of motion with a time-dependent ``G(t)``. The other gradient states have no influence on ``|Ψ^\prime_l⟩``. All ``L`` single-control problems share the same state, and propagating one gradient vector of dimension ``N(L+1)`` solves them all at once.

From here on, we consider a single control ``ϵ``, with ``G = \left(\begin{smallmatrix} \hat{H} & \hat{μ} \\ 0 & \hat{H} \end{smallmatrix}\right)`` and ``Ψ = (|Ψ^\prime⟩, |Ψ⟩)``.


## [The exponential of the gradient generator](@id Background-Exponential)

Let ``\hat{H}(ϵ)`` be constant in time and continuously differentiable in the real control value ``ϵ``, with ``\hat{μ} = \partial \hat{H} / \partial ϵ``. We write ``\hat{U}(t) = e^{-\ii \hat{H}(ϵ) t}``, leaving the dependence on ``ϵ`` implicit, and ``\partial_ϵ \hat{U} \equiv \partial \hat{U} / \partial ϵ``. Note that ``\hat{H}`` and ``\hat{μ}`` generally do not commute.

**Proposition 1.** For every real ``t``,

```math
\begin{equation}\label{eq:gradgen-exp}
e^{-\ii G t} = \begin{pmatrix} \hat{U}(t) & \partial_ϵ \hat{U}(t) \\ 0 & \hat{U}(t) \end{pmatrix}\,.
\end{equation}
```

In particular, for ``t = dt``, ``e^{-\ii G dt} (0, |Ψ⟩) = (\partial_ϵ \hat{U} |Ψ⟩, \hat{U} |Ψ⟩)``, which together with the [reduction to a single control](@ref Background-Reduction) proves Eq. \eqref{eq:main}.

*Proof.* The exponential series ``\hat{U}(t) = \sum_k (-\ii t)^k \hat{H}^k / k!`` with ``\hat{H} = \hat{H}(ϵ)`` and its derivatives converge uniformly for ``t`` and ``ϵ`` in bounded intervals, so ``\partial_t`` and ``\partial_ϵ`` commute. Applying ``\partial_ϵ`` to the equation of motion ``\ii \partial_t \hat{U} = \hat{H} \hat{U}`` with the product rule gives

```math
\begin{equation}\label{eq:ode-dU}
\ii \frac{\partial}{\partial t} \partial_ϵ \hat{U}(t) = \hat{μ} \hat{U}(t) + \hat{H} \partial_ϵ \hat{U}(t)\,.
\end{equation}
```

Let ``W(t)`` be the right-hand side of Eq. \eqref{eq:gradgen-exp}. Since ``\hat{U}(0) = \mathbb{1}`` for all ``ϵ``, ``\partial_ϵ \hat{U}(0) = 0``, and thus ``W(0) = \mathbb{1}``. With Eq. \eqref{eq:ode-dU},

```math
\begin{equation*}
\ii \frac{d W}{dt}
= \begin{pmatrix} \hat{H} \hat{U} & \hat{H} \partial_ϵ \hat{U} + \hat{μ} \hat{U} \\ 0 & \hat{H} \hat{U} \end{pmatrix}
= \begin{pmatrix} \hat{H} & \hat{μ} \\ 0 & \hat{H} \end{pmatrix}
  \begin{pmatrix} \hat{U} & \partial_ϵ \hat{U} \\ 0 & \hat{U} \end{pmatrix}
= G W(t)\,.
\end{equation*}
```

The matrix ``e^{-\ii G t}`` solves the same linear differential equation with the same initial value, so ``W(t) = e^{-\ii G t}`` by uniqueness of the solution. ``\square``

Since ``t`` may be negative, Proposition 1 also covers the backward propagation in GRAPE, with ``\hat{H}`` replaced by ``\hat{H}^\dagger`` and ``dt`` by ``dt^{(-)} < 0``: as ``\partial \hat{H}^\dagger / \partial ϵ = \hat{μ}^\dagger`` for a real ``ϵ``, the gradient generator ``G[\hat{H}^\dagger]`` has ``\hat{μ}^\dagger`` in its upper right block.

Eq. \eqref{eq:ode-dU} is an inhomogeneous linear equation with zero initial value. Its solution by variation of constants is

```math
\begin{equation}\label{eq:duhamel}
\partial_ϵ \hat{U}(t)
= -\ii \int_0^{t} e^{-\ii \hat{H} (t - s)} \hat{μ} e^{-\ii \hat{H} s} ds\,.
\end{equation}
```

That integrals of this form are blocks of the exponential of a block-triangular matrix is the content of [VanLoanITAC1978; Theorem 1](@cite), here with ``A_1 = A_2 = -\ii \hat{H}`` and ``B_1 = -\ii \hat{μ}``. [Van Loan](@cite VanLoanITAC1978) introduced the block-triangular exponential to evaluate integrals of matrix exponentials in sampled-data control theory. [GoodwinJCP2015](@citet) applied this "auxiliary matrix" formalism to obtain exact first and second derivatives of the time-slice propagators in GRAPE. Expanding both exponentials in Eq. \eqref{eq:duhamel} and integrating term by term gives the Taylor series

```math
\begin{equation}\label{eq:taylor}
\partial_ϵ \hat{U}(t)
= \sum_{m=1}^{\infty} \frac{(-\ii t)^m}{m!}
  \sum_{m^\prime=0}^{m-1} \hat{H}^{m^\prime} \hat{μ} \hat{H}^{m-m^\prime-1}\,,
\end{equation}
```

which for ``t = dt`` is the [expansion used by `GRAPE.jl` with `gradient_method=:taylor`](@extref GRAPE :label:`Overview-Taylor`). Its first term gives the approximation ``\partial_ϵ \hat{U} \approx -\ii dt \hat{μ}`` of the original GRAPE method [KhanejaJMR2005](@cite). Even ``-\ii dt \hat{μ} \hat{U}`` is exact only if ``\hat{H}`` and ``\hat{μ}`` commute. The gradient generator avoids both the integral and the truncation of the series.

### Non-zero gradient states and multiple time steps

Applying Eq. \eqref{eq:gradgen-exp} to a gradient vector with an arbitrary gradient state gives the following result.

**Corollary 2.** For any ``|Ψ^\prime⟩`` and ``|Ψ⟩``,

```math
\begin{equation}\label{eq:chain}
e^{-\ii G dt} \begin{pmatrix} |Ψ^\prime⟩ \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix} \hat{U} |Ψ^\prime⟩ + \partial_ϵ \hat{U} |Ψ⟩ \\ \hat{U} |Ψ⟩ \end{pmatrix}\,.
\end{equation}
```

If the state ``|Ψ⟩ = |Ψ(ϵ)⟩`` itself depends on ``ϵ`` and ``|Ψ^\prime⟩ = \partial_ϵ |Ψ⟩``, the upper component is ``\partial_ϵ (\hat{U} |Ψ⟩)``, by the product rule. That is, a gradient vector of the form ``(\partial_ϵ |Ψ⟩, |Ψ⟩)`` is propagated to a gradient vector of the same form.

Now consider a piecewise-constant propagation in which the generators ``\hat{H}_n(ϵ)`` of *all* time steps depend on the same parameter ``ϵ``. With ``G_n = G[\hat{H}_n]``, applying Eq. \eqref{eq:chain} once per time step gives

```math
\begin{equation}\label{eq:multistep}
e^{-\ii G_{N_T} dt_{N_T}} \cdots e^{-\ii G_1 dt_1}
\begin{pmatrix} \partial_ϵ |Ψ(0)⟩ \\ |Ψ(0)⟩ \end{pmatrix}
= \begin{pmatrix} \partial_ϵ |Ψ(T)⟩ \\ |Ψ(T)⟩ \end{pmatrix}\,,
\end{equation}
```

where ``|Ψ(T)⟩ = \hat{U}_{N_T} \cdots \hat{U}_1 |Ψ(0)⟩``, and ``\partial_ϵ |Ψ(0)⟩ = 0`` if the initial state does not depend on ``ϵ``. Propagating the gradient vector without ever resetting the gradient state accumulates the product rule,

```math
\begin{equation}\label{eq:multistep-sum}
\partial_ϵ |Ψ(T)⟩
= \sum_{n=1}^{N_T} \hat{U}_{N_T} \cdots \hat{U}_{n+1} \frac{\partial \hat{U}_n}{\partial ϵ} \hat{U}_{n-1} \cdots \hat{U}_1 |Ψ(0)⟩
+ \hat{U}_{N_T} \cdots \hat{U}_1 \partial_ϵ |Ψ(0)⟩\,,
\end{equation}
```

one time step at a time. GRAPE is the opposite case: each control value ``ϵ_{nl}`` enters only the single propagator ``\hat{U}_n``, so its derivative is a single term of the sum in Eq. \eqref{eq:multistep-sum}. Accordingly, GRAPE resets the gradient states to zero before each time step (with [`resetgradvec!`](@ref)) and uses Eq. \eqref{eq:main} for one time step at a time.


## [Dual numbers and forward-mode differentiation](@id Background-Dual)

Proposition 1 has a second interpretation: the gradient vector is the state together with its "dual part", and propagation under the gradient generator is forward-mode automatic differentiation [GriewankWalther2008](@cite). This section makes this statement precise, which yields an independent proof of Proposition 1 and, more importantly, a statement about the numerical methods that are used in practice to propagate gradient vectors.

### Dual numbers

A dual number is an expression ``a + b η`` with real or complex numbers ``a`` (the value) and ``b`` (the dual part), and a formal unit ``η \neq 0`` that commutes with all numbers and satisfies ``η^2 = 0``. Dual numbers are added componentwise and multiplied according to

```math
\begin{equation}\label{eq:dual-mult}
(a + b η)(c + d η) = ac + (ad + bc) η\,.
\end{equation}
```

The dual part of a product follows the product rule. By the binomial theorem, ``(x + η)^m = x^m + m x^{m-1} η``, since all higher powers of ``η`` vanish. Consequently, any polynomial ``f`` satisfies

```math
\begin{equation}\label{eq:dual-f}
f(x + η) = f(x) + f^\prime(x) η\,.
\end{equation}
```

For example, ``f(x) = x^3 - 2x`` at ``x = 2`` gives ``(2 + η)^3 - 2(2 + η) = (8 + 12η) - (4 + 2η) = 4 + 10η``, which contains ``f(2) = 4`` and ``f^\prime(2) = 10``. The same holds for any computation composed of arithmetic operations (with ``(a + bη)^{-1} = a^{-1} - a^{-2} b η`` for division): evaluating it at the input ``x + η`` yields its value and its derivative with respect to ``x``. A differentiable function ``g`` that is not composed of arithmetic operations is extended to dual numbers by defining ``g(a + b η) = g(a) + g^\prime(a) b η``, consistent with the chain rule. This is forward-mode automatic differentiation, as implemented, e.g., by the `Dual` number type of the [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package.

### Gradient vectors as dual states

Operators and vectors may have dual-number entries. Since ``η`` is a scalar, it commutes with all operators, and a dual operator can be written uniquely as ``\hat{A} + η \hat{B}`` with ordinary complex matrices ``\hat{A}`` and ``\hat{B}``. Likewise, a dual vector is ``|Ψ⟩ + η |Ψ^\prime⟩``. Multiplication follows Eq. \eqref{eq:dual-mult}, keeping the order of the operators:

```math
\begin{equation}\label{eq:dual-op-mult}
\begin{split}
(\hat{A}_1 + η \hat{B}_1)(\hat{A}_2 + η \hat{B}_2) &= \hat{A}_1 \hat{A}_2 + η (\hat{A}_1 \hat{B}_2 + \hat{B}_1 \hat{A}_2)\,, \\
(\hat{A} + η \hat{B})(|Ψ⟩ + η |Ψ^\prime⟩) &= \hat{A} |Ψ⟩ + η (\hat{A} |Ψ^\prime⟩ + \hat{B} |Ψ⟩)\,.
\end{split}
\end{equation}
```

To compare dual operators and dual vectors with gradient generators and gradient vectors, we represent them as ordinary block matrices and block vectors of twice the dimension. We call this representation ``ι`` (for "injection"). It places the value ``\hat{A}`` of a dual operator in both diagonal blocks and the dual part ``\hat{B}`` in the upper right block. For a dual vector, it stacks the dual part ``|Ψ^\prime⟩`` on top of the value ``|Ψ⟩``, in the same order as in a gradient vector:

```math
\begin{equation}\label{eq:iota}
ι(\hat{A} + η \hat{B}) = \begin{pmatrix} \hat{A} & \hat{B} \\ 0 & \hat{A} \end{pmatrix}\,,
\qquad
ι(|Ψ⟩ + η |Ψ^\prime⟩) = \begin{pmatrix} |Ψ^\prime⟩ \\ |Ψ⟩ \end{pmatrix}\,.
\end{equation}
```

Applied to a scalar dual number instead of an operator, Eq. \eqref{eq:iota} gives ``ι(a + bη) = \left(\begin{smallmatrix} a & b \\ 0 & a \end{smallmatrix}\right)``, the standard [matrix representation of dual numbers](https://en.wikipedia.org/wiki/Dual_number#Matrix_representation). In particular, ``ι(η) = \left(\begin{smallmatrix} 0 & 1 \\ 0 & 0 \end{smallmatrix}\right)`` squares to zero, like ``η``.

**Proposition 3.** The map ``ι`` is linear and injective, and for all dual operators ``X``, ``Y`` and dual vectors ``v``,

```math
\begin{equation}\label{eq:iota-homomorphism}
ι(X Y) = ι(X) ι(Y)\,,
\qquad
ι(X v) = ι(X) ι(v)\,,
\qquad
ι(\mathbb{1}) = \mathbb{1}\,.
\end{equation}
```

If ``X = \hat{A} + η \hat{B}`` with an invertible ``\hat{A}``, then ``X^{-1} = \hat{A}^{-1} - η \hat{A}^{-1} \hat{B} \hat{A}^{-1}`` and ``ι(X^{-1}) = ι(X)^{-1}``.

*Proof.* Linearity and injectivity are evident from Eq. \eqref{eq:iota}. Block multiplication gives

```math
\begin{equation*}
\begin{pmatrix} \hat{A}_1 & \hat{B}_1 \\ 0 & \hat{A}_1 \end{pmatrix}
\begin{pmatrix} \hat{A}_2 & \hat{B}_2 \\ 0 & \hat{A}_2 \end{pmatrix}
= \begin{pmatrix} \hat{A}_1 \hat{A}_2 & \hat{A}_1 \hat{B}_2 + \hat{B}_1 \hat{A}_2 \\ 0 & \hat{A}_1 \hat{A}_2 \end{pmatrix}\,,
\qquad
\begin{pmatrix} \hat{A} & \hat{B} \\ 0 & \hat{A} \end{pmatrix}
\begin{pmatrix} |Ψ^\prime⟩ \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix} \hat{A} |Ψ^\prime⟩ + \hat{B} |Ψ⟩ \\ \hat{A} |Ψ⟩ \end{pmatrix}\,,
\end{equation*}
```

which is ``ι`` applied to the right-hand sides of Eq. \eqref{eq:dual-op-mult}. The inverse follows from multiplying out ``X X^{-1} = \mathbb{1}`` and applying ``ι``. ``\square``

Thus, computing with dual operators and dual vectors and computing with the corresponding block matrices and block vectors is the same thing. Since ``ι`` is linear on a finite-dimensional space, it also commutes with limits, and hence with convergent series.

For a generator ``\hat{H}(ϵ)`` that is differentiable in the real control ``ϵ``, we evaluate it at the dual control value ``ϵ + η``,

```math
\begin{equation}\label{eq:H-dual}
\hat{H}(ϵ + η) \equiv \hat{H}(ϵ) + η \frac{\partial \hat{H}}{\partial ϵ} = \hat{H} + η \hat{μ}\,.
\end{equation}
```

For a linear control, or any polynomial dependence on ``ϵ``, this is simply the evaluation of the expression for ``\hat{H}`` in dual arithmetic. Comparing with Eq. \eqref{eq:iota} gives the following identification.

**Proposition 4.** The gradient generator is the image of the dynamical generator evaluated at the dual control value, and the gradient vector is the image of a dual state:

```math
\begin{equation}\label{eq:gradgen-dual}
G[\hat{H}] = ι\big(\hat{H}(ϵ + η)\big)\,,
\qquad
Ψ = \begin{pmatrix} |Ψ^\prime⟩ \\ |Ψ⟩ \end{pmatrix} = ι\big(|Ψ⟩ + η |Ψ^\prime⟩\big)\,.
\end{equation}
```

A gradient vector with a zero gradient state is the image of a state that does not depend on ``ϵ``. For ``L`` controls, ``G[\hat{H}]`` and ``(|Ψ^\prime_1⟩, \dots, |Ψ^\prime_L⟩, |Ψ⟩)`` are the images of ``\hat{H} + \sum_l η_l \hat{μ}_l`` and ``|Ψ⟩ + \sum_l η_l |Ψ^\prime_l⟩`` in the same way, with ``L`` dual units satisfying ``η_l η_{l^\prime} = 0``. This is the dual-number form of the [reduction to a single control](@ref Background-Reduction). For the backward propagation, ``(\hat{A} + η \hat{B})^\dagger = \hat{A}^\dagger + η \hat{B}^\dagger``, so ``G[\hat{H}^\dagger] = ι(\hat{H}^\dagger(ϵ + η))``.

### Functions of dual operators

Eq. \eqref{eq:dual-f} carries over to operators, with one subtlety: ``\hat{A}`` and ``\hat{B}`` generally do not commute, so ``(\hat{A} + η \hat{B})^m \neq \hat{A}^m + m \hat{A}^{m-1} \hat{B} η``. The correct generalization of ``f^\prime(x)`` is the directional derivative

```math
\begin{equation}\label{eq:frechet}
Df(\hat{A})[\hat{B}] \equiv \frac{d}{ds} f(\hat{A} + s \hat{B}) \Big\vert_{s=0}\,.
\end{equation}
```

**Theorem 5.** Let ``f(z) = \sum_{m=0}^{\infty} c_m z^m`` be a power series with complex coefficients and radius of convergence ``R``, and let ``\hat{A}`` and ``\hat{B}`` be complex ``N \times N`` matrices with ``\Vert \hat{A} \Vert < R`` in a submultiplicative norm. Then

```math
\begin{equation}\label{eq:analytic}
\begin{split}
f(\hat{A} + η \hat{B}) &= f(\hat{A}) + η Df(\hat{A})[\hat{B}]\,, \\
Df(\hat{A})[\hat{B}] &= \sum_{m=1}^{\infty} c_m \sum_{k=0}^{m-1} \hat{A}^k \hat{B} \hat{A}^{m-1-k}\,.
\end{split}
\end{equation}
```

*Proof.* Let ``D_m = \sum_{k=0}^{m-1} \hat{A}^k \hat{B} \hat{A}^{m-1-k}``. By induction with Eq. \eqref{eq:dual-op-mult}, ``(\hat{A} + η \hat{B})^m = \hat{A}^m + η D_m``, since ``(\hat{A}^m + η D_m)(\hat{A} + η \hat{B}) = \hat{A}^{m+1} + η (\hat{A}^m \hat{B} + D_m \hat{A})``. Only ``η^2 = 0`` and the fact that ``η`` commutes with everything enter; the order of ``\hat{A}`` and ``\hat{B}`` is preserved. Expanding ``(\hat{A} + s \hat{B})^m`` into ordered products, the terms of first order in ``s`` are exactly those of ``s D_m``. As ``\Vert D_m \Vert \le m \Vert \hat{A} \Vert^{m-1} \Vert \hat{B} \Vert``, all series converge absolutely for small ``|s|``, and may be rearranged into ``f(\hat{A} + s \hat{B}) = f(\hat{A}) + s \sum_m c_m D_m + \mathcal{O}(s^2)``. ``\square``

For the exponential, ``R = \infty``. With ``f(z) = e^{-\ii z dt}``, ``\hat{A} = \hat{H}``, and ``\hat{B} = \hat{μ}``, the dual part in Eq. \eqref{eq:analytic} is the Taylor series \eqref{eq:taylor} for ``t = dt``, and it is the derivative of ``e^{-\ii \hat{H}(ϵ) dt}`` with respect to ``ϵ`` by the chain rule. With Propositions 3 and 4,

```math
\begin{equation}\label{eq:expG-dual}
e^{-\ii G dt} = ι\Big(e^{-\ii \hat{H}(ϵ + η) dt}\Big) = ι\big(\hat{U} + η \partial_ϵ \hat{U}\big) = \begin{pmatrix} \hat{U} & \partial_ϵ \hat{U} \\ 0 & \hat{U} \end{pmatrix}\,,
\end{equation}
```

which is an independent proof of Proposition 1. Applied to a gradient vector, ``(\hat{U} + η \partial_ϵ \hat{U})(|Ψ⟩ + η |Ψ^\prime⟩)`` is the product rule of Corollary 2, and over several time steps, the dual part of the product of the dual propagators ``\hat{U}_n + η \partial_ϵ \hat{U}_n`` is Eq. \eqref{eq:multistep-sum}.

### [Forward-mode differentiation of numerical methods](@id Background-Numerical-Methods)

The dual-number picture extends beyond the exact exponential, to the numerical methods that are actually used to propagate gradient vectors.

**Proposition 6.** Consider a finite computation that produces a vector or an operator from one or more generators ``\hat{H}(t_i; ϵ)``, a state ``|Ψ(ϵ)⟩``, and any number of ``ϵ``-independent operators and vectors, using only linear combinations with ``ϵ``-independent coefficients, products of operators, application of operators to vectors, and the solution of linear systems. Run the same computation with each ``\hat{H}(t_i)`` replaced by ``G[\hat{H}(t_i)]``, ``|Ψ⟩`` by ``(\partial_ϵ |Ψ⟩, |Ψ⟩)``, every ``ϵ``-independent operator ``\hat{O}`` by ``\mathrm{diag}(\hat{O}, \hat{O})``, and every ``ϵ``-independent vector ``|Φ⟩`` by ``(0, |Φ⟩)``. Then, in exact arithmetic, a resulting vector ``|Φ(ϵ)⟩`` is replaced by ``(\partial_ϵ |Φ⟩, |Φ⟩)``, and a resulting operator ``\hat{X}(ϵ)`` by ``\left(\begin{smallmatrix} \hat{X} & \partial_ϵ \hat{X} \\ 0 & \hat{X} \end{smallmatrix}\right)``.

*Proof.* By Proposition 4, the replacements are ``ι`` of the inputs evaluated at ``ϵ + η``, each of the form "value ``+ η`` derivative". By Proposition 3, running the computation on the block objects is the same as running it in dual arithmetic. Each allowed operation preserves the form "value ``+ η`` derivative": linear combinations because ``\partial_ϵ`` is linear, products because Eq. \eqref{eq:dual-op-mult} is the product rule, and inverses because the dual part of ``X^{-1}`` in Proposition 3 is ``\partial_ϵ (X^{-1})``. ``\square``

Two kinds of operations fall outside Proposition 6. First, operations that are not complex-linear in the state, such as inner products and norms (e.g., in an orthogonalization or an error estimate), act on the gradient vector as a whole, not as dual arithmetic would. Second, choices that a method makes based on the data, such as the number of terms of an expansion, a spectral range, or an adaptive time step, are made for ``G`` rather than for ``\hat{H}``. If such choices are fixed before the computation, Proposition 6 applies with these choices frozen.

Most propagation methods for a piecewise-constant generator approximate ``\hat{U} = e^{-\ii \hat{H} dt}`` by a polynomial in ``\hat{H}``. For these, Proposition 6 gives a strong statement.

**Corollary 7.** Let ``p(z) = \sum_{k=0}^{K} c_k z^k`` be a polynomial whose coefficients do not depend on ``ϵ``. Then ``p(G)`` maps ``(\partial_ϵ |Ψ⟩, |Ψ⟩)`` to ``(\partial_ϵ [p(\hat{H}(ϵ)) |Ψ⟩], p(\hat{H}(ϵ)) |Ψ⟩)``.

The gradient state is thus the exact derivative of the *approximate* propagated state, not an approximation of the derivative of the exact propagated state. The error of the propagation enters the state and the gradient consistently: up to floating-point rounding, the gradient is the derivative of the function of ``ϵ`` that the propagation actually evaluates, no matter how few terms the polynomial has. How well it approximates ``\partial_ϵ \hat{U} |Ψ⟩`` depends on how well ``p^\prime`` approximates the derivative of ``z \mapsto e^{-\ii z dt}`` on the spectrum of ``\hat{H}``.

The [Chebychev propagator](@extref QuantumPropagators :label:`method_cheby`) in `QuantumPropagators` is of this form. It shifts and scales ``\hat{H}`` with a spectral range ``[E_{\min}, E_{\min} + Δ]``, sums Chebychev polynomials with Bessel-function coefficients that depend only on ``Δ dt``, and multiplies with a phase factor. The spectral range is determined when the propagator is initialized, from the minimum and maximum values of the controls, and it does not change with the control values in a particular time step. Thus, Corollary 7 applies to every time step. For a gradient generator, the spectral range is estimated for ``G`` itself. Since ``G`` is block upper-triangular with ``\hat{H}`` in all diagonal blocks, ``\det(G - λ \mathbb{1}) = \det(\hat{H} - λ \mathbb{1})^{L+1}``, so ``G`` has the same eigenvalues as ``\hat{H}``, and the same Chebychev coefficients apply. In particular, the eigenvalues of ``G`` are real for a Hermitian ``\hat{H}``, even though ``G`` is not Hermitian. (In general, ``G`` is not diagonalizable: by Proposition 4, its eigenvalues are the dual numbers ``λ_j + η \partial_ϵ λ_j``, and each one with a non-zero dual part corresponds to a ``2 \times 2`` Jordan block. This is irrelevant for polynomials.)

Corollary 7 does not apply to methods whose polynomial depends on the propagated vector. The [Newton propagator](@extref QuantumPropagators :label:`method_newton`) builds a Krylov subspace with an Arnoldi iteration, which uses inner products of the full gradient vector, and chooses its interpolation points from the resulting Ritz values. The propagation of ``Ψ`` under ``G`` is then not the derivative of the Newton propagation of ``|Ψ⟩`` under ``\hat{H}``. Instead, it is an approximation of ``e^{-\ii G dt} Ψ`` that converges together with the propagation, so the gradient states are accurate to the precision of the propagation. The same holds for Krylov methods in general.

### Relation to automatic differentiation

The preceding results confirm that a gradient vector ``(|Ψ^\prime⟩, |Ψ⟩)`` is the faithful matrix representation of the dual state ``|Ψ⟩ + η |Ψ^\prime⟩``, and the gradient generator is the representation of the dynamical generator at the dual control value ``ϵ + η`` (Proposition 4). Propagating a gradient vector under a gradient generator is forward-mode automatic differentiation of the propagation with respect to ``ϵ``. For the exact propagator, this gives ``(\partial_ϵ (\hat{U} |Ψ⟩), \hat{U} |Ψ⟩)``. For a numerical method built from linear algebra with ``ϵ``-independent coefficients, such as a Chebychev propagator, it gives the exact derivative of the approximate result. For methods whose internal choices depend on the propagated vector, such as Krylov methods, it gives an approximation of the exact derivative that converges together with the propagation.

GRAPE combines this forward mode with the reverse mode. The gradient with respect to all ``ϵ_{nl}`` is obtained from one forward propagation of ``|Ψ⟩`` and one backward propagation of ``|χ⟩``, independent of the number ``N_T`` of time steps. That backward propagation is reverse-mode (adjoint) differentiation across the time steps. Within each time step, the gradient generator ``G[\hat{H}_n^\dagger]`` provides the derivative of the single-step propagator in forward mode, for all ``L`` controls at once, with the gradient states reset before each step [GoerzQ2022](@cite).


## [Time-dependent generators and parameters](@id Background-Time-Dependent)

So far, the generator was constant within a time step, and we differentiated with respect to the value of a control. Both restrictions can be lifted. Let ``\hat{H}(t; u)`` depend on time ``t \in [0, T]`` and on a real parameter ``u``. Typically, ``u`` is one of the [control parameters](@extref QuantumControl :label:`Control-Parameters`) ``\{u_n\}`` that the control functions depend on, ``ϵ_l(t) = ϵ_l(\{u_n\}, t)``: the value of a [pulse](@extref QuantumControl :label:`Pulse`) on one interval of the time grid, as in GRAPE, or, e.g., the amplitude or width of an analytic pulse shape. It may also be a parameter of the drift term or of the dissipators of a Liouvillian. We assume that both ``\hat{H}`` and ``\partial_u \hat{H} \equiv \partial \hat{H} / \partial u`` are continuous in ``(t, u)`` on each interval of a fixed (``u``-independent) partition of ``[0, T]``, which covers smooth control functions as well as controls that are piecewise constant on a fixed time grid. The time evolution operator is the solution of

```math
\begin{equation}\label{eq:tdse-U}
\ii \frac{\partial}{\partial t} \hat{U}(t, t_0; u) = \hat{H}(t; u) \hat{U}(t, t_0; u)\,,
\qquad \hat{U}(t_0, t_0; u) = \mathbb{1}\,.
\end{equation}
```

The gradient generator for the parameter ``u`` is

```math
\begin{equation}\label{eq:gradgen-u}
G_u(t) = \begin{pmatrix} \hat{H}(t; u) & \partial_u \hat{H}(t; u) \\ 0 & \hat{H}(t; u) \end{pmatrix}\,,
\end{equation}
```

and ``\hat{U}_G(t, t_0)`` is the time evolution operator for ``\ii \frac{d}{dt} Ψ(t) = G_u(t) Ψ(t)``.

**Proposition 8.** ``\hat{U}(t, t_0; u)`` is continuously differentiable in ``u``, and

```math
\begin{equation}\label{eq:gradgen-u-prop}
\hat{U}_G(t, t_0) = \begin{pmatrix} \hat{U}(t, t_0) & \partial_u \hat{U}(t, t_0) \\ 0 & \hat{U}(t, t_0) \end{pmatrix}\,.
\end{equation}
```

*Proof.* By the standard theorem on the differentiable dependence of solutions of ordinary differential equations on parameters (applied to one interval of the partition after the other), ``\hat{U}`` is continuously differentiable in ``u``, and ``\partial_u \hat{U}`` solves the equation obtained by differentiating Eq. \eqref{eq:tdse-U},

```math
\begin{equation}\label{eq:variational}
\ii \frac{\partial}{\partial t} \partial_u \hat{U}(t, t_0) = \partial_u \hat{H}(t) \hat{U}(t, t_0) + \hat{H}(t) \partial_u \hat{U}(t, t_0)\,,
\qquad \partial_u \hat{U}(t_0, t_0) = 0\,.
\end{equation}
```

The rest of the proof is identical to that of Proposition 1: the right-hand side ``W(t)`` of Eq. \eqref{eq:gradgen-u-prop} satisfies ``W(t_0) = \mathbb{1}`` and ``\ii \frac{d}{dt} W = G_u(t) W``, as does ``\hat{U}_G(t, t_0)``. ``\square``

For a generator that is constant in time, Proposition 8 reduces to Proposition 1. As in Corollary 2, applying Eq. \eqref{eq:gradgen-u-prop} to a gradient vector gives

```math
\begin{equation}\label{eq:gradvec-u-prop}
\hat{U}_G(t, t_0) \begin{pmatrix} \partial_u |Ψ(t_0)⟩ \\ |Ψ(t_0)⟩ \end{pmatrix}
= \begin{pmatrix} \partial_u |Ψ(t)⟩ \\ |Ψ(t)⟩ \end{pmatrix}
\end{equation}
```

by the product rule. Since ``\hat{U}_G(t_2, t_0) = \hat{U}_G(t_2, t_1) \hat{U}_G(t_1, t_0)``, the gradient vector may be propagated over ``[0, T]`` in any number of pieces, without resetting the gradient state. Solving Eq. \eqref{eq:variational} by variation of constants generalizes Eq. \eqref{eq:duhamel} to

```math
\begin{equation}\label{eq:duhamel-td}
\partial_u \hat{U}(T, 0) = -\ii \int_0^T \hat{U}(T, t) \partial_u \hat{H}(t) \hat{U}(t, 0) dt\,.
\end{equation}
```

The propagation under ``G_u(t)`` evaluates this integral without discretizing it.

### [Parameterized controls](@id Background-Parameters)

For ``M`` control parameters ``u_1, \dots, u_M``, the gradient generator has the same structure as Eq. \eqref{eq:gradgen}, with ``\partial_{u_n} \hat{H}(t)`` in place of ``\hat{μ}_l``, and acts on gradient vectors ``Ψ = (|Ψ^\prime_1⟩, \dots, |Ψ^\prime_M⟩, |Ψ⟩)`` of dimension ``N(M+1)``. As in the [reduction to a single control](@ref Background-Reduction), the pair ``(|Ψ^\prime_n(t)⟩, |Ψ(t)⟩)`` evolves under the single-parameter generator ``G_{u_n}(t)``. By Eq. \eqref{eq:gradvec-u-prop}, propagating ``(0, \dots, 0, |Ψ(0)⟩)`` thus yields all ``M`` derivatives ``|Ψ^\prime_n(T)⟩ = \partial_{u_n} |Ψ(T)⟩`` at once.

Typically, the parameters enter through the control functions ``ϵ_l(\{u_n\}, t)``, so that ``\hat{H}(t) = \hat{H}(\{ϵ_l(\{u_n\}, t)\})``. By the chain rule, the derivative of the generator is the [control derivative](@extref QuantumControl :label:`Control-Derivative`) times the [parameter derivative](@extref QuantumControl :label:`Parameter-Derivative`),

```math
\begin{equation}\label{eq:chain-u}
\frac{\partial \hat{H}(t)}{\partial u_n} = \sum_{l=1}^{L} \hat{μ}_l(t) \frac{\partial ϵ_l(t)}{\partial u_n}\,,
\end{equation}
```

plus any explicit dependence of ``\hat{H}`` on ``u_n``. The blocks of the gradient generator are thus built from the same control derivatives ``\hat{μ}_l`` as for the gradient generator of the controls, with the scalar, time-dependent parameter derivatives as weights. Applying ``G(t)`` to a gradient vector requires ``M + 1`` applications of ``\hat{H}``, but only ``L`` applications of the ``\hat{μ}_l``.

This is the setting of optimal control methods for analytic or otherwise parameterized control functions. GOAT [MachnesPRL2018](@cite) optimizes analytic control functions with few parameters by integrating the coupled equations of motion for ``\hat{U}`` and ``\partial_u \hat{U}`` forward in time with an ODE solver, [MachnesPRL2018; Eq. (7)](@cite), which is Eq. \eqref{eq:gradgen-u} with the order of the blocks reversed. GROUP [SorensenPRA2018](@cite) expands the control in a chopped basis, ``ϵ(t) = ϵ_0(t) + S(t) \sum_n u_n f_n(t)``, and GRAFS [LucarelliPRA2018](@cite) expands piecewise-constant controls in band-limited Slepian sequences. Both obtain the gradient with respect to the expansion coefficients from a GRAPE-type adjoint gradient via the chain rule, [SorensenPRA2018; Eq. (16)](@cite) and [LucarelliPRA2018; Eqs. (18, 19)](@cite). The gradient generator for the parameters ``u_n`` is the mathematical basis for implementing GOAT-type methods with `QuantumGradientGenerators`.

GRAPE is the special case where the control parameters are the pulse values themselves, ``\{u_n\} = \{ϵ_{nl}\}``. Then ``\partial ϵ_l(t) / \partial ϵ_{n^\prime l^\prime}`` is ``δ_{l l^\prime}`` on the interval ``n^\prime`` and zero elsewhere. A forward propagation would carry ``N_T L`` gradient states, each driven only during its own time interval and evolving freely under ``\hat{H}`` afterwards. In the gradient of a functional (below), the free evolution can be moved onto the backward-propagated state ``|χ⟩``. This is why GRAPE needs only ``L`` gradient states for one time step at a time. For parameters that affect the controls at all times, as in GOAT, the forward propagation carries all ``M`` gradient states over the entire time interval, but requires no backward propagation and no storage of states at intermediate times.

### Gradients of functionals

For a final-time functional ``J_T(\{|Ψ_k(T)⟩\})`` of the states of several trajectories ``k``, we follow the conventions of the GRAPE documentation, with [Wirtinger derivatives](@extref GRAPE :label:`Wirtinger-Vectors`) and ``|χ_k(T)⟩ \equiv - \partial J_T / \partial ⟨Ψ_k(T)|``. With the gradient states ``|Ψ^\prime_{kn}(T)⟩ = \partial_{u_n} |Ψ_k(T)⟩`` from the forward propagation of the ``k``'th trajectory, the chain rule gives

```math
\begin{equation}\label{eq:grad-JT}
\frac{\partial J_T}{\partial u_n}
= 2 \Re \sum_k \frac{\partial J_T}{\partial |Ψ_k(T)⟩} \frac{\partial |Ψ_k(T)⟩}{\partial u_n}
= -2 \Re \sum_k ⟨χ_k(T) | Ψ^\prime_{kn}(T)⟩\,.
\end{equation}
```

This requires only the states and gradient states at final time. The ``|χ_k(T)⟩`` may be obtained analytically or by automatic differentiation of ``J_T`` alone, as in the semi-automatic differentiation of [GoerzQ2022](@citet). Inserting Eq. \eqref{eq:duhamel-td} gives the equivalent adjoint form

```math
\begin{equation}\label{eq:grad-adjoint}
\frac{\partial J_T}{\partial u_n}
= -2 \Im \sum_k \int_0^T ⟨χ_k(t) | \partial_{u_n} \hat{H}(t) | Ψ_k(t)⟩ dt\,,
\end{equation}
```

with the backward-propagated states ``|χ_k(t)⟩ = \hat{U}^\dagger(T, t) |χ_k(T)⟩``. This is the structure of the GROUP gradient [SorensenPRA2018; Eq. (16)](@cite). With Eq. \eqref{eq:chain-u}, the integrand requires only ``⟨χ_k(t) | \hat{μ}_l(t) | Ψ_k(t)⟩`` for the ``L`` controls. Eq. \eqref{eq:grad-adjoint} trades the ``M`` gradient states for a backward propagation and the storage of the forward-propagated states. Which form is more efficient depends on the number of parameters, the dimension of the system, and the available memory.

### Numerical propagation

Proposition 8 concerns the exact time evolution. In practice, the propagation under ``G_u(t)`` is numerical, either with a general ODE solver or with a piecewise-constant approximation. As before, it suffices to consider a single parameter ``u``.

**Proposition 9.** Consider a Runge–Kutta or linear multistep method (explicit or implicit, with exact solution of the linear systems for implicit methods) on a time grid that does not depend on ``u``. Applying the method to ``\ii \frac{d}{dt} Ψ = G_u(t) Ψ`` with the initial gradient vector ``(\partial_u |Ψ(0)⟩, |Ψ(0)⟩)`` yields gradient vectors ``(\partial_u |Ψ_n⟩, |Ψ_n⟩)``, where ``|Ψ_n⟩`` is the numerical solution of Eq. \eqref{eq:tdse} with ``\hat{H}(t; u)`` by the same method.

*Proof.* Each step of the method uses only linear combinations with coefficients that depend on the method and the time steps, applications of the generator at fixed times, and, for implicit methods, solutions of linear systems with matrices assembled linearly from ``\mathbb{1}`` and the generator at fixed times. Proposition 6 applies. ``\square``

Thus, differentiating the discretized propagation and discretizing the equation of motion for the derivative \eqref{eq:variational} give the same result, and the gradient is consistent with the computed state to machine precision. This requires that the time steps do not depend on ``u``. With an adaptive step size control, the gradient states are not the exact derivative of the computed state, but they still converge to ``\partial_u |Ψ(T)⟩`` as the tolerance decreases, provided the error control includes the gradient states.

The second approach is the piecewise-constant approximation that [`QuantumPropagators`](@extref QuantumPropagators :doc:`index`) uses for time-dependent generators. On each interval of the time grid, the generator is evaluated at a fixed time ``\tilde{t}_n``, by default the midpoint of the interval, giving ``\hat{H}_n(u) = \hat{H}(\tilde{t}_n; u)``, the propagators ``\hat{U}_n = e^{-\ii \hat{H}_n dt_n}``, and the approximate final state ``|\bar{Ψ}(T)⟩ = \hat{U}_{N_T} \cdots \hat{U}_1 |Ψ(0)⟩``.

**Proposition 10.** Let ``G_n = G_u(\tilde{t}_n)`` be the gradient generator evaluated at the same times ``\tilde{t}_n`` as the generator. Then

```math
\begin{equation}\label{eq:pwc}
e^{-\ii G_{N_T} dt_{N_T}} \cdots e^{-\ii G_1 dt_1}
\begin{pmatrix} \partial_u |Ψ(0)⟩ \\ |Ψ(0)⟩ \end{pmatrix}
= \begin{pmatrix} \partial_u |\bar{Ψ}(T)⟩ \\ |\bar{Ψ}(T)⟩ \end{pmatrix}\,,
\end{equation}
```

and as ``\max_n dt_n \rightarrow 0``, the right-hand side converges to ``(\partial_u |Ψ(T)⟩, |Ψ(T)⟩)``.

*Proof.* Since ``\tilde{t}_n`` does not depend on ``u``, ``G_n = G[\hat{H}_n]``, and Eq. \eqref{eq:pwc} is Eq. \eqref{eq:multistep}. The left-hand side is the time evolution under the piecewise-constant ``\bar{G}(t) = G_n`` for ``t \in (t_{n-1}, t_n]``. By variation of constants, ``\hat{U}_{\bar{G}}(T, 0) - \hat{U}_G(T, 0) = -\ii \int_0^T \hat{U}_{\bar{G}}(T, t) (\bar{G}(t) - G_u(t)) \hat{U}_G(t, 0) dt``, which vanishes as ``\max_n dt_n \rightarrow 0`` because ``G_u(t)`` is uniformly continuous on each interval of the partition. The convergence then follows from Proposition 8. ``\square``

The gradient states are thus the exact derivative of the piecewise-constant approximation, and they converge to the exact derivative together with it (to second order in ``dt`` for smooth controls and midpoint evaluation). Each step ``e^{-\ii G_n dt_n}`` is evaluated with one of the propagators in `QuantumPropagators`, so Eq. \eqref{eq:pwc} holds to the precision of that propagator, and exactly for a polynomial propagator with ``u``-independent coefficients, by Corollary 7. It is essential that ``\partial_u \hat{H}`` is evaluated at the same times ``\tilde{t}_n`` as ``\hat{H}``. Otherwise, the gradient states still converge to the exact derivative, but they are not the derivative of the computed state ``|\bar{Ψ}(T)⟩``, and the mismatch between the computed functional and its gradient, of the order of the time discretization error, can stall a gradient-based optimization close to convergence.

### Remarks

Nothing in this section relies on Hermiticity of ``\hat{H}``. All results apply to non-Hermitian Hamiltonians and to Liouvillians, where ``u`` may also parameterize the dissipators, and to a backward propagation under ``\hat{H}^\dagger(t)`` with the gradient generator ``G[\hat{H}^\dagger]``.

Second derivatives can be obtained by nesting: if ``\hat{H}`` is twice continuously differentiable, applying Proposition 8 to ``G_{u_1}(t)`` with respect to a second parameter ``u_2`` gives a generator of dimension ``4N`` whose time evolution operator contains ``\partial_{u_1} \partial_{u_2} \hat{U}`` in its upper right block. This is the matrix representation of numbers with two dual units, ``η_1^2 = η_2^2 = 0``. [GoodwinJCP2015](@citet) obtain second derivatives of single-step propagators from a larger block-triangular matrix in the same spirit, and [MachnesPRL2018](@citet) note that GOAT extends to second derivatives.
