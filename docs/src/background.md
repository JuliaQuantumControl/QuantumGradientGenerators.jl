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

!!! proposition "Proposition 1"

    [](@id Background-prop-gradgen-exp)For every real ``t``,

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

Since the block form of ``e^{-\ii G t}`` in Eq. \eqref{eq:gradgen-exp} holds for negative ``t`` as well, it also covers the backward propagation in GRAPE, with ``\hat{H}`` replaced by ``\hat{H}^\dagger`` and ``dt`` by ``dt^{(-)} < 0``: as ``\partial \hat{H}^\dagger / \partial ϵ = \hat{μ}^\dagger`` for a real ``ϵ``, the gradient generator ``G[\hat{H}^\dagger]`` has ``\hat{μ}^\dagger`` in its upper right block.

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

Applying the block form of ``e^{-\ii G t}``, Eq. \eqref{eq:gradgen-exp}, to a gradient vector with an arbitrary gradient state gives the following result.

!!! corollary "Corollary 2"

    [](@id Background-cor-gradient-state)For any ``|Ψ^\prime⟩`` and ``|Ψ⟩``,

    ```math
    \begin{equation}\label{eq:chain}
    e^{-\ii G dt} \begin{pmatrix} |Ψ^\prime⟩ \\ |Ψ⟩ \end{pmatrix}
    = \begin{pmatrix} \hat{U} |Ψ^\prime⟩ + \partial_ϵ \hat{U} |Ψ⟩ \\ \hat{U} |Ψ⟩ \end{pmatrix}\,.
    \end{equation}
    ```

If the state ``|Ψ⟩ = |Ψ(ϵ)⟩`` itself depends on ``ϵ`` and ``|Ψ^\prime⟩ = \partial_ϵ |Ψ⟩``, the upper component is ``\partial_ϵ (\hat{U} |Ψ⟩)``, by the product rule. That is, a gradient vector of the form ``(\partial_ϵ |Ψ⟩, |Ψ⟩)`` is propagated to a gradient vector of the same form.

Now consider a piecewise-constant propagation in which the generators ``\hat{H}_n(ϵ)`` of *all* time steps depend on the same parameter ``ϵ``. With ``G_n = G[\hat{H}_n]``, applying the single-step product rule, Eq. \eqref{eq:chain}, once per time step gives

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

The block form of ``e^{-\ii G t}``, with ``\hat{U}`` on the diagonal and ``\partial_ϵ \hat{U}`` in the upper right block ([Proposition 1](@ref Background-prop-gradgen-exp)), has a second interpretation: the gradient vector is the state together with its "dual part", and propagation under the gradient generator is forward-mode automatic differentiation [GriewankWalther2008](@cite). This section makes this statement precise, which yields an independent proof of that block form and, more importantly, a statement about the numerical methods that are used in practice to propagate gradient vectors.

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

Operators and vectors may have dual-number entries. Since ``η`` is a scalar, it commutes with all operators, and a dual operator can be written uniquely as ``\hat{A} + η \hat{B}`` with ordinary complex matrices ``\hat{A}`` and ``\hat{B}``. Likewise, a dual vector is ``|Ψ⟩ + η |Ψ^\prime⟩``. Multiplication follows the multiplication rule of dual numbers, Eq. \eqref{eq:dual-mult}, keeping the order of the operators:

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

!!! proposition "Proposition 3"

    [](@id Background-prop-iota)The map ``ι`` is linear and injective, and for all dual operators ``X``, ``Y`` and dual vectors ``v``,

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

For a linear control, or any polynomial dependence on ``ϵ``, this is simply the evaluation of the expression for ``\hat{H}`` in dual arithmetic. Comparing ``\hat{H} + η \hat{μ}`` with the block representation ``ι`` in Eq. \eqref{eq:iota} gives the following identification.

!!! proposition "Proposition 4"

    [](@id Background-prop-gradgen-dual)The gradient generator is the image of the dynamical generator evaluated at the dual control value, and the gradient vector is the image of a dual state:

    ```math
    \begin{equation}\label{eq:gradgen-dual}
    G[\hat{H}] = ι\big(\hat{H}(ϵ + η)\big)\,,
    \qquad
    Ψ = \begin{pmatrix} |Ψ^\prime⟩ \\ |Ψ⟩ \end{pmatrix} = ι\big(|Ψ⟩ + η |Ψ^\prime⟩\big)\,.
    \end{equation}
    ```

A gradient vector with a zero gradient state is the image of a state that does not depend on ``ϵ``. For ``L`` controls, ``G[\hat{H}]`` and ``(|Ψ^\prime_1⟩, \dots, |Ψ^\prime_L⟩, |Ψ⟩)`` are the images of ``\hat{H} + \sum_l η_l \hat{μ}_l`` and ``|Ψ⟩ + \sum_l η_l |Ψ^\prime_l⟩`` in the same way, with ``L`` dual units satisfying ``η_l η_{l^\prime} = 0``. This is the dual-number form of the [reduction to a single control](@ref Background-Reduction). For the backward propagation, ``(\hat{A} + η \hat{B})^\dagger = \hat{A}^\dagger + η \hat{B}^\dagger``, so ``G[\hat{H}^\dagger] = ι(\hat{H}^\dagger(ϵ + η))``.

### Functions of dual operators

The rule ``f(x + η) = f(x) + f^\prime(x) η`` for polynomials, Eq. \eqref{eq:dual-f}, carries over to operators, with one subtlety: ``\hat{A}`` and ``\hat{B}`` generally do not commute, so ``(\hat{A} + η \hat{B})^m \neq \hat{A}^m + m \hat{A}^{m-1} \hat{B} η``. The correct generalization of ``f^\prime(x)`` is the directional derivative

```math
\begin{equation}\label{eq:frechet}
Df(\hat{A})[\hat{B}] \equiv \frac{d}{ds} f(\hat{A} + s \hat{B}) \Big\vert_{s=0}\,.
\end{equation}
```

!!! theorem "Theorem 5"

    [](@id Background-theorem-dual-functions)Let ``f(z) = \sum_{m=0}^{\infty} c_m z^m`` be a power series with complex coefficients and radius of convergence ``R``, and let ``\hat{A}`` and ``\hat{B}`` be complex ``N \times N`` matrices with ``\Vert \hat{A} \Vert < R`` in a submultiplicative norm. Then

    ```math
    \begin{equation}\label{eq:analytic}
    \begin{split}
    f(\hat{A} + η \hat{B}) &= f(\hat{A}) + η Df(\hat{A})[\hat{B}]\,, \\
    Df(\hat{A})[\hat{B}] &= \sum_{m=1}^{\infty} c_m \sum_{k=0}^{m-1} \hat{A}^k \hat{B} \hat{A}^{m-1-k}\,.
    \end{split}
    \end{equation}
    ```

*Proof.* Let ``D_m = \sum_{k=0}^{m-1} \hat{A}^k \hat{B} \hat{A}^{m-1-k}``. By induction with Eq. \eqref{eq:dual-op-mult}, ``(\hat{A} + η \hat{B})^m = \hat{A}^m + η D_m``, since ``(\hat{A}^m + η D_m)(\hat{A} + η \hat{B}) = \hat{A}^{m+1} + η (\hat{A}^m \hat{B} + D_m \hat{A})``. Only ``η^2 = 0`` and the fact that ``η`` commutes with everything enter; the order of ``\hat{A}`` and ``\hat{B}`` is preserved. Expanding ``(\hat{A} + s \hat{B})^m`` into ordered products, the terms of first order in ``s`` are exactly those of ``s D_m``. As ``\Vert D_m \Vert \le m \Vert \hat{A} \Vert^{m-1} \Vert \hat{B} \Vert``, all series converge absolutely for small ``|s|``, and may be rearranged into ``f(\hat{A} + s \hat{B}) = f(\hat{A}) + s \sum_m c_m D_m + \mathcal{O}(s^2)``. ``\square``

For the exponential, ``R = \infty``. With ``f(z) = e^{-\ii z dt}``, ``\hat{A} = \hat{H}``, and ``\hat{B} = \hat{μ}``, the dual part in Eq. \eqref{eq:analytic} is the Taylor series \eqref{eq:taylor} for ``t = dt``, and it is the derivative of ``e^{-\ii \hat{H}(ϵ) dt}`` with respect to ``ϵ`` by the chain rule. Since ``ι`` preserves products and convergent series ([Proposition 3](@ref Background-prop-iota)), and ``G = ι(\hat{H}(ϵ + η))`` ([Proposition 4](@ref Background-prop-gradgen-dual)),

```math
\begin{equation}\label{eq:expG-dual}
e^{-\ii G dt} = ι\Big(e^{-\ii \hat{H}(ϵ + η) dt}\Big) = ι\big(\hat{U} + η \partial_ϵ \hat{U}\big) = \begin{pmatrix} \hat{U} & \partial_ϵ \hat{U} \\ 0 & \hat{U} \end{pmatrix}\,,
\end{equation}
```

which is an independent proof of the block form of ``e^{-\ii G dt}`` in Eq. \eqref{eq:gradgen-exp}. Applied to a gradient vector, ``(\hat{U} + η \partial_ϵ \hat{U})(|Ψ⟩ + η |Ψ^\prime⟩)`` is the product rule for a non-zero gradient state, Eq. \eqref{eq:chain}, and over several time steps, the dual part of the product of the dual propagators ``\hat{U}_n + η \partial_ϵ \hat{U}_n`` is Eq. \eqref{eq:multistep-sum}.

### [Forward-mode differentiation of numerical methods](@id Background-Numerical-Methods)

The dual-number picture extends beyond the exact exponential, to the numerical methods that are actually used to propagate gradient vectors.

!!! proposition "Proposition 6"

    [](@id Background-prop-linear-algebra)Consider a finite computation that produces a vector or an operator from one or more generators ``\hat{H}(t_i; ϵ)``, a state ``|Ψ(ϵ)⟩``, and any number of ``ϵ``-independent operators and vectors, using only linear combinations with ``ϵ``-independent coefficients, products of operators, application of operators to vectors, and the solution of linear systems. Run the same computation with each ``\hat{H}(t_i)`` replaced by ``G[\hat{H}(t_i)]``, ``|Ψ⟩`` by ``(\partial_ϵ |Ψ⟩, |Ψ⟩)``, every ``ϵ``-independent operator ``\hat{O}`` by ``\mathrm{diag}(\hat{O}, \hat{O})``, and every ``ϵ``-independent vector ``|Φ⟩`` by ``(0, |Φ⟩)``. Then, in exact arithmetic, a resulting vector ``|Φ(ϵ)⟩`` is replaced by ``(\partial_ϵ |Φ⟩, |Φ⟩)``, and a resulting operator ``\hat{X}(ϵ)`` by ``\left(\begin{smallmatrix} \hat{X} & \partial_ϵ \hat{X} \\ 0 & \hat{X} \end{smallmatrix}\right)``.

*Proof.* By the identification of gradient generators and gradient vectors with dual objects ([Proposition 4](@ref Background-prop-gradgen-dual)), the replacements are ``ι`` of the inputs evaluated at ``ϵ + η``, each of the form "value ``+ η`` derivative". Since ``ι`` preserves sums, products, and inverses ([Proposition 3](@ref Background-prop-iota)), running the computation on the block objects is the same as running it in dual arithmetic. Each allowed operation preserves the form "value ``+ η`` derivative": linear combinations because ``\partial_ϵ`` is linear, products because Eq. \eqref{eq:dual-op-mult} is the product rule, and inverses because the dual part of ``X^{-1} = \hat{A}^{-1} - η \hat{A}^{-1} \hat{B} \hat{A}^{-1}`` is ``\partial_ϵ (X^{-1})``. ``\square``

[Proposition 6](@ref Background-prop-linear-algebra) allows only linear combinations with ``ϵ``-independent coefficients, products of operators, applications of operators to vectors, and solutions of linear systems. Two kinds of operations fall outside of it. First, operations that are not complex-linear in the state, such as inner products and norms (e.g., in an orthogonalization or an error estimate), act on the gradient vector as a whole, not as dual arithmetic would. Second, choices that a method makes based on the data, such as the number of terms of an expansion, a spectral range, or an adaptive time step, are made for ``G`` rather than for ``\hat{H}``. If such choices are fixed before the computation, the remaining computation is linear algebra of the allowed kind, and its result is the exact derivative for these fixed choices. The [flat, frozen, and dual semantics](@ref Background-Semantics) below distinguish how a method can treat inner products, norms, and data-dependent choices.

Most propagation methods for a piecewise-constant generator approximate ``\hat{U} = e^{-\ii \hat{H} dt}`` by a polynomial in ``\hat{H}``. Evaluating a polynomial requires only linear combinations and products of ``\hat{H}``, which are operations allowed in [Proposition 6](@ref Background-prop-linear-algebra). This gives a strong statement.

!!! corollary "Corollary 7"

    [](@id Background-cor-polynomial)Let ``p(z) = \sum_{k=0}^{K} c_k z^k`` be a polynomial whose coefficients do not depend on ``ϵ``. Then ``p(G)`` maps ``(\partial_ϵ |Ψ⟩, |Ψ⟩)`` to ``(\partial_ϵ [p(\hat{H}(ϵ)) |Ψ⟩], p(\hat{H}(ϵ)) |Ψ⟩)``.

The gradient state is thus the exact derivative of the *approximate* propagated state, not an approximation of the derivative of the exact propagated state. The error of the propagation enters the state and the gradient consistently: up to floating-point rounding, the gradient is the derivative of the function of ``ϵ`` that the propagation actually evaluates, no matter how few terms the polynomial has. How well it approximates ``\partial_ϵ \hat{U} |Ψ⟩`` depends on how well ``p^\prime`` approximates the derivative of ``z \mapsto e^{-\ii z dt}`` on the spectrum of ``\hat{H}``.

The [Chebychev propagator](@extref QuantumPropagators :label:`method_cheby`) in `QuantumPropagators` [Tal-EzerJCP1984](@cite) is of this form. It shifts and scales ``\hat{H}`` with a spectral range ``[E_{\min}, E_{\min} + Δ]``, sums Chebychev polynomials with Bessel-function coefficients that depend only on ``Δ dt``, and multiplies with a phase factor. The spectral range is determined when the propagator is initialized, from the minimum and maximum values of the controls, and it does not change with the control values in a particular time step. Thus, the polynomial of each time step has ``ϵ``-independent coefficients, and the gradient state after each time step is the exact derivative of the Chebychev approximation ([Corollary 7](@ref Background-cor-polynomial)). For a gradient generator, the spectral range of ``G`` is that of ``\hat{H}`` ([Proposition 8](@ref Background-prop-charpoly)), and `specrange` computes it from ``\hat{H}``, so the same Chebychev coefficients apply.

This exactness does not hold for methods whose polynomial depends on the propagated vector. The [Newton propagator](@extref QuantumPropagators :label:`method_newton`) [BermanJPA1992, AshkenaziJCP1995](@cite) builds a Krylov subspace with an Arnoldi iteration [ArnoldiQAM1951](@cite), which uses inner products of the full gradient vector, and chooses its interpolation points from the resulting Ritz values. The propagation of ``Ψ`` under ``G`` is then not the derivative of the Newton propagation of ``|Ψ⟩`` under ``\hat{H}``. Instead, it is an approximation of ``e^{-\ii G dt} Ψ`` that converges together with the propagation, so the gradient states are accurate to the precision of the propagation. The same holds for Krylov methods in general.

### [Flat, frozen, and dual semantics](@id Background-Semantics)

Operations beyond linear combinations of vectors and operators with ``ϵ``-independent coefficients, products of operators, applications of operators to vectors, and solutions of linear systems ([Proposition 6](@ref Background-prop-linear-algebra)), such as inner products, norms, and the choices that a method makes based on the data (a spectral range, interpolation points, the number of terms, or a time step), determine the interpretation and role of the gradient states in a numerical propagation. A propagation method that is applied to a gradient vector can treat these operations in three ways.

In the *flat* semantics, ``G`` and ``Ψ`` are an ordinary matrix and vector of dimension ``N(L+1)``. Inner products and norms include the gradient states, and every choice that depends on the data is made for ``G`` and ``Ψ`` as a whole. The state component of the result then differs from the propagation of ``|Ψ⟩`` under ``\hat{H}`` alone within the precision of the propagation, and the gradient states approximate the exact derivative ``\partial_ϵ |Ψ(T)⟩`` of the propagated state to that same precision. The gradient converges together with the propagation.

In the *frozen* semantics, every quantity that a method derives from the data, such as a spectral range, interpolation points, expansion coefficients, the number of terms, or a time step, is computed from ``\hat{H}`` and the state ``|Ψ⟩`` alone, exactly as in the propagation of ``|Ψ⟩`` without gradients, and then held fixed. The state component of the result is then the same as the propagation of ``|Ψ⟩`` under ``\hat{H}`` (up to floating-point rounding). With the frozen quantities, the remaining computation consists only of linear combinations, products, and linear solves with ``ϵ``-independent coefficients, so by [Proposition 6](@ref Background-prop-linear-algebra), the gradient states are the exact derivative of the computed state, except for the dependence of the frozen quantities on ``ϵ``. The computed state depends on these quantities only through the error of the propagation, so the missing contribution is of the order of that error. For ODE solvers, this approach is known as internal numerical differentiation [BockMOCRS1981; Sec. 8.3.3](@cite).

In the *full dual* semantics, the quantities that a method derives from the data are themselves dual numbers, computed in dual arithmetic. For example, the inner product of two dual vectors is ``⟨Ψ|Φ⟩ + η (⟨Ψ^\prime|Φ⟩ + ⟨Ψ|Φ^\prime⟩)``. Thus, their dependence on ``ϵ`` is differentiated as well, and only discrete decisions, such as the number of terms or the acceptance of a time step, remain frozen. The gradient states are then the exact derivative of the computed state, wherever the discrete decisions do not change. This is forward-mode automatic differentiation with a dual-number type like that of [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl). Gradient vectors cannot represent it, since an inner product of gradient vectors is an ordinary complex number; it requires dual numbers as the scalar type throughout the propagation method. Whether the error control of an adaptive method is itself frozen is a choice of the method. In [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl), the default error norm of a *complex* state of ForwardDiff dual numbers is a dual number whose value is the norm of the values of the state, and the solver chooses its time steps from that value, as in the frozen semantics. For a *real* state of dual numbers, the default error norm also includes the dual parts, as in the flat semantics.

The three semantics coincide for methods that make no choices based on the data, or whose choices do not depend on ``ϵ``: the exact exponential ([Proposition 1](@ref Background-prop-gradgen-exp)), polynomial propagators with fixed coefficients ([Corollary 7](@ref Background-cor-polynomial)), and ODE solvers with a fixed time grid ([Proposition 12](@ref Background-prop-ode-solvers)). They differ for Krylov methods and for adaptive time steps. The propagation methods in `QuantumPropagators` behave as follows when propagating a gradient vector:

| Method                                                              | Choices based on the data                                                         | Semantics                 |
|:------------------------------------------------------------------- |:--------------------------------------------------------------------------------- |:------------------------- |
| `ExpProp`                                                           | none                                                                              | exact (all coincide)      |
| [Chebychev](@extref QuantumPropagators :label:`method_cheby`)       | spectral range of ``\hat{H}`` ([Proposition 8](@ref Background-prop-charpoly)), fixed at initialization            | all coincide              |
| [Newton](@extref QuantumPropagators :label:`method_newton`)         | Krylov subspace and interpolation points, from the full gradient vector           | flat                      |
| ODE solvers with adaptive time steps                                | time steps, from an error norm of the state ([`frozen_internalnorm`](@ref), the default for a [`GradVector`](@ref)) or of the full gradient vector ([`flat_internalnorm`](@ref)), see the [how-to guide](@ref Howto-ODE) | frozen (default) or flat |


### Relation to automatic differentiation

A gradient vector ``(|Ψ^\prime⟩, |Ψ⟩)`` is the faithful matrix representation of the dual state ``|Ψ⟩ + η |Ψ^\prime⟩``, and the gradient generator is the representation of the dynamical generator at the dual control value ``ϵ + η`` (Propositions [3](@ref Background-prop-iota) and [4](@ref Background-prop-gradgen-dual)). Propagating a gradient vector under a gradient generator is forward-mode automatic differentiation of the propagation with respect to ``ϵ``. For the exact propagator, this gives ``(\partial_ϵ (\hat{U} |Ψ⟩), \hat{U} |Ψ⟩)``. For a numerical method built from linear algebra with ``ϵ``-independent coefficients, such as a Chebychev propagator, it gives the exact derivative of the approximate result. For methods whose internal choices depend on the propagated vector, such as Krylov methods, a gradient vector follows the flat semantics, and the gradient is an approximation of the exact derivative that converges together with the propagation.

GRAPE combines this forward mode with the reverse mode. The gradient with respect to all ``ϵ_{nl}`` is obtained from one forward propagation of ``|Ψ⟩`` and one backward propagation of ``|χ⟩``, independent of the number ``N_T`` of time steps. That backward propagation is reverse-mode (adjoint) differentiation across the time steps. Within each time step, the gradient generator ``G[\hat{H}_n^\dagger]`` provides the derivative of the single-step propagator in forward mode, for all ``L`` controls at once, with the gradient states reset before each step [GoerzQ2022](@cite).


## [Properties of the gradient generator](@id Background-Properties)

Propagation methods use more than the product of the generator with a vector. They estimate spectral ranges, diagonalize, compute inner products and norms, and may rely on Hermiticity or use the adjoint of the generator. Each subsection below starts with a practical consequence for the propagation of gradient vectors and then derives it from a property of the gradient generator ``G = G[\hat{H}]`` for ``L`` controls, Eq. \eqref{eq:gradgen}. Throughout, ``G`` is block upper-triangular, with ``\hat{H}`` in all ``L+1`` diagonal blocks.

### [Spectrum](@id Background-Spectrum)

The spectral range of a gradient generator, which methods like the Chebychev propagator require, is the spectral range of ``\hat{H}`` and does not have to be estimated separately. Accordingly, `QuantumGradientGenerators` evaluates `eigvals`, `tr`, and `det` of a [`GradgenOperator`](@ref), as well as its spectral range ([`specrange`](@extref QuantumPropagators.SpectralRange.specrange)), from ``\hat{H}`` alone. For an exact diagonalization, this reduces the dimension from ``N(L+1)`` to ``N``. For an Arnoldi estimate, it avoids the Ritz values of ``G``, which can lie outside of the spectral range, as shown [below](@ref Background-Non-Normality).

!!! proposition "Proposition 8"

    [](@id Background-prop-charpoly)The characteristic polynomial of the gradient generator ``G`` for ``L`` controls is

    ```math
    \begin{equation}\label{eq:charpoly}
    \det(G - λ \mathbb{1}) = \det(\hat{H} - λ \mathbb{1})^{L+1}\,.
    \end{equation}
    ```

    Thus, ``G`` has the same eigenvalues as ``\hat{H}``, each with ``L+1`` times its algebraic multiplicity, and ``\operatorname{tr} G = (L+1) \operatorname{tr} \hat{H}`` and ``\det G = (\det \hat{H})^{L+1}``. In particular, ``G`` has the same spectral range and spectral radius as ``\hat{H}``, and its eigenvalues are real if ``\hat{H}`` is Hermitian, even though ``G`` is not Hermitian.

*Proof.* The matrix ``G - λ \mathbb{1}`` is block upper-triangular with diagonal blocks ``\hat{H} - λ \mathbb{1}``, and the determinant of a block-triangular matrix is the product of the determinants of its diagonal blocks. The trace of ``G`` is the sum of the traces of its diagonal blocks, and ``\det G`` is Eq. \eqref{eq:charpoly} at ``λ = 0``. ``\square``

### [Diagonalizability](@id Background-Diagonalization)

A gradient generator cannot be propagated by diagonalizing it once and evaluating ``V e^{-\ii D t} V^{-1}`` for any number of time steps, as is common for a constant generator ``\hat{H} = V D V^{-1}``. The gradient generator ``G`` does not have a complete set of eigenvectors, so no such ``V`` exists. A numerical eigensolver does not report this as an error. Instead, it returns a matrix ``V`` with nearly parallel columns, and ``V e^{-\ii D t} V^{-1}`` is then wrong in all digits [MolerSIREV2003; Method 14](@cite). For example, for random Hermitian ``20 \times 20`` matrices ``\hat{H}`` and ``\hat{μ}``, the relative error of the resulting gradient block is larger than one. A method that relies on diagonalization must therefore diagonalize ``\hat{H}``, not ``G``. Methods that evaluate the exponential of ``G`` without diagonalizing it, such as the Chebychev propagator (a polynomial in ``G``) or `ExpProp` (a Padé approximation), are not affected.

The reason is that the derivative of a phase grows linearly in time. Let ``L = 1`` ([a single control](@ref Background-Reduction)), let ``\hat{H}`` be Hermitian, and let ``\hat{H} |r⟩ = λ |r⟩`` for a non-degenerate eigenvalue ``λ``. Propagating the gradient vector ``(0, |r⟩)`` yields ``(\partial_ϵ \hat{U}(t) |r⟩, e^{-\ii λ t} |r⟩)`` by [Proposition 1](@ref Background-prop-gradgen-exp): the state only acquires a phase. Inserting ``\hat{U}(s) |r⟩ = e^{-\ii λ s} |r⟩`` and ``⟨r| \hat{U}(t - s) = e^{-\ii λ (t - s)} ⟨r|`` into the integral for ``\partial_ϵ \hat{U}(t)``, Eq. \eqref{eq:duhamel}, gives the component of the gradient state along ``|r⟩``,

```math
\begin{equation}\label{eq:phase-derivative}
⟨r| \partial_ϵ \hat{U}(t) |r⟩
= -\ii t ⟨r| \hat{μ} |r⟩ e^{-\ii λ t}
= \frac{\partial}{\partial ϵ} e^{-\ii λ(ϵ) t}\,,
\end{equation}
```

where ``⟨r| \hat{μ} |r⟩ = \partial_ϵ λ`` is the derivative of the eigenvalue (the Hellmann–Feynman theorem [FeynmanPR1939](@cite)). If ``G`` had a complete set of eigenvectors, every component of ``e^{-\ii G t} Ψ`` would be a linear combination of the pure exponentials ``e^{-\ii λ_j t}``, which cannot produce the term ``t e^{-\ii λ t}``. Thus, whenever an eigenvalue of ``\hat{H}`` depends on the control, ``G`` lacks an eigenvector.

!!! proposition "Proposition 9"

    [](@id Background-prop-jordan)Let ``L = 1`` ([a single control](@ref Background-Reduction)), and let ``λ`` be a non-degenerate eigenvalue of ``\hat{H}`` with right and left eigenvectors, ``\hat{H} |r⟩ = λ |r⟩`` and ``⟨\tilde{r}| \hat{H} = λ ⟨\tilde{r}|``. If ``⟨\tilde{r}| \hat{μ} |r⟩ \neq 0``, the only eigenvectors of ``G`` for the eigenvalue ``λ`` are the multiples of ``(|r⟩, 0)``, although ``λ`` is a double eigenvalue of ``G``. Hence, ``G`` is not diagonalizable.

*Proof.* The equation ``(G - λ \mathbb{1}) (|x⟩, |y⟩) = 0`` means ``(\hat{H} - λ \mathbb{1}) |y⟩ = 0`` and ``(\hat{H} - λ \mathbb{1}) |x⟩ = -\hat{μ} |y⟩``. The first equation gives ``|y⟩ = c |r⟩``. Applying ``⟨\tilde{r}|`` to the second gives ``0 = -c ⟨\tilde{r}| \hat{μ} |r⟩``, so ``c = 0``, and ``|x⟩`` is a multiple of ``|r⟩``. By the characteristic polynomial, Eq. \eqref{eq:charpoly}, ``λ`` is a double eigenvalue of ``G``. ``\square``

In the language of linear algebra, ``λ`` belongs to a ``2 \times 2`` Jordan block of ``G``. In the dual-number picture, the eigenvalue of ``\hat{H}(ϵ + η)`` is ``λ + η \partial_ϵ λ``, and its block representation ``ι(λ + η \partial_ϵ λ) = \left(\begin{smallmatrix} λ & \partial_ϵ λ \\ 0 & λ \end{smallmatrix}\right)`` is that Jordan block. Its exponential, ``e^{-\ii λ t} \left(\begin{smallmatrix} 1 & -\ii t \partial_ϵ λ \\ 0 & 1 \end{smallmatrix}\right)``, contains the derivative of the phase in Eq. \eqref{eq:phase-derivative}. The Jordan structure is thus not a defect of the construction, but the way in which ``G`` encodes the derivative. By the [reduction to a single control](@ref Background-Reduction), the ``l``'th gradient state evolves exactly as under the single-control gradient generator ``G_l``, so for ``L`` controls, ``G`` is not diagonalizable as soon as ``⟨\tilde{r}| \hat{μ}_l |r⟩ \neq 0`` for any of the controls.

### [Non-normality](@id Background-Non-Normality)

Methods that assume a Hermitian generator, such as the Lanczos iteration, norm conservation, or bounds on expectation values, do not apply to a gradient generator, even if ``\hat{H}`` is Hermitian.

The norm of a gradient vector is not conserved, since ``e^{-\ii G t}`` is not unitary. By Eq. \eqref{eq:gradgen-exp}, the gradient states grow from zero, and by Eq. \eqref{eq:phase-derivative}, they may grow linearly in time. A propagation method must not renormalize gradient vectors or test them for norm conservation.

Krylov methods must use the Arnoldi iteration [ArnoldiQAM1951](@cite) rather than the Lanczos iteration [LanczosJRNBS1950](@cite), which assumes a Hermitian generator [GolubVanLoan2013; Chapter 10](@cite). Accordingly, `ishermitian` is `false` for a `GradgenOperator`.

Expectation values ``⟨v| G |v⟩`` in normalized vectors ``v`` (also called Rayleigh quotients) can lie outside the spectral range of ``\hat{H}``, which is also the spectral range of ``G``. For example, let ``L = 1`` ([a single control](@ref Background-Reduction)), let ``\hat{H}`` be Hermitian, and let ``|e⟩`` be the eigenvector for its largest eigenvalue ``E_{\max}``. For ``v = (s |e⟩, |e⟩) / \sqrt{2}`` with ``s = \pm 1``,

```math
\begin{equation}\label{eq:rayleigh}
⟨v| G |v⟩ = E_{\max} + \frac{s}{2} ⟨e| \hat{μ} |e⟩\,,
\end{equation}
```

which exceeds ``E_{\max}`` for one choice of ``s`` whenever ``⟨e| \hat{μ} |e⟩ \neq 0``. The Ritz values of an Arnoldi iteration are expectation values of this kind, so an Arnoldi estimate of the spectral range of ``G`` can be too large, which is why `specrange` uses ``\hat{H}`` [instead](@ref Background-Spectrum). Likewise, a check that bounds expectation values by the spectral range is not valid for a gradient generator. For example, the optional `check_normalization` of the Chebychev propagator can fail for a gradient vector, even though the spectral range is correct.

All of these shortcuts rely on the generator being Hermitian, or at least *normal*. A matrix ``A`` is normal if it commutes with its adjoint, ``A A^\dagger = A^\dagger A``. Equivalently, it has a complete set of orthonormal eigenvectors, that is, it can be diagonalized by a unitary transformation. Hermitian and unitary matrices are normal. In the orthonormal eigenbasis of a normal matrix, an expectation value ``⟨v| A |v⟩ = \sum_j |⟨j|v⟩|^2 λ_j`` is a weighted average of the eigenvalues ``λ_j``, and thus lies within the spectral range. For a non-normal matrix, there is no orthonormal eigenbasis, and expectation values can lie outside of it [HornJohnson1991; Chapter 1](@cite). A gradient generator is never normal.

!!! proposition "Proposition 10"

    [](@id Background-prop-non-normal)If ``G`` is normal, ``G G^\dagger = G^\dagger G``, then all ``\hat{μ}_l = 0``. In particular, a gradient generator with a non-zero control derivative is neither Hermitian nor unitary.

*Proof.* For ``l \le L``, the ``l``'th diagonal block of ``G G^\dagger`` is ``\hat{H} \hat{H}^\dagger + \hat{μ}_l \hat{μ}_l^\dagger``, and that of ``G^\dagger G`` is ``\hat{H}^\dagger \hat{H}``. If the two are equal, taking the trace and using ``\operatorname{tr}(\hat{H} \hat{H}^\dagger) = \operatorname{tr}(\hat{H}^\dagger \hat{H})`` gives ``\operatorname{tr}(\hat{μ}_l \hat{μ}_l^\dagger) = 0``, so ``\hat{μ}_l = 0``. ``\square``

### [Adjoint](@id Background-Adjoint)

For a backward propagation, as in GRAPE, the gradient generator must be constructed from the adjoint dynamical generator. This is ``G[\hat{H}^\dagger]``, with ``\hat{μ}^\dagger`` in its upper right block, as discussed for the [exponential of the gradient generator](@ref Background-Exponential). The adjoint of the gradient generator, ``G[\hat{H}]^\dagger``, is a different matrix, and propagating under it does not yield any gradient. It is block lower-triangular. For ``L = 1`` ([a single control](@ref Background-Reduction)),

```math
\begin{equation}\label{eq:gradgen-adjoint}
G[\hat{H}]^\dagger
= \begin{pmatrix} \hat{H}^\dagger & 0 \\ \hat{μ}^\dagger & \hat{H}^\dagger \end{pmatrix}
= P \, G[\hat{H}^\dagger] \, P\,,
\qquad
P = \begin{pmatrix} 0 & \mathbb{1} \\ \mathbb{1} & 0 \end{pmatrix}\,,
\end{equation}
```

so a gradient vector ``(0, |χ⟩)`` propagated under ``G[\hat{H}]^\dagger`` keeps a zero gradient state. The two matrices are similar, but ``G[\hat{H}]^\dagger \neq G[\hat{H}^\dagger]`` unless ``\hat{μ} = 0``. To prevent this mistake, `QuantumGradientGenerators` does not define `adjoint` for a gradient generator. See the [how-to guide on backward propagation](@ref Howto-Backward) for an example.

### [Inner products and norms](@id Background-Norm)

Inner products and norms treat a gradient vector as one long vector of dimension ``N(L+1)``: for two gradient vectors ``Ψ`` and ``Φ``, `dot(Ψ, Φ)` is the inner product of the block vectors, ``⟨Ψ|Φ⟩ + \sum_l ⟨Ψ^\prime_l|Φ^\prime_l⟩`` in terms of their states and gradient states, and `norm(Ψ)` is the corresponding Euclidean norm, ``\big(⟨Ψ|Ψ⟩ + \sum_l ⟨Ψ^\prime_l|Ψ^\prime_l⟩\big)^{1/2}``. These are the inner product and norm of the [flat semantics](@ref Background-Semantics), not the dual-valued inner product of the full dual semantics. That `norm` and `dot` are consistent is essential for the Arnoldi iteration, which normalizes its basis vectors with the norm and orthogonalizes them with the inner product.


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

!!! proposition "Proposition 11"

    [](@id Background-prop-time-dependent)``\hat{U}(t, t_0; u)`` is continuously differentiable in ``u``, and

    ```math
    \begin{equation}\label{eq:gradgen-u-prop}
    \hat{U}_G(t, t_0) = \begin{pmatrix} \hat{U}(t, t_0) & \partial_u \hat{U}(t, t_0) \\ 0 & \hat{U}(t, t_0) \end{pmatrix}\,.
    \end{equation}
    ```

*Proof.* By the standard theorem on the differentiable dependence of solutions of ordinary differential equations on parameters [Hartman2002; Chapter V](@cite) (applied to one interval of the partition after the other), ``\hat{U}`` is continuously differentiable in ``u``, and ``\partial_u \hat{U}`` solves the equation obtained by differentiating Eq. \eqref{eq:tdse-U},

```math
\begin{equation}\label{eq:variational}
\ii \frac{\partial}{\partial t} \partial_u \hat{U}(t, t_0) = \partial_u \hat{H}(t) \hat{U}(t, t_0) + \hat{H}(t) \partial_u \hat{U}(t, t_0)\,,
\qquad \partial_u \hat{U}(t_0, t_0) = 0\,.
\end{equation}
```

The rest of the proof is identical to that of the block form of ``e^{-\ii G t}`` ([Proposition 1](@ref Background-prop-gradgen-exp)): the right-hand side ``W(t)`` of Eq. \eqref{eq:gradgen-u-prop} satisfies ``W(t_0) = \mathbb{1}`` and ``\ii \frac{d}{dt} W = G_u(t) W``, as does ``\hat{U}_G(t, t_0)``. ``\square``

For a generator that is constant in time, Eq. \eqref{eq:gradgen-u-prop} reduces to the block form of ``e^{-\ii G t}``, Eq. \eqref{eq:gradgen-exp}. As for a single time step, Eq. \eqref{eq:chain}, applying Eq. \eqref{eq:gradgen-u-prop} to a gradient vector gives

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

GRAPE is the special case where the control parameters are the pulse values themselves, ``\{u_n\} = \{ϵ_{nl}\}``. Then ``\partial ϵ_l(t) / \partial ϵ_{n^\prime l^\prime}`` is ``δ_{l l^\prime}`` on the interval ``n^\prime`` and zero elsewhere. A forward propagation would carry ``N_T L`` gradient states, each driven only during its own time interval and evolving freely under ``\hat{H}`` afterwards. In the [gradient of a functional](@ref Background-Functionals), the free evolution can be moved onto the backward-propagated state ``|χ⟩``. This is why GRAPE needs only ``L`` gradient states for one time step at a time. For parameters that affect the controls at all times, as in GOAT, the forward propagation carries all ``M`` gradient states over the entire time interval, but requires no backward propagation and no storage of states at intermediate times.

### [Gradients of functionals](@id Background-Functionals)

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

The block form of ``\hat{U}_G``, Eq. \eqref{eq:gradgen-u-prop}, holds for the exact time evolution. In practice, the propagation under ``G_u(t)`` is numerical, either with a general ODE solver or with a piecewise-constant approximation. As in the [reduction to a single control](@ref Background-Reduction), it suffices to consider a single parameter ``u``.

!!! proposition "Proposition 12"

    [](@id Background-prop-ode-solvers)Consider a Runge–Kutta or linear multistep method (explicit or implicit, with exact solution of the linear systems for implicit methods) on a time grid that does not depend on ``u``. Applying the method to ``\ii \frac{d}{dt} Ψ = G_u(t) Ψ`` with the initial gradient vector ``(\partial_u |Ψ(0)⟩, |Ψ(0)⟩)`` yields gradient vectors ``(\partial_u |Ψ_n⟩, |Ψ_n⟩)``, where ``|Ψ_n⟩`` is the numerical solution of Eq. \eqref{eq:tdse} with ``\hat{H}(t; u)`` by the same method.

*Proof.* Each step of the method uses only linear combinations with coefficients that depend on the method and the time steps, applications of the generator at fixed times, and, for implicit methods, solutions of linear systems with matrices assembled linearly from ``\mathbb{1}`` and the generator at fixed times. These are the operations for which forward-mode differentiation through gradient generators is exact ([Proposition 6](@ref Background-prop-linear-algebra)). ``\square``

Thus, differentiating the discretized propagation and discretizing the equation of motion for the derivative \eqref{eq:variational} give the same result, and the gradient is consistent with the computed state to machine precision. This requires that the time steps do not depend on ``u``. With an adaptive step size control, the time steps depend on ``u`` through the error estimate, and the gradient states are the exact derivative of the computed state only for time steps held fixed. If the time steps are chosen from the state alone, as in the [frozen semantics](@ref Background-Semantics), the gradient states are the solution of the equation of motion for the derivative, Eq. \eqref{eq:variational}, by the same method on the time grid of the state, and they converge to ``\partial_u |Ψ(T)⟩`` as the tolerance decreases and the time grid is refined, but the tolerance does not control their error directly. If the error control includes the gradient states, as in the [flat semantics](@ref Background-Semantics), it does.

The second approach is the piecewise-constant approximation that [`QuantumPropagators`](@extref QuantumPropagators :doc:`index`) uses for time-dependent generators. On each interval of the time grid, the generator is evaluated at a fixed time ``\tilde{t}_n``, by default the midpoint of the interval, giving ``\hat{H}_n(u) = \hat{H}(\tilde{t}_n; u)``, the propagators ``\hat{U}_n = e^{-\ii \hat{H}_n dt_n}``, and the approximate final state ``|\bar{Ψ}(T)⟩ = \hat{U}_{N_T} \cdots \hat{U}_1 |Ψ(0)⟩``.

!!! proposition "Proposition 13"

    [](@id Background-prop-pwc)Let ``G_n = G_u(\tilde{t}_n)`` be the gradient generator evaluated at the same times ``\tilde{t}_n`` as the generator. Then

    ```math
    \begin{equation}\label{eq:pwc}
    e^{-\ii G_{N_T} dt_{N_T}} \cdots e^{-\ii G_1 dt_1}
    \begin{pmatrix} \partial_u |Ψ(0)⟩ \\ |Ψ(0)⟩ \end{pmatrix}
    = \begin{pmatrix} \partial_u |\bar{Ψ}(T)⟩ \\ |\bar{Ψ}(T)⟩ \end{pmatrix}\,,
    \end{equation}
    ```

    and as ``\max_n dt_n \rightarrow 0``, the right-hand side converges to ``(\partial_u |Ψ(T)⟩, |Ψ(T)⟩)``.

*Proof.* Since ``\tilde{t}_n`` does not depend on ``u``, ``G_n = G[\hat{H}_n]``, and Eq. \eqref{eq:pwc} is the product rule over multiple time steps, Eq. \eqref{eq:multistep}. The left-hand side is the time evolution under the piecewise-constant ``\bar{G}(t) = G_n`` for ``t \in (t_{n-1}, t_n]``. By variation of constants, ``\hat{U}_{\bar{G}}(T, 0) - \hat{U}_G(T, 0) = -\ii \int_0^T \hat{U}_{\bar{G}}(T, t) (\bar{G}(t) - G_u(t)) \hat{U}_G(t, 0) dt``, which vanishes as ``\max_n dt_n \rightarrow 0`` because ``G_u(t)`` is uniformly continuous on each interval of the partition. The convergence then follows from the block form of ``\hat{U}_G(T, 0)``, Eq. \eqref{eq:gradgen-u-prop}. ``\square``

The gradient states are thus the exact derivative of the piecewise-constant approximation, and they converge to the exact derivative together with it (to second order in ``dt`` for smooth controls and midpoint evaluation). Each step ``e^{-\ii G_n dt_n}`` is evaluated with one of the propagators in `QuantumPropagators`, so Eq. \eqref{eq:pwc} holds to the precision of that propagator, and exactly for a polynomial propagator with ``u``-independent coefficients, whose gradient states are the exact derivative of its approximation ([Corollary 7](@ref Background-cor-polynomial)). It is essential that ``\partial_u \hat{H}`` is evaluated at the same times ``\tilde{t}_n`` as ``\hat{H}``. Otherwise, the gradient states still converge to the exact derivative, but they are not the derivative of the computed state ``|\bar{Ψ}(T)⟩``, and the mismatch between the computed functional and its gradient, of the order of the time discretization error, can stall a gradient-based optimization close to convergence.

### Remarks

Nothing in this section relies on Hermiticity of ``\hat{H}``. All results apply to non-Hermitian Hamiltonians and to Liouvillians, where ``u`` may also parameterize the dissipators, and to a backward propagation under ``\hat{H}^\dagger(t)`` with the gradient generator ``G[\hat{H}^\dagger]``.

Second derivatives can be obtained by nesting: if ``\hat{H}`` is twice continuously differentiable, applying the time-dependent block form, Eq. \eqref{eq:gradgen-u-prop}, to ``G_{u_1}(t)`` with respect to a second parameter ``u_2`` gives a generator of dimension ``4N`` whose time evolution operator contains ``\partial_{u_1} \partial_{u_2} \hat{U}`` in its upper right block. This is the matrix representation of numbers with two dual units, ``η_1^2 = η_2^2 = 0``. [GoodwinJCP2015](@citet) obtain second derivatives of single-step propagators from a larger block-triangular matrix in the same spirit, and [MachnesPRL2018](@citet) note that GOAT extends to second derivatives.
