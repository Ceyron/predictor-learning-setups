# Predictor Learning Setups

Assume you have a partial differential equation with boundary conditions turned
into a timestepper. In other words, there is discrete mechanism that advances
your discrete vector of degrees of freedom at time level $[t]$, $u_h^{[t]}$, such
that

$$ u^{[t+1]}_h = \mathcal{P}_h(u_h^{[t]}).$$

We assume this timestepper is consistent with the original PDE, is stable and
converges against hypothetical analytical solutions. It shall always operate on a
fixed time step size, $[t+1] - [t] = \Delta t$ Then, this timestepper could be
used recursively to produce a rollout/trajectory

$$ \left \{ u_h^{[s]}=\mathcal{P}_h^s(u_h^{[0]})\right\}_{s=0}^t,$$

given an initial condition $u_h^{[0]}$.

Now, we want to approximately learn this timestepper by a neural network,

$$ f(u_h^{[t]}; \theta) \approx \mathcal{P}_h(u_h^{[t]}).$$

In other words, we need to find a set of parameters $\theta$ that yields a
neural timestepper with desirable properties, e.g.:

* It is also stable over long rollouts
* It matches the numerical timestepper, or at least has a similar qualitative behavior
* It is fast to evaluate (in comparison to the numerical timestepper) [our desire would be orders of magnitude faster, in order to use it as a cheap surrogate model]
* Its trajectories are consistent with the underlying continuous description

In essence, this is an optimization problem over the parameter space $\Theta$.
We could frame it as a **one-step supervised training** problem using some
distribution of dof vectors of initial conditions

$$
\theta = \arg \min_\theta \mathbb{E}_{u_h^{[0]} \propto \mathcal{U}_h^{[0]}} \left[ l\left(f(u_h^{[0]}; \theta) - \mathcal{P}_h(u_h^{[0]}) \right)\right],
$$

with a suitable loss metric $l$. However, there are many more options to
consider, which are listed in the following table. They differ in the length of their rollouts, their interplay with the (differentiable) ground truth solver and potential manipulations of the reverse pass to achieve desirable gradient properties.

| Primal | Reverse-1 | Reverse-2 | Reverse-3 |
| --- | --- | --- | --- |
| One-step supervised training | full gradient | - | - |
| sup-1-none-true-primal ![](https://ceyron.github.io/predictor-learning-setups/sup-1-none-true-primal.svg) | sup-1-none-true-full_gradient ![](https://ceyron.github.io/predictor-learning-setups/sup-1-none-true-full_gradient.svg)| - | - |
| Two-step supervised training with loss aggregated over all time steps | full gradient | no backpropagation through time over network | - |
| sup-2-none-true-primal ![](https://ceyron.github.io/predictor-learning-setups/sup-2-none-true-primal.svg) | sup-2-none-true-full_gradient ![](https://ceyron.github.io/predictor-learning-setups/sup-2-none-true-full_gradient.svg) | sup-2-none-true-no_bptt ![](https://ceyron.github.io/predictor-learning-setups/sup-2-none-true-no_net_bptt.svg) | - |