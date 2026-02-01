# MAML Gradient Structure

This note derives the outer-loop gradient of MAML and identifies the exact
location where Hessian–vector products enter. The goal is not to re-derive
MAML exhaustively, but to isolate the minimal differentiation primitives
required by theria.

## FO-MAML semantics in theria

The `--fo` flag in Phase 10 runs implements the standard “first-order MAML”
approximation: the inner-loop gradients are **not** part of the computation
graph. In code (`theria/maml/loops.py::inner_adapt`):

```python
grads = torch.autograd.grad(
    loss_s,
    tuple(phi.values()),
    create_graph=not fo,   # fo=True → create_graph=False
    retain_graph=True,
    allow_unused=False,
)
phi = OrderedDict((name, p - inner_lr * g) for (name, p), g in zip(phi.items(), grads))
```

Implications:
- `fo=True`: inner gradients are detached; outer backprop treats the updated
  parameters as constants → no second-order terms.
- `fo=False`: inner gradients keep their graph; outer backprop includes the
  full second-order path (HVPs).

This distinction is exactly what the Phase 10 `second_order_path` diagnostic
reports in the experiment scripts.
