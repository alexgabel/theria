# MAML Gradient Structure

This note derives the outer-loop gradient of MAML and identifies the exact
location where Hessian–vector products enter. The goal is not to re-derive
MAML exhaustively, but to isolate the minimal differentiation primitives
required by theria.