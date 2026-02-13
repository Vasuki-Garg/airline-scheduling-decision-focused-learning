import time
import numpy as np
import torch
import pyepo
from scipy.optimize import OptimizeResult

def get_flat_params(model):
    with torch.no_grad():
        vec = torch.nn.utils.parameters_to_vector(model.parameters()).cpu().numpy()
    return vec

def set_flat_params(model, theta_flat):
    with torch.no_grad():
        t = torch.tensor(theta_flat, dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(t, model.parameters())

def regret_objective(theta_flat, model, optmodel, loader):
    model.cpu()
    set_flat_params(model, theta_flat)
    model.eval()
    with torch.no_grad():
        r = pyepo.metric.regret(model, optmodel, loader)
    return float(r)

def run_pso(model, optmodel, loader_train,
            num_particles=10, max_iters=30,
            w=0.7, c1=1.5, c2=1.5, verbose=True):

    theta0 = get_flat_params(model)
    dim = theta0.shape[0]

    positions = np.array([theta0 + 0.1 * np.random.randn(dim) for _ in range(num_particles)])
    velocities = 0.1 * np.random.randn(num_particles, dim)

    pbest_pos = positions.copy()
    pbest_val = np.full(num_particles, np.inf, dtype=float)

    gbest_pos = None
    gbest_val = np.inf
    gbest_history = []

    start_time = time.time()

    for it in range(max_iters):
        for i in range(num_particles):
            regret = regret_objective(positions[i], model, optmodel, loader_train)

            if regret < pbest_val[i]:
                pbest_val[i] = regret
                pbest_pos[i] = positions[i].copy()

            if regret < gbest_val:
                gbest_val = regret
                gbest_pos = positions[i].copy()

        gbest_history.append(gbest_val)

        for i in range(num_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_pos[i] - positions[i])
                + c2 * r2 * (gbest_pos - positions[i])
            )
            positions[i] += velocities[i]

        if verbose:
            print(f"PSO iter {it+1}/{max_iters}, best train regret: {gbest_val:.4f}")

    elapsed = time.time() - start_time
    set_flat_params(model, gbest_pos)

    res = OptimizeResult()
    res.x = gbest_pos
    res.fun = gbest_val
    res.nit = max_iters
    res.history = gbest_history
    res.elapsed = elapsed
    return res
