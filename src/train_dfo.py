# This code has been adapted from: https://github.com/harrylui1995/ASP_E2EPO , the official GitHub Repo for Lui, G. N., & Demirel, S. (2025). Gradient-based smart predict-then-optimize framework for aircraft arrival scheduling problem. Journal of Open Aviation Science, 2(2). https://doi.org/10.59490/joas.2024.7891

import os, time, json
from datetime import datetime
import torch
import torch.nn as nn
import pyepo
from torch.utils.data import DataLoader
from pyepo.data.dataset import optDataset

from matplotlib import pyplot as plt

def visLearningCurve(loss_log, loss_log_regret, method='spo', n_aircraft=30):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
    ax1.plot(loss_log, lw=2)
    ax1.set_xlabel("Iters"); ax1.set_ylabel("Loss")
    ax1.set_title("Learning Curve on Training Set")

    ax2.plot(loss_log_regret, ls="--", alpha=0.7, lw=2)
    num_epochs = len(loss_log_regret)
    tick_spacing = max(1, num_epochs // 10)
    ax2.set_xticks(range(0, num_epochs, tick_spacing))
    ax2.set_ylim(0, 0.5)
    ax2.set_xlabel("Epochs"); ax2.set_ylabel("Regret")
    ax2.set_title("Learning Curve on Test Set")

    plt.savefig(f'{method}_{n_aircraft}.png', dpi=300)
    plt.show()

def create_experiment_dirs():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'experiments_{timestamp}'
    dirs = {
        'base': base_dir,
        'logs': f'{base_dir}/logs',
        'figures': f'{base_dir}/figures',
        'models': f'{base_dir}/models'
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs

def save_experiment_results(results, dirs):
    filename = f"{results['model_name']}_{results['method_name']}_results.json"
    filepath = os.path.join(dirs['logs'], filename)
    results['loss_log'] = [float(x) for x in results['loss_log']]
    results['loss_log_regret'] = [float(x) for x in results['loss_log_regret']]
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def save_model(model, model_name, method_name, dirs):
    filepath = os.path.join(dirs['models'], f"{model_name}_{method_name}_model.pth")
    torch.save(model.state_dict(), filepath)

def save_learning_curve(loss_log, loss_log_regret, model_name, method_name, dirs, n_aircraft):
    current_dir = os.getcwd()
    os.chdir(dirs['figures'])
    visLearningCurve(loss_log, loss_log_regret, method=f"{model_name}_{method_name}", n_aircraft=n_aircraft)
    os.chdir(current_dir)

def run_experiment(model, method_name, loss_func, loader_train, loader_test, optmodel,
                  experiment_dirs, model_name, num_epochs=20, lr=1e-2):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    loss_log = []
    loss_log_regret = [pyepo.metric.regret(model, optmodel, loader_test)]
    elapsed = 0

    criterion = nn.L1Loss() if method_name in ["dbb", "nid"] else None

    for epoch in range(num_epochs):
        tick = time.time()
        for _, data in enumerate(loader_train):
            x, c, w, z = data
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()

            cp = model(x)

            if method_name == "spo+":
                loss = loss_func(cp, c, w, z)
            elif method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
                loss = loss_func(cp, w)
            elif method_name in ["dbb", "nid"]:
                wp = loss_func(cp)
                zp = (wp * c).sum(1).view(-1, 1)
                loss = criterion(zp, z)
            elif method_name == "ltr":
                loss = loss_func(cp, c)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())

        elapsed += time.time() - tick
        regret = pyepo.metric.regret(model, optmodel, loader_test)
        loss_log_regret.append(regret)

    results = {
        'model_name': model_name,
        'method_name': method_name,
        'loss_log': loss_log,
        'loss_log_regret': loss_log_regret,
        'elapsed_time': float(elapsed),
        'final_regret': float(regret),
        'hyperparameters': {
            'num_epochs': int(num_epochs),
            'learning_rate': float(lr),
            'batch_size': int(loader_train.batch_size)
        }
    }

    save_experiment_results(results, experiment_dirs)
    save_model(model, model_name, method_name, experiment_dirs)
    save_learning_curve(loss_log, loss_log_regret, model_name, method_name, experiment_dirs, optmodel.n_aircraft)
    return results

def run_pipeline(x_train, x_test, c_train, c_test, optmodel, models, methods, batch_size=32):
    dirs = create_experiment_dirs()
    dataset_train = optDataset(optmodel, x_train, c_train)
    dataset_test = optDataset(optmodel, x_test, c_test)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        for m in models.values():
            m.cuda()

    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for method_name, loss_func in methods.items():
            results[model_name][method_name] = run_experiment(
                model=model,
                method_name=method_name,
                loss_func=loss_func,
                loader_train=loader_train,
                loader_test=loader_test,
                optmodel=optmodel,
                experiment_dirs=dirs,
                model_name=model_name
            )
    return results, dirs
