"""Training-only SVD diagnostics; does not modify the selected model or test."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from sklearn.preprocessing import StandardScaler


def gavish_donoho_threshold(singular_values, shape):
    """Unknown-noise hard threshold, using the Marchenko-Pastur median.

    This is a reconstruction criterion under low-rank plus white-noise
    assumptions, not an optimum for classification of these radiographs.
    """
    beta = min(shape) / max(shape)
    lower, upper = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2

    def density(t):
        return np.sqrt(max(0, (upper - t) * (t - lower))) / (2 * np.pi * beta * t)

    median = brentq(lambda value: quad(density, lower, value)[0] - 0.5,
                    lower + 1e-10, upper - 1e-10)
    coefficient = np.sqrt(2 * (beta + 1) + 8 * beta /
                          (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1)))
    omega = coefficient / np.sqrt(median)
    threshold = omega * np.median(singular_values)
    return dict(beta=float(beta), omega=float(omega), threshold=float(threshold),
                rank=int(np.count_nonzero(singular_values > threshold)))


def fit_svd_diagnostics(X_train, y_train, labels, selected_k):
    """Fit the common scaler and full class spectra using train alone."""
    scaler = StandardScaler().fit(X_train)
    standardized = scaler.transform(X_train).astype(np.float64)
    classes, rows = {}, []
    for target, label in enumerate(labels):
        values = standardized[np.asarray(y_train) == target]
        mean = values.mean(axis=0)
        centered = (values - mean).T
        basis, singular, _ = np.linalg.svd(centered, full_matrices=False)
        linear = np.cumsum(singular) / singular.sum()
        variance = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        gd = gavish_donoho_threshold(singular, centered.shape)
        classes[target] = dict(mean=mean, basis=basis, singular=singular,
                               linear=linear, variance=variance, gd=gd)
        rows.append(dict(target=target, label=label, train_images=len(values),
                         selected_k=int(selected_k),
                         variance_at_selected_k=float(variance[selected_k - 1]),
                         k_90=int(np.searchsorted(variance, .90) + 1),
                         k_95=int(np.searchsorted(variance, .95) + 1),
                         gd_rank=gd['rank'], gd_threshold=gd['threshold'],
                         gd_beta=gd['beta'], gd_omega=gd['omega']))
    return scaler, classes, pd.DataFrame(rows)


def plot_svd_selection(classes, labels, validation, selected_k):
    """Show spectral diagnostics and the already evaluated validation sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.get_cmap('tab10').colors
    for target, label in enumerate(labels):
        data = classes[target]
        k = np.arange(1, len(data['singular']) + 1)
        axes[0, 0].semilogy(k, np.maximum(data['singular'], 1e-14),
                            label=label, color=colors[target])
        axes[0, 1].plot(k, data['linear'], label=label, color=colors[target])
        axes[1, 0].plot(k, data['variance'], label=label, color=colors[target])
    axes[0, 0].set_title('Valores singulares por grado (train)')
    axes[0, 0].set_ylabel('Magnitud, escala log')
    axes[0, 1].set_title('Suma acumulada normalizada de valores singulares')
    axes[0, 1].set_ylabel('Suma sigma / suma total sigma')
    axes[1, 0].set_title('Varianza acumulada de matrices centradas')
    axes[1, 0].set_ylabel('Suma sigma² / suma total sigma²')
    for ax in [axes[0, 1], axes[1, 0]]:
        ax.axhline(.90, color='gray', linestyle=':', label='90 %')
        ax.axhline(.95, color='black', linestyle='--', label='95 %')
        ax.set_ylim(0, 1.03)
    sweep = validation[validation.family == 'SVD'].copy()
    sweep['k'] = sweep.Modelo.str.extract(r'k=(\d+)').astype(int)
    sweep = sweep.sort_values('k')
    axes[1, 1].plot(sweep.k, sweep['F1 Macro'], 'o-', color='#007c83')
    chosen = sweep.loc[sweep.k == selected_k].iloc[0]
    axes[1, 1].scatter([selected_k], [chosen['F1 Macro']], s=85,
                        color='#c04a21', zorder=5, label=f'k={selected_k} elegido')
    axes[1, 1].set_title('F1 macro de validación frente a k')
    axes[1, 1].set_ylabel('F1 macro de validación')
    axes[1, 1].set_ylim(0, 1)
    for ax in axes.flat:
        ax.axvline(selected_k, color='#437f3c', linestyle=':', alpha=.8)
        ax.set_xlabel('Número de componentes k')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('SVD de rodilla: cinco subespacios y elección con validación', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig


def plot_gavish_donoho(classes, labels, selected_k):
    """Display theoretical truncation separately for each class aspect ratio."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for target, label in enumerate(labels):
        ax = axes.flat[target]
        data, gd = classes[target], classes[target]['gd']
        k = np.arange(1, len(data['singular']) + 1)
        ax.semilogy(k, np.maximum(data['singular'], 1e-14), color='#176b85')
        ax.axhline(gd['threshold'], color='#c04a21', linestyle='--',
                   label=f"Umbral = {gd['threshold']:.2f}")
        ax.axvline(gd['rank'] + .5, color='#c04a21', linestyle=':',
                   label=f"Rango teórico = {gd['rank']}")
        ax.axvline(selected_k, color='#437f3c', linestyle=':',
                   label=f'k validación = {selected_k}')
        ax.set(title=f"{label} · beta={gd['beta']:.3f}",
               xlabel='Índice singular (desde 1)', ylabel='Valor singular, log')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes.flat[5].axis('off')
    axes.flat[5].text(0, .85,
        'Análisis complementario\n\n'
        'Gavish-Donoho busca reconstrucción\ncon ruido blanco y estructura de bajo rango.\n\n'
        'Estas hipótesis no están verificadas\nen las radiografías estandarizadas.\n\n'
        'No reemplaza la selección con validación\nni modifica el SVM final.',
        transform=axes.flat[5].transAxes, fontsize=11, va='top', linespacing=1.5)
    fig.suptitle('Truncamiento Gavish-Donoho: un umbral por cada grado', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig


def plot_svd_reconstructions(scaler, classes, X_validation, y_validation,
                             labels, selected_k, image_shape=(36, 64)):
    """Use one fixed validation example per class for geometric illustration.

    Bases/means/scaler are learned on train only. Inverse scaling restores
    original pixel units for mean and reconstruction. Signed modes remain
    in standardized coordinates and are not anatomical structures.
    """
    fig, axes = plt.subplots(len(labels), 5, figsize=(15, 10))
    examples = []
    for target, label in enumerate(labels):
        index = int(np.flatnonzero(np.asarray(y_validation) == target)[0])
        raw = np.asarray(X_validation[index], dtype=np.float64)
        z = scaler.transform(raw[None])[0]
        data = classes[target]
        basis = data['basis'][:, :selected_k]
        delta = z - data['mean']
        reconstructed_z = data['mean'] + (delta @ basis) @ basis.T
        reconstructed = scaler.inverse_transform(reconstructed_z[None])[0]
        mean = scaler.inverse_transform(data['mean'][None])[0]
        residual = np.abs(raw - reconstructed)
        mode = data['basis'][:, 0]
        bound = max(float(np.abs(mode).max()), 1e-12)
        items = [mean, mode, raw, reconstructed, residual]
        names = ['Media de train', 'Modo 1 (signo arbitrario)',
                 'Entrada de validación', f'Reconstrucción k={selected_k}',
                 'Error absoluto (sin recortar)']
        for column, (values, name) in enumerate(zip(items, names)):
            ax = axes[target, column]
            if column == 1:
                ax.imshow(values.reshape(image_shape), cmap='coolwarm',
                          vmin=-bound, vmax=bound)
            elif column == 4:
                ax.imshow(values.reshape(image_shape), cmap='magma', vmin=0, vmax=1)
            else:
                ax.imshow(values.reshape(image_shape), cmap='gray', vmin=0, vmax=1)
            if target == 0:
                ax.set_title(name, fontsize=9, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[target, 0].set_ylabel(label, fontsize=11)
        examples.append(dict(target=target, label=label, validation_index=index,
                             reconstruction_rmse=float(np.sqrt(np.mean(residual ** 2)))))
    fig.suptitle('SVD: medias, modos y reconstrucción de píxeles 64 × 36\n'
                 'Un ejemplo fijo por grado; escala gris [0,1], error [0,1]; modos ±máximo por fila',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=1.4)
    return fig, examples
