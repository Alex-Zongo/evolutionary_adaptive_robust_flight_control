import os
from dataclasses import dataclass
from pprint import pprint
import numpy as np
from typing import List
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


@dataclass
class Episode:
    fitness: np.float64
    smoothness: np.float64
    length: np.float64
    state_history: List
    ref_signals: List
    actions: List
    reward_lst: List

    def get_history(self) -> np.ndarray:
        """
        Returns the time traced state history of the episode.
        """
        time_trace_hist = np.linspace(0, self.length, len(self.state_history))
        ref_values = np.array([[ref(t_i) for t_i in time_trace_hist]
                              for ref in self.ref_signals]).transpose()
        reward_lst = np.asarray(self.reward_lst).reshape(
            (len(self.state_history), 1))

        return np.concatenate((ref_values, self.actions, self.state_history, reward_lst), axis=1)


def calc_nMAE(error: np.array) -> float:
    """Calculates the normalized Mean absolute error using the error time history
    Args:
        error (np.array): error time history
    Returns:
        float: normalized Mean absolute error in percentage
    """
    mae = np.mean(np.abs(error), axis=0)
    theta_range = np.deg2rad(20)
    phi_range = np.deg2rad(20)
    beta_range = max(np.abs(np.average(error[:, -1])), 3.14159/100)
    signal_range = np.array([theta_range, phi_range, beta_range])

    nmae = mae/signal_range
    return np.mean(nmae) * 100  # normalized (in percentage)


def calc_nMAE_from_ref(ref: np.ndarray, x_crtl: np.ndarray):
    """Calculates the normalized Mean absolute error using the error time history
    Args:
        ref (np.array): ref signal
        x_ctrl (np.array): controlled state signal
    Returns:
        float: normalized Mean absolute error in percentage
    """
    error = ref - x_crtl
    mae = np.mean(np.abs(error), axis=0)
    signal_range = np.deg2rad(np.array([20, 20, 1]))
    nmae = mae/signal_range
    print(np.average(error, axis=0))
    print(f"Theta, phi, beta: {nmae*100} -> total: {100*np.mean(nmae)}\n")
    return np.mean(nmae) * 100  # normalized (in percentage)


def calc_smoothness(y: np.ndarray, dt: float = 0.01, **kwargs) -> float:
    """Calculates the smoothness of a signal
    Args:
        y (np.ndarray): signal
        dt (float, optional): time step. Defaults to 0.01.
    Returns:
        float: smoothness of the signal
    """
    N = y.shape[0]
    A = y.shape[1]
    T = N * dt
    freq = np.linspace(dt, 1/(2*dt), N//2-1)
    Syy = np.zeros((N // 2-1, A))
    for i in range(A):
        Y = fft(y[:, i], N)
        Syy_disc = Y[1:N//2] * np.conjugate(Y[1:N//2])  # discrete
        Syy[:, i] = np.abs(Syy_disc) * dt   # continuous

    # Smoothnes of each signal
    signal_roughness = np.einsum('ij,i -> j', Syy, freq) * 2/N
    _S = np.sum(signal_roughness, axis=-1)
    roughness = np.sqrt(_S) * 100 * (80/T)

    if kwargs.get('plot_spectra'):
        plt.figure(num='spectra')
        plt.title(
            f'Spectra actuating Signals\n Smoothness: {-roughness:0.0f}')
        plt.loglog(freq, Syy[:, 0] * 2/N, linestyle='-', label=r'$\delta_e$')
        plt.loglog(freq, Syy[:, 1] * 2/N, linestyle='--', label=r'$\delta_a$')
        plt.loglog(freq, Syy[:, 2] * 2/N, linestyle=':', label=r'$\delta_r$')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.legend(loc='best')

    return -roughness


def load_config(model_path: str, verbose: bool = False) -> dict:
    """ Load controller confiuaration from file.

    Args:
        model_path (str): Abosolute path to logging folder.

    Returns:
        dict: Configuration dictionary.
    """
    model_path = model_path / Path('files/')
    config_path = os.path.join(model_path, 'config.yaml')
    conf_raw = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))

    conf_dict = {}
    for k, _v in conf_raw.items():
        if isinstance(_v, dict):
            value = _v['value']
        else:
            value = _v
        conf_dict[k] = value

    if verbose:
        pprint(conf_dict)
    return conf_dict


if __name__ == '__main__':
    N = 2000
    dt = 0.01
    x = np.linspace(0.0, N*dt, N, endpoint=False)
    y = np.zeros_like(x)

    clipped_noise = np.clip(0.3 * np.random.randn(y.shape[0]), - 0.5, 0.5)
    _action = np.clip(x + clipped_noise, -1.0, 1.0)
    action = np.array([_action, _action, _action]).reshape(-1, 3)

    smoothness = calc_smoothness(action, plot_spectra=True)

    print(smoothness * N*dt)

    plt.show()
