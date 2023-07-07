
import numpy as np
from environments.aircraftenv import AircraftEnv
try:
    from environments.lunarlander import LunarLanderWrapper
except:
    print("LunarLanderWrapper not available")


def select_env(environment_name: str, render_mode: bool = False, realtime: bool = False):
    _name = environment_name.lower()

    if 'lunar' in _name:
        wrapper = LunarLanderWrapper()
        return wrapper.env

    elif 'ph' in _name:
        tokens = _name.split('_')
        phlab_mode = 'nominal'
        if len(tokens) == 3:
            _, phlab_config, phlab_mode = tokens
        else:
            phlab_config = tokens[-1]
            phlab_mode = ""

        return AircraftEnv(configuration=phlab_config, mode=phlab_mode, render_mode=render_mode, realtime=realtime)
    else:
        raise ValueError(f"Unknown environment name: {environment_name}")


if __name__ == '__main__':
    env = select_env('Phlab_fullControl_nominal')
    env.reset()
    print(env.step(np.array([0.1, 0.1, 0.05])))
