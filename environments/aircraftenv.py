from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Union
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import jsbsim as jsb


def printRed(skk): print(f"\033[91m {skk}\033[00m")
def printGreen(skk): print(f"\033[92m {skk}\033[00m")
def printLightPurple(skk): print(f"\033[94m {skk}\033[00m")
def printPurple(skk): print(f"\033[95m {skk}\033[00m")
def printCyan(skk): print(f"\033[96m {skk}\033[00m")
def printYellow(skk): print(f"\033[93m {skk}\033[00m")

class BaseEnv(gym.Env, ABC):
    """
    Base class for all environments. For the purpose of writing generic training, rendering and code that applies to all the Citation environments
    """
    
    @property
    @abstractmethod
    def action_space(self) -> Box:
        """
        Returns the action space of the environment
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation_space(self) -> Box:
        """
        Returns the observation space of the environment
        """
        raise NotImplementedError
    
    @abstractmethod
    def reference_value(self) -> List[float]:
        """
        Returns the reference value of the environment
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_controlled_state(self) -> List[float]:
        """
        Returns the list of controlled states of the environment
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_reward(self) -> float:
        """
        Returns the reward of the environment
        """
        pass
    
    @abstractmethod
    def error_calculation(self) -> np.array:
        """
        Returns the error of the environment
        """
        pass
    
    @abstractmethod
    def scale_action(self, action: np.array) -> np.array:
        """
        Returns the scaled action of the environment from the clipped action [-1,1] to [action_space.low, action_space.high]
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (action + 1.0) * 0.5 * (high - low)
    
    @abstractmethod
    def unscale_action(self, action: np.array) -> np.array:
        """
        Rescale the ction from [action_space.low, action_space.high] to [-1,1]
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * (action - low) / (high - low) - 1
    
    
class AircraftEnv(BaseEnv):
    """ Wrapper Environment """
    
    
    n_actions_full: int = 10
    n_obs_full: int = 12
    t: float = 0.
    dt: np.float16 = 0.01
    mToFeet: float = 3.28084
    
    def __init__(self, configuration: str = None, mode: str = "nominal", render: bool = False, **kwargs):
        
        # full state control
        self.n_actions = 3 # aileron da, elevator de, rudder dr
        self.obs_idx = range(10) # 10 states, all states
        
        if mode == "nominal":
            from .h2000_v90 import citation as citation_h2000_v90
            self.aircraft = citation_h2000_v90
            printGreen('Trim mode: h=2000 m v=90 m/s (nominal)')
        
        if render:
            self.fdm = jsb.FGFDMExec("./JSBSim", None)
            self.fdm.load_model("citation")
            self.fdm.set_dt(dt)
            self.fdm.set_output_directive("data_output/flightgear.xml")
            self.fdm.load_ic("cruise_init", True)
            self.fdm.run_ic()
            self.fdm.do_trim(1)
            self.fdm.print_simulation_configuration()
            
    
        
            
    
    