from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Union
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import jsbsim as jsb
from signals.base_signal import BaseSignal


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
        
        if render:
            self.fdm = jsb.FGFDMExec("./JSBSim", None)
            self.fdm.load_model("citation")
            self.fdm.set_dt(self.dt)
            self.fdm.set_output_directive("data_output/flightgear.xml")
            self.fdm.load_ic("cruise_init", True)
            self.fdm.run_ic()
            self.fdm.do_trim(1)
            self.fdm.print_simulation_configuration()
            
        
        if mode == "nominal":
            from .h2000_v90 import citation as citation_h2000_v90
            self.aircraft = citation_h2000_v90
            printGreen('Trim mode: h=2000 m v=90 m/s (nominal)')
        else:
            raise ValueError("Unknown trim condition or control mode")
        
        self.use_incremental = kwargs['incremental']
        if self.use_incremental:
            print('Incremental Control.')
            
        # Evaluation mode
        self.eval_mode: bool = False
        self.t_max: float = 20 # [s] to be changed
        
        # DASMAT Inputs ---> DASMAT States
        """
        0: de                       0: p
        1: da                       1: q
        2: dr                       2: r
        3: de trim                  3: V
        4: da trim                  4: alpha
        5: dr trim         --->     5: beta
        6: df                       6: phi
        7: gear                     7: theta
        8: throttle1                8: psi
        9: throttle2                9: he
                                   10: xe
                                   11: ye
        """
        self.x: np.ndarray = None # observed state vector
        self.obs: np.ndarray = None
        self.last_obs: np.ndarray = None
        self.V0: float = None # [m/s] 
        self.last_u: np.ndarray = None # last input action or control
        
        # references to track
        self.ref: List[BaseSignal] = None
        self.ref_value: np.ndarray = None
        self.theta_trim: float = 0.22 # standard theta trim in degree
        
        # actuator bounds
        if self.use_incremental:
            self.bound = np.deg2rad(25) # [deg/s]
        else:
            self.bound = np.deg2rad(10) # [deg]
            
        # state bounds
        self.max_theta = np.deg2rad(60.0) # [deg]
        self.max_phi = np.deg2rad(75.0) # [deg]
        
            
        if self.use_incremental:
            # aircraft state + actuator state + control states error (equal size with actuator states)
            self.n_obs: int = len(self.obs_idx) + 2*self.n_actions
        else:
            # aircraft state + control states error
            self.n_obs: int = len(self.obs_idx) + self.n_actions

        # state error initialization
        self.error: np.ndarray = np.zeros((self.n_actions))
        
        