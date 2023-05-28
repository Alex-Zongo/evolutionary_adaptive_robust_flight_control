from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Union
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import jsbsim as jsb
from signals.base_signal import BaseSignal, Const
from signals.stochastic_signals import RandomizedCosineStepSequence
import time


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
    def calc_reference_value(self) -> List[float]:
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
    def calc_error(self) -> np.array:
        """
        Returns the error of the environment
        """
        pass

    def scale_action(self, action: np.array) -> np.array:
        """
        Returns the scaled action of the environment from the clipped action [-1,1] to [action_space.low, action_space.high]
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (action + 1.0) * 0.5 * (high - low)

    def unscale_action(self, action: np.array) -> np.array:
        """
        Rescale the action from [action_space.low, action_space.high] to [-1,1]
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
    g = 9.80665

    def __init__(self, configuration: str = None, mode: str = "nominal", render_mode: bool = False, realtime: bool = False, **kwargs):
        self.render_mode = render_mode
        # full state control
        self.n_actions = 3  # aileron da, elevator de, rudder dr
        if "attitude" in configuration.lower():
            # p, q, r, and alpha -> might change that to [phi, theta, psi, p, q, r] #TODO
            print("Attitude Control.\n")
            self.obs_idx = [0, 1, 2, 4]
        else:
            print("Full State Control.\n")
            self.obs_idx = range(10)  # 10 states, all states

        if self.render_mode:
            self.fdm = jsb.FGFDMExec("environments/JSBSim", None)
            self.fdm.load_model("citation")
            self.fdm.set_dt(self.dt)
            self.physicsPerSecond = int(1.0/self.dt)
            self.pauseDelay = 0.1  # how long an action is applied
            self.fdm.set_output_directive("data_output/flightgear.xml")
            self.fdm.load_ic("cruise_init", True)
            self.fdm.run_ic()
            self.fdm["gear/gear-cmd-norm"] = 0
            for _ in range(10):
                self.fdm.run()
            self.fdm.do_trim(1)
            self.fdm.print_simulation_configuration()
            self.realtime = realtime

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
        self.t_max: float = 20  # [s] to be changed

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
        self.x: np.ndarray = None  # observed state vector
        self.obs: np.ndarray = None
        self.last_obs: np.ndarray = None
        self.V0: float = None  # [m/s]
        self.last_u: np.ndarray = None  # last input action or control

        # references to track
        self.ref: List[BaseSignal] = None
        self.ref_values: np.ndarray = None
        self.theta_trim: float = 0.22  # standard theta trim in degree

        # actuator bounds
        if self.use_incremental:
            self.bound = np.deg2rad(25)  # [deg/s]
        else:
            self.bound = np.deg2rad(10)  # [deg]

        # state bounds
        self.max_theta = np.deg2rad(60.0)  # [deg]
        self.max_phi = np.deg2rad(75.0)  # [deg]

        if self.use_incremental:
            # aircraft state + actuator state + control states error (equal size with actuator states)
            self.n_obs: int = len(self.obs_idx) + 2*self.n_actions
        else:
            # aircraft state + control states error
            self.n_obs: int = len(self.obs_idx) + self.n_actions

        # state error initialization
        self.error: np.ndarray = np.zeros((self.n_actions))

        # error scaler
        self.error_scaler = np.array([1.0, 1.0, 4.0]) * 6/np.pi

        self.error_scaler = self.error_scaler[:self.n_actions]
        self.max_bound = np.ones(self.error.shape)  # bounds for state error

    @property
    def action_space(self) -> Box:
        """actuators bounds in radians"""
        return Box(
            low=-self.bound*np.ones(self.n_actions),
            high=self.bound*np.ones(self.n_actions),
            dtype=np.float64,
        )

    @property
    def observation_space(self) -> Box:
        """states bounds in degrees (phi, theta, psi, ...)"""
        return Box(
            low=-30*np.ones(self.n_obs),
            high=30*np.ones(self.n_obs),
            dtype=np.float64,
        )

    @property
    def p(self) -> float:
        """p: is the roll rate"""
        return self.x[0]

    @property
    def q(self) -> float:
        """q: is the pitch rate"""
        return self.x[1]

    @property
    def r(self) -> float:
        """r: is the yaw rate"""
        return self.x[2]

    @property
    def V(self) -> float:
        """v: is the airspeed"""
        return self.x[3]

    @property
    def alpha(self) -> float:
        """ alpha is the angle of attack"""
        return self.x[4]

    @property
    def beta(self) -> float:
        """ beta is the sideslip angle"""
        return self.x[5]

    @property
    def phi(self) -> float:
        """ phi is the roll angle"""
        return self.x[6]

    @property
    def theta(self) -> float:
        """ theta is the pitch angle"""
        return self.x[7]

    @property
    def psi(self) -> float:
        """ psi is the yaw angle"""
        return self.x[8]

    @property
    def h(self) -> float:
        """ h is the altitude"""
        return self.x[9]

    @property
    def nz(self) -> float:
        """ nz is the load factor"""
        return 1.0 + self.q * self.V / (self.g)

    def set_eval_mode(self, t_max: int = 80):
        """Switch to Evaluation Mode"""
        self.t_max = t_max
        self.eval_mode = True
        # if user_eval_refs is not None: self.user_refs = user_eval_refs
        printYellow(
            f"Switching to evaluation mode:\n Tmax = {self.t_max} seconds \n")

    def init_ref(self, **kwargs):
        """ Assuming n_actions: 3 """
        step_beta = Const(0.0, self.t_max, 0.0)

        # refs signals for phi and theta
        if "user_refs" not in kwargs:
            self.theta_trim = np.rad2deg(self.theta_trim)
            step_theta = RandomizedCosineStepSequence(
                t_max=self.t_max,
                ampl_max=30,
                block_width=self.t_max//5,
                smooth_width=self.t_max//6,
                n_levels=self.t_max//2,
                vary_timings=self.t_max/500
            )

            step_phi = RandomizedCosineStepSequence(
                t_max=self.t_max,
                ampl_max=30,
                block_width=self.t_max//5,
                smooth_width=self.t_max//6,
                n_levels=self.t_max//2,
                vary_timings=self.t_max/500
            )
        else:
            if not self.eval_mode:
                Warning(
                    "user reference signals have been given while env is not in evaluation mode")
            step_theta = kwargs["user_refs"]['theta_ref']
            step_phi = kwargs["user_refs"]['phi_ref']

        step_theta += Const(0.0, self.t_max, self.theta_trim)
        self.ref = [step_theta, step_phi, step_beta]

    def calc_reference_value(self) -> List[float]:
        # Calculates the reference value for the current time step (theta, phi, psi)
        self.ref_values = np.asarray(
            [np.deg2rad(ref_signal(self.t)) for ref_signal in self.ref])
        return self.ref_values

    def get_controlled_state(self) -> List[float]:
        """ Returns the values of the controlled states """
        ctrl = np.asarray([self.theta, self.phi, self.psi]
                          )  # replaced beta with psi
        return ctrl[:self.n_actions]

    def calc_error(self) -> np.ndarray:
        """ Calculates the error between the controlled states and the reference values """
        self.calc_reference_value()
        self.error[:self.n_actions] = self.ref_values - \
            self.get_controlled_state()

    def get_reward(self) -> float:
        """ Calculates the reward """
        self.calc_error()
        reward = -np.sum(np.abs(np.clip(self.error *
                         self.error_scaler, -self.max_bound, self.max_bound)))
        return reward/self.error.shape[0]

    def get_cost(self) -> float:
        """ the binary cost of  the last transition -> defining good behavior of the airplane"""

        if np.rad2deg(np.abs(self.alpha)) > 11.0 or \
           np.rad2deg(np.abs(self.phi)) > 0.75 * self.max_phi or \
           self.V < self.V0/3:
            return 1
        return 0

    def incremental_control(self, action: np.ndarray) -> np.ndarray:
        """ low-pass filtered Incremental control input for the citation model """
        return self.last_u + action * self.dt

    def pad_action(self, action: np.ndarray) -> np.ndarray:
        """ Pad action with zeros to correspond to the simulink model input dimensions"""
        # TODO might modify the control inputs to include throttle cmd for a full body control
        citation_input = np.pad(
            action, (0, self.n_actions_full - self.n_actions), 'constant', constant_values=(0.0,))
        return citation_input

    def check_bounds(self):
        """ Additional penalty for exceeding the bounds"""
        if self.t >= self.t_max \
                or np.abs(self.theta) > self.max_theta \
                or np.abs(self.phi) > self.max_phi \
                or self.h < 50:
            # negative reward for dying soon
            penalty = -1/self.dt * (self.t_max - self.t) * 2
            return True, penalty
        return False, 0.

    def reset(self, **kwargs):
        """ Reset the env to initial conditions """
        self.t = 0.0  # reset time
        self.aircraft.initialize()  # initialize the aircraft simulink model

        # initial input step (full-zero input to retrieve retrive the states)
        self.last_u = np.zeros(self.n_actions)

        # init state vector after padding the input
        _input = self.pad_action(self.last_u)
        self.x = self.aircraft.step(_input)

        # init aircraft reference conditions and randomized reference signal sequence
        self.V0 = self.V
        self.init_ref(**kwargs)

        # the observed state
        self.obs = np.hstack((self.error.flatten(), self.x[self.obs_idx]))
        self.last_obs = self.obs[:]

        if self.use_incremental:
            self.obs = np.hstack((self.obs, self.last_u))

        if self.render_mode:
            self.fdm.load_ic("cruise_init", True)
            self.fdm.run_ic()
            self.fdm.do_trim(1)

        return self.obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """ One Step in time by the agent in the environment
        Args:
            action: the action taken by the agent - Un-scaled input in the interval of [-1, 1]
        Returns:
            Tuple: new_state, the reward, is_done mask and info {reference signal value, time, cost of step}
        """
        is_done = False
        self.last_obs = self.obs

        # scale the action to the actuator rate bounds
        action = self.scale_action(action)  # rad

        if self.use_incremental:
            u = self.incremental_control(action)
        else:
            u = action
        # rendering
        if self.render_mode:
            self.render(u)

        # pad the input action to match the dimensions of the simulink model
        _input = self.pad_action(u)
        self.x = self.aircraft.step(_input)

        # get the reward
        reward = self.get_reward()

        # cost:
        cost = self.get_cost()

        # update observation based on perfect observations and actuator state:
        self.calc_error()  # update observation state error
        self.obs = np.hstack((self.error.flatten(), self.x[self.obs_idx]))
        self.last_u = u
        if self.use_incremental:
            self.obs = np.hstack((self.obs, self.last_u))

        # check the bounds and add corresponding penalty for terminated early:
        is_done, penalty = self.check_bounds()
        reward += penalty

        # step time:
        self.t += self.dt

        # info:
        info = {
            "ref": self.ref_values,
            "x": self.x,
            "t": self.t,
            "cost": cost,
        }
        return self.obs, reward, is_done, info

    def finish(self):
        """ Terminate the simulink simulation"""
        self.aircraft.terminate()

    def send_to_fg(self, action: np.ndarray):
        [de, da, dr] = action  # Un-scaled action in the interval of [-1, 1]

        self.fdm['fcs/aileron-cmd-norm'] = da
        self.fdm['fcs/elevator-cmd-norm'] = de
        self.fdm['fcs/rudder-cmd-norm'] = dr

    def render(self, action: np.ndarray):
        u = self.unscale_action(action)

        self.send_to_fg(u)

        for _ in range(int(self.pauseDelay*self.physicsPerSecond)):
            self.send_to_fg(u)
            self.fdm.run()
            if self.realtime:
                time.sleep(self.dt)
