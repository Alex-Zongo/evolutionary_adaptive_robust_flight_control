import jsbsim
import time
if __name__ == "__main__":
    fdm = jsbsim.FGFDMExec("./environments/JSBSim", None)
    fdm.load_model("c310")
    fdm.run_ic()
    fdm.print_simulation_configuration()
    
    def state_initializer(fdm):
        fdm.set_property_value("ic/h-sl-ft", 6000)
        fdm.set_property_value("ic/long-gc-deg", -90.0)
        fdm.set_property_value("ic/lat-gc-deg", 28.0)
        fdm.set_property_value("ic/vt-fps", 270)
        fdm.set_property_value("ic/psi-true-deg", 200)
        fdm["ic/phi-deg"] = 0.0
        fdm["fcs/gamma-deg"] = 0.0
        fdm.set_property_value("propulsion/set-running", -1)
        # fdm.set_property_value("propulsion/starter_cmd", 1)
        fdm["propulsion/refuel"] = True
        fdm["propulsion/active_engine"] = True
        # fdm["propulsion/engine[0]/set-running"] = 1
        # fdm["propulsion/engine[1]/set-running"] = 1
        # fdm.set_property_value("fcs/mixture-cmd-norm[0]", 1.0)
        # fdm.set_property_value("fcs/mixture-cmd-norm[1]", 1.0)
        
        # fdm.set_property_value("fcs/throttle-cmd-norm[0]", 1.0)
        # fdm.set_property_value("fcs/throttle-cmd-norm[1]", 1.0)
        

    state_initializer(fdm)
    fdm.run_ic()
    
    # fdm.run_ic()
    # frame_duration = fdm.get_delta_t()
    # end = 1E99
    # time.sleep(0.05)
    # fdm.set_property_value("simulation/do_simple_trim", 1)
    # result = fdm.run()
    # while result and fdm.get_sim_time() < end:
    #     fdm.check_incremental_hold()
    #     result = fdm.run()
        
    
    
    # for i in range(10):
    #     fdm.run()
    # fdm.print_property_catalog()
    
    print(fdm["position/h-sl-ft"])
    print(fdm["velocities/vt-fps"])
    time.sleep(5.0)
    fdm["simulation/do_simple_trim"] = 1