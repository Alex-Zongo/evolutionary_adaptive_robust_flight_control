import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
plt.rcParams['figure.figsize'] = [15, 12]


def plot(data, name, fault):
    """Evaluation Results on the fault

    Args:
        data (array timex20): [ref_values, actions, x_lst, rewards]
        name (str): name of the figure:
        fault (str): fault name
    Returns:
        fig (figure): figure object
        rewards (array): rewards #TODO: replace with rewards fig plot
    """
    print(data.shape)
    ref_values = data[:, :3]  # ref_values [theta, phi, beta] unit: rad
    actions = data[:, 3:6]  # actions [de, da, dr] unit: rad
    # x_lst [p, q, r, V, alpha, beta, phi, theta, psi, h] unit: rad, rad/s, m, m/s
    x_lst = data[:, 6:16]
    rewards = data[:, -2]

    # ************* PARSING THE DATA *************
    # TODO: Convert from read to deg
    _t = data[:, -1]
    theta_ref, phi_ref, psi_ref = ref_values[:, 0], ref_values[:, 1], \
        ref_values[:, 2]

    p, q, r, alpha, beta, phi, theta, psi = x_lst[:, 0], x_lst[:, 1], \
        x_lst[:, 2], x_lst[:, 4], x_lst[:, 5], x_lst[:, 6], x_lst[:, 7], \
        x_lst[:, 8]

    de, da, dr = actions[:, 0], actions[:, 1], actions[:, 2]
    # ******* convert the rad values to degrees:

    theta_ref = np.rad2deg(theta_ref)
    phi_ref = np.rad2deg(phi_ref)
    psi_ref = np.rad2deg(psi_ref)
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)
    psi = np.rad2deg(psi)
    alpha = np.rad2deg(alpha)
    de = np.rad2deg(de)
    da = np.rad2deg(da)
    dr = np.rad2deg(dr)

    # *****************************************************
    # ********* Checking Faults **************
    # TODO: do something: saturation limits to plot
    # check each fault:
    # plot the saturation limits accordingly
    # *********  Plotting  ****************
    fig, axis = plt.subplots(3, 2)
    labels = ['Tracked State [deg]', 'Reference [deg]',
              'Tracked State Rate [deg/s]', 'Actuator Deflection [deg]']

    l1 = axis[0, 0].plot(_t, theta, color='red')
    l2 = axis[0, 0].plot(_t, theta_ref, color='black', linestyle='--')
    l3 = axis[0, 0].plot(_t, q, '*--', color='pink')
    # axis[0, 0].plot(_t, alpha, 'y-.', label='alpha')
    # axis[0, 0].legend()
    axis[0, 0].set_ylabel(r'$\theta, q$', fontsize=20)

    l4 = axis[0, 1].plot(_t, de, color='green')
    axis[0, 1].set_ylabel(r'$\delta_e$', fontsize=20)
    print('$\theta \phi \psi$')
    axis[1, 0].plot(_t, phi, 'r')
    axis[1, 0].plot(_t, phi_ref, 'k--', )
    axis[1, 0].plot(_t, p, '*--', color='pink')
    axis[1, 0].set_ylabel(r'$\phi, p$', fontsize=20)

    axis[1, 1].plot(_t, da, color='green')
    axis[1, 1].set_ylabel(r'$\delta_a$', fontsize=20)

    axis[2, 0].plot(_t, psi, 'r')
    axis[2, 0].plot(_t, psi_ref, 'k--')
    axis[2, 0].set_xlabel('Time [S]', fontsize=20)
    axis[2, 0].set_ylabel(r'$\psi$', fontsize=20)

    axis[2, 1].plot(_t, dr, color='green')
    axis[2, 1].set_ylabel(r'$\delta_r$', fontsize=20)
    # axis[2, 1].legend()

    axis[2, 1].set_xlabel('Time [S]', fontsize=20)
    fig.suptitle(name, fontsize=20)
    fig.legend([l1, l2, l3, l4], labels=labels,
               loc='lower center', ncol=4, fontsize=15)
    # plt.legend()
    # plt.show()
    # ***************************************
    return fig, rewards


if __name__ == "__main__":
    # testing the function
    pass
