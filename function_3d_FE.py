import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.linalg as la


## Matrice de rigidité par éléments
def stima3D_tetra(coor, lbd, mu):
    Ve = np.concatenate((np.array([[1, 1, 1, 1]]).T, coor), axis=1)
    forme_deriv = np.linalg.solve(Ve, np.identity(4))

    # Matrice de forme
    Be = np.zeros([6, 12])
    Be[np.array([[0, 3, 4]]).T, [0, 3, 6, 9]] = forme_deriv[1:, :]
    Be[np.array([[3, 1, 5]]).T, [1, 4, 7, 10]] = forme_deriv[1:, :]
    Be[np.array([[4, 5, 2]]).T, [2, 5, 8, 11]] = forme_deriv[1:, :]

    # Matrice de matière
    De = np.zeros([6, 6])
    De[np.array([[0, 1, 2]]).T, [0, 1, 2]] = lbd * np.ones((3, 3)) + 2 * mu * np.identity(3)
    De[np.array([[3, 4, 5]]).T, [3, 4, 5]] = mu * np.identity(3)

    # Volume de l'élément
    vol = (1 / 6) * np.linalg.det(Ve)
    Kj = vol * Be.T @ De @ Be

    return Kj


## Affichage déplacements
def plotDef(coor, U, axe):
    cmap = cm.get_cmap('jet')

    if axe != 'x' and axe != 'y' and axe != 'z' and axe != 'mean':
        print('Error in the axis to plot')
        print(' axes must be : x or y or z or mean')
        return

    # Get the max value
    Umax = 0.
    if axe == 'x':
        idx = 3
    elif axe == 'y':
        idx = 2
    elif axe == 'z':
        idx = 1

    for ii in range(0, coor.shape[0], 1):
        if axe == 'x' or axe == 'y' or axe == 'z':
            Uii = abs(U[(ii + 1) * 3 - idx])
            if Uii > Umax:
                Umax = Uii

        elif axe == 'mean':
            Uii = abs(np.mean([U[(ii + 1) * 3 - 3], U[(ii + 1) * 3 - 2], U[(ii + 1) * 3 - 1]]))
            if Uii > Umax:
                Umax = Uii

    # Plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.grid()
    # axs.set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')

    if axe == 'x' or axe == 'y' or axe == 'z':
        for ii in range(0, coor.shape[0], 1):
            Uii = abs(U[(ii + 1) * 3 - idx])
            ax.plot3D(coor[ii, 0], coor[ii, 1], coor[ii, 2], '.',
                      color=cmap(Uii / Umax), markersize=15)

        if axe == 'x':
            plt.title('Déplacements sur x')
        elif axe == 'y':
            plt.title('Déplacements sur y')
        elif axe == 'z':
            plt.title('Déplacements sur z')

    elif axe == 'mean':
        for ii in range(0, coor.shape[0], 1):
            Uii = abs(np.mean([U[(ii + 1) * 3 - 3], U[(ii + 1) * 3 - 2], U[(ii + 1) * 3 - 1]]))
            ax.plot3D(coor[ii, 0], coor[ii, 1], coor[ii, 2], '.',
                    color=cmap(Uii / Umax), markersize=15)

        plt.title('Déplacements moyen')

    ax = fig.add_subplot(1, 16, 9)
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=Umax)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Déplacements (mm)')
    # fig.show()


## Récupération déforamtions et contraintes par éléments
def strain_stress(elements, coordinates, U, lbd, mu):
    strain_nodes = np.zeros((elements.shape[0], 6))
    stress_nodes = np.zeros((elements.shape[0], 6))
    coor_center = np.zeros((elements.shape[0], 3))

    for e in range(0, elements.shape[0]):
        coor_nodes = coordinates[elements[e, :].astype(int) - 1, :]
        Ve = np.concatenate((np.array([[1, 1, 1, 1]]).T, coor_nodes), axis=1)
        Phi_deriv = np.linalg.solve(Ve, np.identity(4))

        # Central node
        coor_center[e, :] = [np.mean(coor_nodes[:, 0]), np.mean(coor_nodes[:, 1]), np.mean(coor_nodes[:, 2])]

        # Matrice de forme
        Be = np.zeros([6, 12])
        Be[np.array([[0, 3, 4]]).T, [0, 3, 6, 9]] = Phi_deriv[1:, :]
        Be[np.array([[3, 1, 5]]).T, [1, 4, 7, 10]] = Phi_deriv[1:, :]
        Be[np.array([[4, 5, 2]]).T, [2, 5, 8, 11]] = Phi_deriv[1:, :]

        # Déplacement des noeuds
        U0 = U[3 * elements[e, 0].astype(int) - 3:3 * elements[e, 0].astype(int)]
        U1 = U[3 * elements[e, 1].astype(int) - 3:3 * elements[e, 1].astype(int)]
        U2 = U[3 * elements[e, 2].astype(int) - 3:3 * elements[e, 2].astype(int)]
        U3 = U[3 * elements[e, 3].astype(int) - 3:3 * elements[e, 3].astype(int)]
        U_node = np.reshape(np.vstack([U0, U1, U2, U3]), (12, 1))

        # Srain element
        epsi = Be @ U_node
        strain_nodes[e, :] = np.squeeze(epsi)

        # Stress element
        De = np.zeros([6, 6])
        De[np.array([[0, 1, 2]]).T, [0, 1, 2]] = lbd * np.ones((3, 3)) + 2 * mu * np.identity(3)
        De[np.array([[3, 4, 5]]).T, [3, 4, 5]] = mu * np.identity(3)
        sigma = De @ epsi
        stress_nodes[e, :] = np.squeeze(sigma)

    return strain_nodes, stress_nodes, coor_center


## Affichage déformations
def plotStrain(coordinates_center, epsi, axe):
    cmap = cm.get_cmap('jet')

    if axe != 'x' and axe != 'y' and axe != 'z' and axe != 'VM' and axe != 'tresca':
        print('Error in the axis to plot')
        print(' axes must be : x or y or z or VM or tresca')
        return

    # Get the max and min value
    Epsi_max = 0.
    if axe == 'x':
        idx = 0
    elif axe == 'y':
        idx = 1
    elif axe == 'z':
        idx = 2

    for ii in range(0, epsi.shape[0]):
        if axe == 'x' or axe == 'y' or axe == 'z':
            Epsi_ii = abs(epsi[ii, idx])
        elif axe == 'VM' or axe == 'tresca':
            epsi_matrix = np.array([[epsi[ii, 0], 0.5 * epsi[ii, 5], 0.5 * epsi[ii, 4]],
                                    [0.5 * epsi[ii, 5], epsi[ii, 1], 0.5 * epsi[ii, 3]],
                                    [0.5 * epsi[ii, 4], 0.5 * epsi[ii, 3], epsi[ii, 2]]])
            # Valeurs propres
            eigvals, eigvecs = la.eig(epsi_matrix)
            eigvals = eigvals.real

            if axe == 'VM':
                Epsi_ii = (1 / np.sqrt(2)) * np.sqrt((eigvals[0] - eigvals[1]) ** 2
                                                     + (eigvals[1] - eigvals[2]) ** 2 + (eigvals[2] - eigvals[0]) ** 2)
            else:
                Epsi_ii = np.max(
                    [abs(eigvals[0] - eigvals[1]), abs(eigvals[1] - eigvals[2]), abs(eigvals[2] - eigvals[0])])

        if ii == 0:
            Epsi_min = Epsi_ii

        if Epsi_ii > Epsi_max:
            Epsi_max = Epsi_ii

        if Epsi_ii < Epsi_min:
            Epsi_min = Epsi_ii

    # Plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.grid()
    # axs.set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    if axe == 'x' or axe == 'y' or axe == 'z':
        for ii in range(0, coordinates_center.shape[0]):
            Epsi_ii = abs(epsi[ii, idx])
            ax.plot3D(coordinates_center[ii, 0], coordinates_center[ii, 1], coordinates_center[ii, 2], '.',
                      color=cmap(Epsi_ii / Epsi_max), markersize=15)

        if axe == 'x':
            plt.title('Déformations sur x')
        elif axe == 'y':
            plt.title('Déformations sur y')
        elif axe == 'z':
            plt.title('Déformations sur z')

    elif axe == 'VM' or axe == 'tresca':
        for ii in range(0, coordinates_center.shape[0]):
            epsi_matrix = np.array([[epsi[ii, 0], 0.5 * epsi[ii, 5], 0.5 * epsi[ii, 4]],
                                    [0.5 * epsi[ii, 5], epsi[ii, 1], 0.5 * epsi[ii, 3]],
                                    [0.5 * epsi[ii, 4], 0.5 * epsi[ii, 3], epsi[ii, 2]]])
            # Valeurs propres
            eigvals, eigvecs = la.eig(epsi_matrix)
            eigvals = eigvals.real

            if axe == 'VM':
                Epsi_ii = (1 / np.sqrt(2)) * np.sqrt((eigvals[0] - eigvals[1]) ** 2
                                                 + (eigvals[1] - eigvals[2]) ** 2 + (eigvals[2] - eigvals[0]) ** 2)
                plt.title('Déformations (Von Mises)')
            else:
                Epsi_ii = np.max(
                    [abs(eigvals[0] - eigvals[1]), abs(eigvals[1] - eigvals[2]), abs(eigvals[2] - eigvals[0])])
                plt.title('Déformations (Tresca)')

            ax.plot3D(coordinates_center[ii, 0], coordinates_center[ii, 1], coordinates_center[ii, 2], '.',
                      color=cmap(Epsi_ii / Epsi_max), markersize=15)

    ax = fig.add_subplot(1, 16, 9)
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=Epsi_min, vmax=Epsi_max)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Déformations')
    # fig.show()


## Affichage contraintes
def plotStress(coordinates_center, sigma, axe):
    cmap = cm.get_cmap('jet')

    if axe != 'x' and axe != 'y' and axe != 'z' and axe != 'VM' and axe != 'tresca':
        print('Error in the axis to plot')
        print(' axes must be : x or y or z or VM or tresca')
        return

    # Get the max and min value
    Sigma_max = 0.
    if axe == 'x':
        idx = 0
    elif axe == 'y':
        idx = 1
    elif axe == 'z':
        idx = 2

    for ii in range(0, sigma.shape[0]):
        if axe == 'x' or axe == 'y' or axe == 'z':
            Sigma_ii = abs(sigma[ii, idx])
        elif axe == 'VM' or axe == 'tresca':
            sigma_matrix = np.array([[sigma[ii, 0], sigma[ii, 5], sigma[ii, 4]],
                                     [sigma[ii, 5], sigma[ii, 1], sigma[ii, 3]],
                                     [sigma[ii, 4], sigma[ii, 3], sigma[ii, 2]]])
            # Valeurs propres
            eigvals, eigvecs = la.eig(sigma_matrix)
            eigvals = eigvals.real

            if axe == 'VM':
                Sigma_ii = (1 / np.sqrt(2)) * np.sqrt((eigvals[0] - eigvals[1]) ** 2
                                                      + (eigvals[1] - eigvals[2]) ** 2 + (eigvals[2] - eigvals[0]) ** 2)
            else:
                Sigma_ii = np.max(
                    [abs(eigvals[0] - eigvals[1]), abs(eigvals[1] - eigvals[2]), abs(eigvals[2] - eigvals[0])])

        if ii == 0:
            Sigma_min = Sigma_ii

        if Sigma_ii > Sigma_max:
            Sigma_max = Sigma_ii

        if Sigma_ii < Sigma_min:
            Sigma_min = Sigma_ii

    # Plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.grid()
    # axs.set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    if axe == 'x' or axe == 'y' or axe == 'z':
        for ii in range(0, coordinates_center.shape[0]):
            Sigma_ii = abs(sigma[ii, idx])
            ax.plot3D(coordinates_center[ii, 0], coordinates_center[ii, 1], coordinates_center[ii, 2], '.',
                      color=cmap(Sigma_ii / Sigma_max), markersize=15)

        if axe == 'x':
            plt.title('Contraintes sur x')
        elif axe == 'y':
            plt.title('Contraintes sur y')
        elif axe == 'z':
            plt.title('Contraintes sur z')

    elif axe == 'VM' or axe == 'tresca':
        for ii in range(0, coordinates_center.shape[0]):
            sigma_matrix = np.array([[sigma[ii, 0], sigma[ii, 5], sigma[ii, 4]],
                                     [sigma[ii, 5], sigma[ii, 1], sigma[ii, 3]],
                                     [sigma[ii, 4], sigma[ii, 3], sigma[ii, 2]]])
            # Valeurs propres
            eigvals, eigvecs = la.eig(sigma_matrix)
            eigvals = eigvals.real

            if axe == 'VM':
                Sigma_ii = (1 / np.sqrt(2)) * np.sqrt((eigvals[0] - eigvals[1]) ** 2
                                                     + (eigvals[1] - eigvals[2]) ** 2 + (eigvals[2] - eigvals[0]) ** 2)
                plt.title('Contraintes (Von Mises)')
            else:
                Sigma_ii = np.max(
                    [abs(eigvals[0] - eigvals[1]), abs(eigvals[1] - eigvals[2]), abs(eigvals[2] - eigvals[0])])
                plt.title('Contraintes (Tresca)')

            ax.plot3D(coordinates_center[ii, 0], coordinates_center[ii, 1], coordinates_center[ii, 2], '.',
                      color=cmap(Sigma_ii / Sigma_max), markersize=15)

    ax = fig.add_subplot(1, 16, 9)
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=Sigma_min, vmax=Sigma_max)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Contraintes')
    # fig.show()
