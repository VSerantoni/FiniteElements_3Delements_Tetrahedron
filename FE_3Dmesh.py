# Version python du code de Franck Jourdan pouvant charger un maillage en .msh et réaliser le calcul EF

import os
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import function_3d_FE as fc3d
import copy as cp

# Chargement du fichier .msh
root = 'G:\\Os Nasal\\Matlab\\Finit Elements\\'
File2Load = 'barre2'
filename = root + File2Load + '.msh'

f = open(filename, 'r')
next(f)
next(f)
next(f)
next(f)
Nbn = int(next(f))  # Nombre de noeuds
tmp = np.zeros([Nbn, 4])
for i in range(0, Nbn):
    t = next(f)
    tmp[i, :] = [float(k) for k in t[:-1].split()]

coordinates = tmp[:, 1:]

if 1 == 0:
    plt.figure()
    axs = plt.gca()
    ax = plt.axes(projection='3d')
    plt.grid()
    for ii in range(0, coordinates.shape[0], 1):
        ax.plot3D(coordinates[ii, 0], coordinates[ii, 1], coordinates[ii, 2], 'b.')

    axs.set_aspect('equal')
    plt.show()

next(f)
next(f)
Nbe = int(next(f))  # Nombre d'éléments

tmp = np.zeros([Nbe, 9])
for i in range(0, Nbe):
    t = next(f)
    tmp[i, :] = [float(k) for k in t[:-1].split()]

elements = tmp[:, 5:]

f.close()

# Nombre de dof
GDof = 3 * Nbn

# Paramètres matériaux
E = 1500
nu = 0.3
Fg = 80  # Global Load
mu = E / (2 * (1 + nu))
lbd = E * nu / ((1 + nu) * (1 - 2 * nu))

# Liste des ddl bloques
tab_dirichlet1 = np.array([1, 2, 3, 4, 25])  # Noeuds bloqués
tab_dirichlet2 = np.array([5, 6, 7, 8, 46])  # Noeuds avec force imposé
tab_dirichlet = np.append(tab_dirichlet1, tab_dirichlet2)

# Vecteur second membre
ff2 = np.zeros((3 * coordinates.shape[0], 1))

# Assemblage de la matrice de rigidite
# Trois vecteurs pour stocker matrice de rigidité directement en une matrice creuse
il = np.zeros((12 * 12 * Nbe))
jc = np.zeros((12 * 12 * Nbe))
sck = np.zeros((12 * 12 * Nbe))

for j in range(0, Nbe):
    Iel = 3 * elements[j, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])] \
          - np.array([2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
    Kj = fc3d.stima3D_tetra(coordinates[elements[j, :].astype(int) - 1, :], lbd, mu)

    jc[np.arange(144 * j, 144 * (j + 1), 1)] = np.reshape(np.ones((12, 1)) * Iel, (144,)) - 1
    il[np.arange(144 * j, 144 * (j + 1), 1)] = np.reshape(Iel.reshape((-1, 1)) * np.ones((1, 12)), (144,)) - 1
    sck[np.arange(144 * j, 144 * (j + 1), 1)] = np.reshape(Kj, (144,))

    # A[ I I.reshape((-1, 1))] = A[ I, I.reshape((-1, 1))] + Kj

A = sparse.csr_matrix((sck, (il.astype(int), jc.astype(int))))

# Vecteur numéro des noeuds selon l'axe
j = 0
ddl_bloques = np.zeros(3 * len(tab_dirichlet1))
for i in range(0, len(tab_dirichlet1)):
    ddl_bloques[j] = 3 * tab_dirichlet1[i] - 3
    ddl_bloques[j + 1] = 3 * tab_dirichlet1[i] - 2
    ddl_bloques[j + 2] = 3 * tab_dirichlet1[i] - 1
    j += 3

# Prise en compte de la force dans le second membre
Fn = Fg / len(tab_dirichlet2)
for i in range(0, len(tab_dirichlet2)):
    ff2[3 * tab_dirichlet2[i] - 1, 0] = Fn  # z
    ff2[3 * tab_dirichlet2[i] - 2, 0] = 0  # y
    ff2[3 * tab_dirichlet2[i] - 3, 0] = 0  # x

resolution = 1  # 0: methode de réduction de matrice; 1: methode de pénalité
if resolution == 0:
    activeDof = np.expand_dims(np.setdiff1d(np.arange(0, GDof), ddl_bloques), axis=0)
    U1 = spsolve(A[activeDof.T, activeDof], ff2[activeDof.T, 0])
    U = np.zeros(GDof)
    U[activeDof] = U1
    ff = A @ U

elif resolution == 1:
    # Prise en compte des ddl bloques dans la matrice A et f
    Aorigine = cp.deepcopy(A)
    for j in range(0, 3 * len(tab_dirichlet1)):
        A[ddl_bloques[j], ddl_bloques[j]] = 1e10

    # for i in range(0, len(tab_dirichlet1)):
    #     ff2[3 * tab_dirichlet1[i]] = 1e10 * Uimp      # z
    #     ff2[3 * tab_dirichlet1[i] - 1] = 1e10 * Uimp  # y
    #     ff2[3 * tab_dirichlet1[i] - 2] = 1e10 * Uimp  # x

    U = spsolve(A, ff2)
    ff = Aorigine @ U

# New position
coordinates_line = coordinates.flatten()
coordinates_new = coordinates_line + U
coordinates_new = np.reshape(coordinates_new, (Nbn, 3))

plt.figure()
axs = plt.gca()
ax = plt.axes(projection='3d')
plt.grid()
for ii in range(0, coordinates_new.shape[0], 1):
    ax.plot3D(coordinates[ii, 0], coordinates[ii, 1], coordinates[ii, 2], 'k.', markersize=15)
    ax.plot3D(coordinates_new[ii, 0], coordinates_new[ii, 1], coordinates_new[ii, 2], 'rx', markersize=15)

axs.set_aspect('equal')
plt.title('Barre2.msh')
plt.legend(['Struture Initiale', 'Structure déformée'])
# plt.show()

# Affichage des déformations par axe
# Théorie sur z: Uz_max = 0.26667
# Résultats: 0.26598
fc3d.plotDef(coordinates_new, U, 'mean')  # dernier paramètre : 'x', 'y', 'z' ou 'mean'

# Récupération des déformations et contraintes par éléments
epsi, sigma, coordinates_center = fc3d.strain_stress(elements, coordinates, U, lbd, mu)

# Affichage des déformations par éléments
# Théorie sur z: 0.0533
# Résultats moyen: 0.0526
fc3d.plotStrain(coordinates_center, epsi, 'VM')  # dernier paramètre : 'x', 'y', 'z', 'VM' ou 'tresca'

# Affichage des contraintes par éléments
# Théorie sur z: 80
# Résultats moyen: 80.09
fc3d.plotStress(coordinates_center, sigma, 'VM')  # dernier paramètre : 'x', 'y', 'z', 'VM' ou 'tresca'

plt.show()
os.system("pause")
