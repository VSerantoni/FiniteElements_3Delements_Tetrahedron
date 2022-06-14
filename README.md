# FiniteElements

Use FE_3Dmesh.py to run a linear Finit Elements code.

The mesh must use only tetrahedron elements and be saved with gmsh in .msh (Version 2 ASCII)
- In tab_dirchlet1 you can indicate your node with 0 dof
- In tab_dirchlet2 you can indicate your node with a specif load

Choose your resolution method (penalty or reduced stifness matrix) \
If you have displacements boundary conditions, use the penalty method (reduced stifness matrix not yet working)

Enjoy
