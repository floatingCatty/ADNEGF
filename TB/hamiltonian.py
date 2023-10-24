"""
The module contains a library of classes facilitating computations of Hamiltonian matrices.
"""
from __future__ import print_function, division
from __future__ import absolute_import
from collections import OrderedDict
from functools import reduce
import torch
import logging
import inspect
from operator import mul
import numpy as np
import scipy
from TB.abstract_interfaces import AbstractBasis
from TB.structure_designer import StructDesignerXYZ, CyclicTopology
from TB.diatomic_matrix_element import me, me_diatomic, get_bl_param
from TB.orbitals import Orbitals
from TB.aux_functions import dict2xyz, xyz2tensor
from TB.block_tridiagonalization import find_nonzero_lines, split_into_subblocks_optimized, cut_in_blocks, \
    split_into_subblocks
from torch.autograd import gradcheck

unique_distances = set()


class BasisTB(AbstractBasis, StructDesignerXYZ):
    """The class contains information about sets of quantum numbers and
    dimensionality of the Hilbert space.
    It is also equipped with the member functions translating quantum numbers
    into a matrix index and vise versa using a set of index offsets.
    Examples
    --------
    >>> from nanonet.verbosity import set_verbosity
    >>> import nanonet.tb as tb
    >>> set_verbosity(0)
    >>> orb = tb.Orbitals('A')
    >>> orb.add_orbital(title='s', energy=-1)
    >>> orb.add_orbital(title='1s', energy=0)
    >>> tb.Orbitals('B').add_orbital(title='s', energy=0)
    >>> xyz = '''2
    ... Two atoms
    ... A1 0 0 0
    ... B1 0 0 1'''
    >>> basis = tb.hamiltonian.BasisTB(xyz=xyz)
    >>> print(basis.basis_size)
    3
    >>> print(basis.atom_list['B1'])
    [0. 0. 1.]
    >>> print(basis.qn2ind({'atoms': 0, 'l': 0}))
    0
    >>> print(basis.qn2ind({'atoms': 0, 'l': 1}))
    1
    >>> print(basis.qn2ind({'atoms': 1, 'l': 0}))
    2
    >>> print(type(basis.orbitals_dict['A']))
    <class 'nanonet.tb.orbitals.Orbitals'>
    """

    def __init__(self, **kwargs):

        # parent class StructDesignerXYZ stores atom list initialized from xyz-file
        super(BasisTB, self).__init__(**kwargs)

        # each entry of the dictionary stores a label of the atom species as a key and
        # corresponding Atom object as a value. Each atom object contains infomation about number,
        # energy and symmetry of the orbitals
        self._orbitals_dict = Orbitals.atoms_factory(list(self.num_of_species.keys()))


        # `quantum_number_lims` counts number of species and corresponding number
        # of orbitals for each; each atom kind is enumerated
        self.quantum_numbers_lims = []
        for item in list(self.num_of_species.keys()):
            self.quantum_numbers_lims.append(OrderedDict([('atoms', self.num_of_species[item]),
                                                          ('l', self.orbitals_dict[item].num_of_orbitals)]))
        # count total number of basis functions
        self.basis_size = 0
        for item in self.quantum_numbers_lims:
            self.basis_size += reduce(mul, list(item.values()))

        # compute offset index for each atom
        self._offsets = [0]
        for j in range(len(self.atom_list) - 1):
            self._offsets.append(self.orbitals_dict[list(self.atom_list.keys())[j]].num_of_orbitals)
        self._offsets = np.cumsum(self._offsets)


        # make a log
        # logging.info("Basis set \n Num of species {} \n".format(self.num_of_species))
        # for key, label in self._orbitals_dict.items():
        #     logging.info("\n {} {} ".format(key, label.generate_info()))
        # logging.info("---------------------------------\n")

    def qn2ind(self, qn):
        """Computes a matrix index of an matrix element from the index of atom and the index of atomic orbital.
        Parameters
        ----------
        qn : dict
            A dictionary with two keys `atoms` and `l`, where the fist one is the atom index and
            the later is the orbital index.
        Returns
        -------
        type int
            Index of the TB matrix
        """

        qn = OrderedDict(qn)

        if list(qn.keys()) == list(self.quantum_numbers_lims[0].keys()):  # check if the input is
            # a proper set of quantum numbers
            return self._offsets[qn['atoms']] + qn['l']
        else:
            raise IndexError("Wrong set of quantum numbers")

    def ind2qn(self, ind):
        """
        Parameters
        ----------
        ind :

        Returns
        -------
        """
        pass  # TODO

    @property
    def orbitals_dict(self):
        """Returns the dictionary data structure of orbitals. In the dictionary"""

        class MyDict(dict):
            """ """

            def __getitem__(self, key):
                key = ''.join([i for i in key if not i.isdigit()])
                return super(MyDict, self).__getitem__(key)

        return MyDict(self._orbitals_dict)


class Hamiltonian(BasisTB):
    """Class defines a Hamiltonian matrix as well as a set of member-functions
    allowing to build, diagonalize and visualize the matrix.
    Examples
    --------
    >>> from nanonet.verbosity import set_verbosity
    >>> import nanonet.tb as tb
    >>> set_verbosity(0)
    >>> tb.Orbitals('A').add_orbital(title='s', energy=-1)
    >>> tb.Orbitals('B').add_orbital(title='s', energy=-2)
    >>> xyz_file = '''2
    ... Two atoms
    ... A1 0 0 0
    ... B1 0 0 1.5'''
    >>> tb.set_tb_params(PARAMS_A_B={'ss_sigma': 0.1})
    >>> h = tb.Hamiltonian(xyz=xyz_file, nn_distance=2.0).initialize()
    >>> h.h_matrix
    array([[-1. +0.j,  0.1+0.j],
           [ 0.1+0.j, -2. +0.j]])
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        nn_distance : float, list
            Nearest neighbour distance of distances
            if the tight-binding method is beyond the first-nearest neighbour approximation (Default value = 2.39)
        radial_dep : func
             Radial dependence function (Default value = None)
        """

        nn_distance = kwargs.get('nn_distance', 2.39)
        self.int_radial_dependence = None
        nn_distance = self._set_nn_distances(nn_distance)
        self.compute_overlap = kwargs.get('comp_overlap', False)
        self.compute_angular = kwargs.get('comp_angular_dep', True)
        self.dtype = kwargs.get('dtype', torch.complex128)

        # logging.info('The verbosity level is {}'.format(verbosity.VERBOSITY))
        # logging.info('The radius of the neighbourhood is {} Ang'.format(nn_distance))
        # logging.info("\n---------------------------------\n")

        kwargs['nn_distance'] = nn_distance

        if not isinstance(kwargs['xyz'], str):
            kwargs['xyz'] = dict2xyz(kwargs['xyz'])
        if not isinstance(kwargs['xyz_new'], str):                      # add
            kwargs['xyz_new'] = dict2xyz(kwargs['xyz_new'])

        super(Hamiltonian, self).__init__(**kwargs)

        self._coords = None  # coordinates of sites
        self.h_matrix = None  # Hamiltonian for an isolated system
        self.ov_matrix = None  # overlap matrix for an isolated system
        self.h_matrix_bc_factor = None  # exponential Bloch factors for pbc
        self.h_matrix_bc_add = None  # additive Bloch exponentials for pbc
        self.ov_matrix_bc_add = None  # additive Bloch exponentials for pbc
        # (interaction with virtual neighbours
        # in adacent primitive cells due to pbc)

        self.h_matrix_left_lead = None
        self.h_matrix_right_lead = None
        self.k_vector = 0  # default value of the wave vector
        self.ct = None
        self.radial_dependence = None
        self.so_coupling = kwargs.get('so_coupling', 0.0)

        radial_dep = kwargs.get('radial_dep', None)
        # if radial_dep is not None:
        #     xyz = kwargs.get('eqxyz', None)
        #     if xyz is not None:
        #         try:
        #             with open(xyz, 'r') as read_file:
        #                 reader = read_file.read()
        #         except IOError:
        #             reader = xyz
        #
        #         eqlabels, eqcoords = xyz2tensor(reader)
        #     else:
        #         eqlabels = kwargs.get('labels', None)
        #         eqcoords = kwargs.get('coords', None)

            # if eqlabels != None and eqcoords != None:
            #     self.eqAtom_list = OrderedDict(list(zip(eqlabels, eqcoords)))
            #     self.allow_strain = True
        # if radial_dep is None:
        #     logging.info('Radial dependence function: None')
        #     logging.info("\n---------------------------------\n")
        # else:
        #     logging.info('Radial dependence function:\n\n{}'.format(inspect.getsource(radial_dep)))
        #     logging.info("\n---------------------------------\n")

        self.radial_dependence = radial_dep


    def initialize(self):
        # print(list(self._atom_list.values()))
        # print("aaaaaaaaaaaaaaaa")
        # print([x.detach() for x in self._atom_list.values()])
        if self.sort_func is not None:
            labels = [item[0] for item in self._atom_list_np.items()]
            coords = [item[1] for item in self._atom_list_np.items()]
            self._sort_device(labels, coords)
            

            # reconstruct offset
            self._offsets = [0]
            for j in range(len(self.atom_list) - 1):
                self._offsets.append(self.orbitals_dict[list(self.atom_list.keys())[j]].num_of_orbitals)
            self._offsets = np.cumsum(self._offsets)
        # self._kd_tree = scipy.spatial.cKDTree(torch.stack([x.detach() for x in self._atom_list.values()]),
        #                                       leafsize=1,
        #                                       balanced_tree=True)
        self._kd_tree = scipy.spatial.cKDTree(np.array(list(self._atom_list_np.values())),
                                              leafsize=1,
                                              balanced_tree=True)
        # print("bbbbbbbbbbbbbbbb")
        # self._bond_length = [[torch.norm(x - y) for y in self._atom_list_new.values()] \
        #     for x in self._atom_list_new.values()]
        """Compute matrix elements of the Hamiltonian.
        Returns
        -------
        type Hamiltonian
            Returns the instance of the class Hamiltonian
        """
        self._coords = [0 for _ in range(self.basis_size)]
        # initialize Hamiltonian matrices
        self.h_matrix = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)
        self.h_matrix_bc_add = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)
        self.h_matrix_bc_factor = torch.ones((self.basis_size, self.basis_size), dtype=self.dtype)

        if self.compute_overlap:
            self.ov_matrix = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)
            self.ov_matrix_bc_add = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)

        # loop over all nodes
        for j1 in range(self.num_of_nodes):

            # find neighbours for each node
            list_of_neighbours = self.get_neighbours(j1)

            for j2 in list_of_neighbours:
                # on site interactions
                if j1 == j2:
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                        self.h_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1)

                        if self.compute_overlap:
                            self.ov_matrix[ind1, ind1] = self._get_me(j1, j2, l1, l1, overlap=True)
                        # update coords
                        self._coords[ind1] = list(self.atom_list.values())[j1]

                        if self.so_coupling != 0:
                            for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                                ind2 = self.qn2ind([('atoms', j1), ('l', l2)], )
                                self.h_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2)

                # nearest neighbours interaction
                else:
                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j2]].num_of_orbitals):
                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )

                            self.h_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2)
                           
                            if self.compute_overlap:
                                self.ov_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2, overlap=True)

        # logging.info("Unique distances: \n    {}".format("\n    ".join(unique_distances)))
        # logging.info("---------------------------------\n")
        return self

    def set_periodic_bc(self, primitive_cell):
        """Set periodic boundary conditions.
        The function sets the periodic boundary conditions by creating an object of the class CyclicTopology.
        Parameters
        ----------
        primitive_cell : list
            list of vectors defining a primitive cell
        """
        if list(primitive_cell):
            self.ct = CyclicTopology(primitive_cell,
                                     list(self.atom_list_eq.keys()),
                                     list(self.atom_list_eq.values()),
                                     self._nn_distance)
        else:
            self.ct = None

    def diagonalize(self):
        """Diagonalize the Hamiltonian matrix for the finite isolated system
        (without periodic boundary conditions)
        Returns
        -------
        vals : numpy.ndarray
            Eigenvalues
        vects : numpy.ndarray
            Eigenvectors
        """
        if self.compute_overlap:
            vals, vects = scipy.linalg.eigh(self.h_matrix, self.ov_matrix)
        else:
            vals, vects = torch.linalg.eigh(self.h_matrix, self.ov_matrix)
        vals = np.real(vals)
        ind = np.argsort(vals)

        return vals[ind], vects[:, ind]

    def diagonalize_periodic_bc(self, k_vector):
        """Diagonalize the Hamiltonian matrix with the periodic boundary conditions
        for a certain value of the wave vector k_vector
        Parameters
        ----------
        k_vector : numpy.ndarray
            wave vector
        Returns
        -------
        vals : numpy.ndarray
            Eigenvalues
        vects : numpy.ndarray
            Eigenvectors
        """

        k_vector = list(k_vector)

        # reset previous wave vector if any
        if k_vector != self.k_vector:
            self._reset_periodic_bc()
            self.k_vector = torch.tensor(k_vector)
            self._compute_h_matrix_bc_factor()
            self._compute_h_matrix_bc_add(overlap=self.compute_overlap)

        if self.compute_overlap:

            # vals, vects = scipy.linalg.eigh((self.h_matrix_bc_factor * self.h_matrix + self.h_matrix_bc_add).detach().numpy(),
            #                                 (self.h_matrix_bc_factor * self.ov_matrix + self.ov_matrix_bc_add).detach()())
            # vals, vects = torch.tensor(vals), torch.tensor(vects)
            vals, vects = torch.linalg.eigh(torch.linalg.solve(self.h_matrix_bc_factor * self.ov_matrix + self.ov_matrix_bc_add, self.h_matrix_bc_factor * self.h_matrix + self.h_matrix_bc_add))

        else:
            vals, vects = torch.linalg.eigh(self.h_matrix_bc_factor * self.h_matrix + self.h_matrix_bc_add)

        # vals = np.real(vals)
        ind = torch.argsort(vals)

        return vals[ind], vects[:, ind]

    def _set_nn_distances(self, nn_dist):
        if nn_dist is not None:
            if isinstance(nn_dist, list):
                # logging.info('{} nearest neighbour interactions are taken into account.'.format(len(nn_dist)))
                # logging.info("\n---------------------------------\n")
                nn_dist.sort()
                self._nn_distance = nn_dist[-1]

                def int_radial_dep(coords):
                    """
                        Step-wise radial dependence function
                    """
                    norm_of_coords = torch.norm(coords)

                    ans = int(sum([norm_of_coords > item for item in nn_dist]) + 1)

                    if norm_of_coords > nn_dist[-1]:
                        return 100
                    else:
                        return ans

                self.int_radial_dependence = int_radial_dep
            else:
                self._nn_distance = nn_dist
        # else:
        #     logging.info('The first nearest-neighbour approximation is used.')
        #     logging.info("\n---------------------------------\n")

        return self._nn_distance

    def _ind2atom(self, ind):
        """
        Parameters
        ----------
        ind :

        Returns
        -------
        """

        return self.orbitals_dict[list(self.atom_list.keys())[ind]]

    def _get_me(self, atom1, atom2, l1, l2, coords=None, overlap=False):
        """Compute the matrix element <atom1, l1|H|l2, atom2>.
        The function is called in the member function initialize() and invokes the function
        me() from the module diatomic_matrix_element.
        Parameters
        ----------
        atom1 : int
            Atom index
        atom2 : int
            Atom index
        l1 : int
            Index of a localized basis function
        l2 : int
            Index of a localized basis function
        coords : numpy.ndarray
            Coordinates of radius vector pointing from one atom to another
            it may differ from the actual coordinates of atoms (Default value = None)
        overlap : bool
            A flag indicating that the overlap matrix element has to be computed
        Returns
        -------
        type float
            Inter-cites matrix element
        """

        # on site (pick right table of parameters for a certain atom)
        if atom1 == atom2 and coords is None:
            atom_obj = self._ind2atom(atom1)
            if l1 == l2:
                if overlap:
                    return 1.0
                else:
                    return torch.tensor(atom_obj.orbitals[l1]['energy'], dtype=torch.float64)
            else:
                return self._comp_so(atom_obj, l1, l2)

        # nearest neighbours (define bound type and atomic quantum numbers)
        if atom1 != atom2 or coords is not None:

            atom_kind1 = self._ind2atom(atom1)
            atom_kind2 = self._ind2atom(atom2)

            # compute radius vector pointing from one atom to another
            if coords is None:
                coords1 = list(self.atom_list.values())[atom1] - \
                          list(self.atom_list.values())[atom2]
            else:
                coords1 = coords.clone()


            norm = torch.norm(coords1)

            # if verbosity.VERBOSITY > 0:
            #
            #     coordinates = np.array2string(norm, precision=4) + " Ang between atoms " + \
            #                   self._ind2atom(atom1).title + " and " + self._ind2atom(atom2).title
            #
            #     if coordinates not in unique_distances:
            #         unique_distances.add(coordinates)

            # compute directional cosines
            if self.compute_angular:
                coords1 = coords1 / norm
            else:
                coords1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            # coords1 = coords1.detach()

            if self.int_radial_dependence is None:
                which_neighbour = ""
            else:
                which_neighbour = self.int_radial_dependence(norm)

            # adding radial dependence
            # if self.radial_dependence is None:
            #     factor = 1.0
            # else:
                # if self.allow_strain:
                #     eqCorrd = self.eqAtom_list[list(self.atom_list.keys())[atom1]] - \
                #               self.eqAtom_list[list(self.atom_list.keys())[atom2]]
                #     factor = self.radial_dependence(norm, torch.norm(eqCorrd))
                # else:# determine type of bonds
                # atoms = sorted([item.upper() for item in [atom_kind1.title, atom_kind2.title]])
                # atoms = atoms[0] + '_' + atoms[1]

                # quantum numbers for the first atom
                # n1 = atom_kind1.orbitals[l1]['n']
                # ll1 = atom_kind1.orbitals[l1]['l']
                # m1 = atom_kind1.orbitals[l1]['m']
                # s1 = atom_kind1.orbitals[l1]['s']
                #
                # # quantum numbers for the second atom
                # n2 = atom_kind2.orbitals[l2]['n']
                # ll2 = atom_kind2.orbitals[l2]['l']
                # m2 = atom_kind2.orbitals[l2]['m']
                # s2 = atom_kind2.orbitals[l2]['s']
                #
                # if s1 == s2:
                #
                #     L = coords1[0]
                #     M = coords1[1]
                #     N = coords1[2]
                #
                #     gamma = torch.atan2(L, M)
                #
                #     if ll1 > ll2:
                #         code = [n2, n1]
                #     elif ll1 == ll2:
                #         code = [min(n1, n2), max(n1, n2)]
                #     else:
                #         code = [n1, n2]
                #
                #     for j, item in enumerate(code):
                #         if item == 0:
                #             code[j] = ""
                #         else:
                #             code[j] = str(item)

                    # l_min = min(ll1, ll2)
                    # l_max = max(ll1, ll2)

                # param = get_bl_param(atoms, which_neighbour, mode='pow')
                    # print("##########################################")
                    # print("neighbor: {}, get param {}".format(which_neighbour, param))
                    # print("##########################################")
                # else:
                #     param=0

                # if atom1 != atom2:
                #     radial_dep = self.radial_dependence
                #     # test = gradcheck(self.radial_dependence, (self._bond_length[atom1][atom2], norm, param))
                #     # print(test)
                #     # factor = self.radial_dependence(get_bl_param(atoms, which_neighbour, mode='bl'), norm, param)             # add
                # else:
                #     radial_dep = None

            if atom1 != atom2:
                radial_dep = self.radial_dependence
                # test = gradcheck(self.radial_dependence, (self._bond_length[atom1][atom2], norm, param))
                # print(test)
                # factor = self.radial_dependence(get_bl_param(atoms, which_neighbour, mode='bl'), norm, param)             # add
            else:
                radial_dep = None


            # print(coords1)
            # test = gradcheck(me, (atom_kind1, l1, atom_kind2, l2, coords1.detach().clone().requires_grad_(), which_neighbour,
            #           overlap))
            # print("test: ")
            # print(test)
            # return me(atom_kind1, l1, atom_kind2, l2, coords1, which_neighbour,
            #           overlap=overlap, radial_dep=radial_dep) * factor
            return me(atom_kind1, l1, atom_kind2, l2, coords1, norm, which_neighbour,
                      overlap=overlap, radial_dep=radial_dep)

    def _comp_so(self, atom, ind1, ind2):
        """
        Parameters
        ----------
        atom : Atom

        ind1 :

        ind2 :

        Returns
        -------
        type float
            Spin-orbit coupling energy
        """

        type1 = atom.orbitals[ind1]['title']
        type2 = atom.orbitals[ind2]['title']

        # quantum numbers
        l1 = atom.orbitals[ind1]['l']
        s1 = atom.orbitals[ind1]['s']
        l2 = atom.orbitals[ind2]['l']
        s2 = atom.orbitals[ind2]['s']

        if l1 == 1 and l2 == 1:

            if type1 == 'px' and type2 == 'py' and s1 == 0 and s2 == 0:
                return -1j * self.so_coupling / 3

            elif type1 == 'px' and type2 == 'pz' and s1 == 0 and s2 == 1:
                return self.so_coupling / 3

            elif type1 == 'py' and type2 == 'pz' and s1 == 0 and s2 == 1:
                return -1j * self.so_coupling / 3

            elif type1 == 'pz' and type2 == 'px' and s1 == 0 and s2 == 1:
                return -self.so_coupling / 3

            elif type1 == 'pz' and type2 == 'py' and s1 == 0 and s2 == 1:
                return 1j * self.so_coupling / 3

            elif type1 == 'px' and type2 == 'py' and s1 == 1 and s2 == 1:
                return 1j * self.so_coupling / 3

            elif type1 == 'py' and type2 == 'px' and s1 == 0 and s2 == 0:
                return 1j * self.so_coupling / 3

            elif type1 == 'pz' and type2 == 'px' and s1 == 1 and s2 == 0:
                return self.so_coupling / 3

            elif type1 == 'pz' and type2 == 'py' and s1 == 1 and s2 == 0:
                return 1j * self.so_coupling / 3

            elif type1 == 'px' and type2 == 'pz' and s1 == 1 and s2 == 0:
                return -self.so_coupling / 3

            elif type1 == 'py' and type2 == 'pz' and s1 == 1 and s2 == 0:
                return -1j * self.so_coupling / 3

            elif type1 == 'py' and type2 == 'px' and s1 == 1 and s2 == 1:
                return -1j * self.so_coupling / 3
            else:
                return 0
        else:
            return 0

    def _reset_periodic_bc(self):
        """Reset the matrices determining periodic boundary conditions to their default state
        :return:
        Parameters
        ----------
        Returns
        -------
        """

        self.h_matrix_bc_add = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)
        self.ov_matrix_bc_add = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)
        self.h_matrix_bc_factor = torch.ones((self.basis_size, self.basis_size), dtype=self.dtype)
        self.k_vector = None

    def _compute_h_matrix_bc_factor(self):
        """Compute the exponential Bloch factors needed when the periodic boundary conditions are applied."""

        for j1 in range(self.num_of_nodes):

            list_of_neighbours = self.get_neighbours(j1)

            for j2 in list_of_neighbours:
                if j1 != j2:
                    coords = list(self.atom_list.values())[j1] - \
                             list(self.atom_list.values())[j2]
                    phase = torch.exp(1j * torch.dot(self.k_vector, coords))

                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[j2]].num_of_orbitals):
                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)], )
                            ind2 = self.qn2ind([('atoms', j2), ('l', l2)], )

                            self.h_matrix_bc_factor[ind1, ind2] = phase
                            # self.h_matrix[ind2, ind1] = self.h_matrix[ind1, ind2]

    def _compute_h_matrix_bc_add(self, split_the_leads=False, overlap=False):
        """Compute additive Bloch exponentials needed to specify pbc
        Parameters
        ----------
        split_the_leads :
             (Default value = False)
        overlap :
             (Default value = False)
        Returns
        -------
        """

        two_leads = False

        if self.ct is not None:
            if self.ct.pcv.shape[0] == 1:
                two_leads = True

            if split_the_leads:
                if two_leads:
                    flag = None
                else:
                    flag = 'L'
            # loop through all interfacial atoms
            for j1 in self.ct.interfacial_atoms_ind:

                list_of_neighbours = self.ct.get_neighbours(list(self.atom_list_np.values())[j1])

                for j2 in list_of_neighbours:

                    coords = list(self.atom_list.values())[j1] - \
                             torch.tensor(list(self.ct.virtual_and_interfacial_atoms.values())[j2])

                    if split_the_leads and two_leads:
                        flag = self.ct.atom_classifier(list(self.ct.virtual_and_interfacial_atoms.values())[j2],
                                                       self.ct.pcv[0])

                    phase = torch.exp(1j * torch.dot(self.k_vector, coords))

                    ind = int(list(self.ct.virtual_and_interfacial_atoms.keys())[j2].split('_')[2])

                    for l1 in range(self.orbitals_dict[list(self.atom_list.keys())[j1]].num_of_orbitals):
                        for l2 in range(self.orbitals_dict[list(self.atom_list.keys())[ind]].num_of_orbitals):

                            ind1 = self.qn2ind([('atoms', j1), ('l', l1)])
                            ind2 = self.qn2ind([('atoms', ind), ('l', l2)])

                            if split_the_leads:
                                if flag == 'R':
                                    self.h_matrix_left_lead[ind1, ind2] += phase * \
                                                                           self._get_me(j1, ind, l1, l2, coords=coords)
                                    if self.compute_overlap:
                                        self.ov_matrix_left_lead[ind1, ind2] += phase * \
                                                                               self._get_me(j1, ind, l1, l2, coords=coords, overlap=True)
                                elif flag == 'L':
                                    self.h_matrix_right_lead[ind1, ind2] += phase * \
                                                                            self._get_me(j1, ind, l1, l2, coords=coords)
                                    if self.compute_overlap:
                                        self.ov_matrix_right_lead[ind1, ind2] += phase * \
                                                                               self._get_me(j1, ind, l1, l2, coords=coords, overlap=True)
                                else:
                                    raise ValueError("Wrong flag value")
                            else:
                                self.h_matrix_bc_add[ind1, ind2] += phase * \
                                                                    self._get_me(j1, ind, l1, l2, coords)
                                if overlap:
                                    self.ov_matrix_bc_add[ind1, ind2] += phase * \
                                                                         self._get_me(j1, ind, l1, l2,
                                                                                      coords=coords, overlap=True)

    def get_hamiltonians(self):
        """Return a list of Hamiltonian matrices. For 1D systems, the list is [Hl, Hc, Hr],
        where Hc is the Hamiltonian describing interactions between atoms within a unit cell,
        Hl and Hr are Hamiltonians describing couplings between atoms in the unit cell
        and atoms in the left and right adjacent unit cells.
        Parameters
        ----------
        Returns
        -------
        list
            list of Hamiltonians
        """

        self.k_vector = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

        self.h_matrix_left_lead = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)
        self.h_matrix_right_lead = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)
        if self.compute_overlap:
            self.ov_matrix_left_lead = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)
            self.ov_matrix_right_lead = torch.zeros((self.basis_size, self.basis_size), dtype=self.dtype)

        self._compute_h_matrix_bc_add(split_the_leads=True)
        self.k_vector = None

        if self.compute_overlap:
            return self.h_matrix_left_lead.T, self.h_matrix, self.h_matrix_right_lead.T, \
                   self.ov_matrix_left_lead.T, self.ov_matrix, self.ov_matrix_right_lead.T
        else:
            return self.h_matrix_left_lead.T, self.h_matrix, self.h_matrix_right_lead.T

    def get_site_coordinates(self):
        """Return coordinates of atoms.
        Parameters
        ----------
        Returns
        -------
        numpy.ndarray
            atomic coordinates
        """
        return torch.stack(self._coords)

    def get_hamiltonians_block_tridiagonal(self, left=-1, right=-1, optimized=True):
        """
        Parameters
        ----------
        left :
             (Default value = None)
             (Default value = None)
        right :
             (Default value = None)
        optimized :
             (Default value = True)
        Returns
        -------
        """
        if self.compute_overlap:
            hl, h0, hr, sl, s0, sr = self.get_hamiltonians()
        else:
            hl, h0, hr = self.get_hamiltonians()

        if left == -1 and right == -1:
            h_r_h = find_nonzero_lines(hr, 'bottom')
            h_r_v = find_nonzero_lines(hr[-h_r_h:, :], 'left')
            h_l_h = find_nonzero_lines(hl, 'top')
            h_l_v = find_nonzero_lines(hl[:h_l_h, :], 'right')
            left = max(h_l_h, h_r_v)
            right = max(h_r_h, h_l_v)

        with torch.no_grad():
            if optimized:
                subblocks = split_into_subblocks_optimized(h0, left=left, right=right)
            else:
                subblocks = split_into_subblocks(h0, left, right)

        h01, hl1, hr1 = cut_in_blocks(h0, subblocks)

        if self.compute_overlap:
            s01, sl1, sr1 = cut_in_blocks(s0, subblocks)

        if left is not None and right is not None:
            hl1.append(hl[:left, -right:])
            hr1.append(hr[-right:, :left])

            if self.compute_overlap:
                sl1.append(sl[:left, -right:])
                sr1.append(sr[-right:, :left])

        if self.compute_overlap:
            return hl1, h01, hr1, sl1, s01, sr1, subblocks
        else:
            return hl1, h01, hr1, subblocks