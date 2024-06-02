""" Variation operators "move" the candidate solutions across S
The module `gpol.variators` contains some relevant variation operators
(variators) used to "move" the candidate solutions across the solve
space in between the iterations. Given the fact this library supports
different types of iterative solve algorithms (ISAs), the module
contains a collection of variators suitable for every kind of ISA
implemented in this library.
"""

import copy
import random

import torch
import numpy as np

from gpolnel.utils.inductive_programming import tanh1, lf1, add2, sub2, mul2, _execute_tree, get_subtree, _Function


# +++++++++++++++++++++++++++ Inductive Programming
def swap_xo(p1, p2):
    """ Implements the swap crossover

    The swap crossover (a.k.a. standard GP's crossover) consists of
    exchanging (swapping) parents' two randomly selected subtrees.

    Parameters
    ----------
    p1 : list
        Representation of the first parent.
    p2 : list
        Representation of the second parent.

    Returns
    -------
    list, list
        Tuple of two lists, each representing an offspring obtained
        from swapping two randomly selected sub-trees in the parents.
    """
    # Selects start and end indexes of the first parent's subtree
    p1_start, p1_end = get_subtree(p1)
    # Selects start and end indexes of the second parent's subtree
    p2_start, p2_end = get_subtree(p2)

    return p1[:p1_start] + p2[p2_start:p2_end] + p1[p1_end:], p2[:p2_start] + p1[p1_start:p1_end] + p2[p2_end:]


def prm_point_mtn(sspace, prob):
    """ Implements the point mutation

    This function is used to provide the point_mtn (inner function)
    with the necessary environment (the outer scope) - the solve
    space (𝑆), necessary to for the mutation function to access the
    set of terminals and functions.

    Parameters
    ----------
    sspace : dict
        The formal definition of the 𝑆. For GP-based algorithms, it
        contains the set of constants, functions and terminals used
        to perform point mutation.
    prob : float
        Probability of mutating one node in the representation.

    Returns
    -------
    point_mtn : function
        The function which implements the point mutation for GP.
    """
    def point_mtn(repr_):
        """ Implements the point mutation

        The point mutation randomly replaces some randomly selected
        nodes from the individual's representation. The terminals are
        replaced by other terminals and functions are replaced by other
        functions with the same arity.

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from replacing a randomly selected
            subtree in the parent by a random tree.
        """
        # Creates a copy of parent's representation
        repr_copy = copy.deepcopy(repr_)
        # Performs point replacement
        for i, node in enumerate(repr_copy):
            if random.random() < prob:
                if isinstance(node, _Function):
                    # Finds a valid replacement with same arity
                    node_ = sspace['function_set'][random.randint(0, len(sspace['function_set'])-1)]
                    while node.arity != node_.arity:
                        node_ = sspace['function_set'][random.randint(0, len(sspace['function_set']) - 1)]
                    # Performs the replacement, once a valid function was found
                    repr_copy[i] = node_
                else:
                    if random.random() < sspace['p_constants']:
                        repr_copy[i] = sspace['constant_set'][random.randint(0, len(sspace['constant_set']) - 1)]
                    else:
                        repr_copy[i] = random.randint(0, sspace['n_dims'] - 1)

        return repr_copy

    return point_mtn


def prm_subtree_mtn(initializer):
    """ Implements the the subtree mutation

    This function is used to provide the subtree_mtn (inner function) with
    the necessary environment (the outer scope) - the random trees'
    generator, required to perform the mutation itself.

    Parameters
    ----------
    initializer : function
        Parametrized initialization function to generate random trees.

    Returns
    -------
    subtree_mtn : function
        The function which implements the sub-tree mutation for GP.
    """
    def subtree_mtn(repr_):
        """ Implements the the subtree mutation

        The subtree mutation (a.k.a. standard GP's mutation) replaces a
        randomly selected subtree of the parent individual by a completely
        random tree.

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from replacing a randomly selected
            subtree in the parent by a random tree.
        """
        # Generates a random tree
        random_tree = initializer()
        # Calls swap crossover to swap repr_ with random_tree
        return swap_xo(repr_, random_tree)[0]

    return subtree_mtn
