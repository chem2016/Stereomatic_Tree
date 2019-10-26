import random
import sys
import itertools
import numpy as np
import time
import argparse
import tracemalloc
from schrodinger.infra import mm
from schrodinger.structure import StructureReader
from generate_stereomatic_step1 import stereomatic_descriptor

def parse_args():
    """
    parse commandline arguments
    """

    parser = argparse.ArgumentParser(
        description="calculate stereomatic overlap between two molecules give specific atoms and level",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'st_1',
        type=str,
        help='name of the molecule1 st file'
    )

    parser.add_argument(
        'st_2',
        type=str,
        help='name of the molecule2 st file'
    )

    parser.add_argument(
        '-atom_of_interest',
        dest='atom_pair',
        metavar='<Xi>',
        default=(1, 1),
        nargs=2,
        type=int,
        help='specify two atom indexes for comparison'
    )

    parser.add_argument(
        '-level',
        dest='level',
        type=int,
        help='the level of the overlap one wants to calculate'
    )

    parser.add_argument(
        '-debug',
        default=False,
        dest='debug',
        action='store_true',
        help='debug option to print all the descriptors'  
    )

    return parser.parse_args()

def get_bond_order(st, at1, at2):
    """
    A function to get bond order base on build-in stereomatic desciptor

    :type  st: schrodinger.structure
    :param st: structure of a molecule

    :type  at1: schrodinger.structure.atom
    :param at1: atom1 instance

    :type  at2: schrodinger.structure.atom
    :param at2: atom2 instance

    return: a float number that represent the bond order between at1 and at2
    """

    if at1.atomic_number < at2.atomic_number:
        atoms_pair = '%s_%s'%(at1.element, at2.element)
    else:
        atoms_pair = '%s_%s'%(at2.element, at1.element)
    distance = st.measure(at1, at2)
    
    return stereomatic_descriptor(atoms_pair, distance)


class Node:
    """
    A data structure to convert a molecule into a tree structure
    """

    def __init__(self, value, idx, element, charge, parent_idx, visited):
        self.value = value
        self.idx = idx
        self.element = element
        self.charge = charge
        self.parent_idx = parent_idx
        self.visited = list(visited)
        self.visited.append(idx)
        self.children = []

    def __str__(self):
        msg = f'idx = {self.idx}\n'
        msg += f'element = {self.element}\n'
        msg += f'charge = {self.charge}\n'
        msg += f'parent_idx = {self.parent_idx}\n'
        msg += f'value = {self.value}\n'
        msg += f'visited = {self.visited}\n'
        msg += f'len(children) = {len(self.children)}\n'
        return msg

    def insert(self, value, idx, element, charge, parent_idx, visited):
        self.children.append(Node(value, idx, element, charge, parent_idx, visited))
    
def get_nodes_by_level(root, k, array):
    """
    A function to find all the nodes on its kth level

    :type  root: Node
    :param root: root node of a tree

    :type  k: int
    :param k: kth level of tree root

    :type  array: list
    :param array: a queue to take all the nodes on kth level
    """

    if root is None:
        return
    if k == 0:
        # print(root)
        array.append(root)
    else:
        for child in root.children:
            get_nodes_by_level(child, k-1, array)


def get_stereomatic_desc(st, tree, sorted_atoms):
    """
    A function to update all the nodes of a tree with stereomatic value

    :type  st: schrodinger.structure
    :param st: structure of a molecule

    :type  tree: Node
    :param tree: The tree structure of st
    """

    origin = tree.idx
    # print(tree)
    # for at in st.atom:
    for at in sorted_atoms:
        value = get_bond_order(st, st.atom[origin], at)
        if value > 0.5:
            if at.index not in tree.visited:
#                print('adding ', at.index, at.element)
                tree.insert(value, at.index, at.element, at.partial_charge, origin, tree.visited)

    for child in tree.children:
#        print(f'child {child.idx} of {origin} \n', child)
        get_stereomatic_desc(st, child, sorted_atoms)

def sum_bonded_atomic(atom):
    """
    A helper function to calculate the sum of atomic numbers of bonded atom
    """

    sum = 0
    for bonded_atom in atom.bonded_atoms:
        sum += bonded_atom.atomic_number

    return sum

def sort_atom_by_atomic(st):
    """
    A helper method to sort the atoms of a structure base on their atomic_number of itself and neighbour
    """
    retArr = [at for at in st.atom]
    
    return sorted(retArr, key=lambda atom: (atom.atomic_number, sum_bonded_atomic(atom)))

def calculate_overlap(arr1, arr2):
    """
    A function to calculate two arrays of nodes with same dimension

    arr1: [node1, node2, node3, etc]
    arr2: [node1, node2, node3, etc]
    """
    sum = 0
    for node1, node2 in zip(arr1, arr2):
        sum += (node1.value - node2.value) * (node1.value - node2.value)

    return sum

def generate_tree(maefile, origin):
    """
    A helper function to generate tree structure from a given maefile and origin

    :type  maefile: str
    :param maefile: name of the maefile, endwith .mae

    :type  origin: int
    :param origin: atom of interest

    return tree structure
    """
    print(f'Processing {maefile}')
    st = next(StructureReader(maefile))
    sorted_atoms = sort_atom_by_atomic(st)
    print(f'Using atom {origin} ({st.atom[origin].element}) as origin.\n')
    root = Node(None, origin, st.atom[origin].element, st.atom[origin].partial_charge, None, set())
    get_stereomatic_desc(st, root, sorted_atoms)

    return root

def print_node(node_arr):
    """
    helper function for debug 
    """

    for node in node_arr:
        print(node)

def pack_array(arr1, arr2):
    """
    A function to pack two arrays with new Nodes to ensure same dimensitionality
    """
    # for item1, item2 in itertools.zip_longest(arr1, arr2, fillvalue=Node(0, origin, st.atom[origin].element, charge, None, set())): 

    retArr1 = []
    retArr2 = []
    dummy_parent_idx = 10000
    for node1, node2 in itertools.zip_longest(arr1, arr2):

        if node1 == None and node2 != None:
            retArr1.append(Node(0, dummy_parent_idx, node2.element, 0, None, set()))
            retArr2.append(node2)
        elif node1 != None and node2 == None:
            retArr1.append(node1)
            retArr2.append(Node(0, dummy_parent_idx, node1.element, 0, None, set()))
        elif node1.element == node2.element:
            retArr1.append(node1)
            retArr2.append(node2)
        elif node1.element != node2.element and node1 != None and node2 != None:
            retArr1.append(node1)
            retArr2.append(Node(0, node2.parent_idx, node1.element, 0, None, set()))
            retArr1.append(Node(0, node1.parent_idx, node2.element, 0, None, set()))
            retArr2.append(node2)

    return retArr1, retArr2

def main():

    args = parse_args()
    print("args: ", args)

    maefile1 = args.st_1
    origin1, origin2 = args.atom_pair
    maefile2 = args.st_2
    level = args.level

    tree1 = generate_tree(maefile1, origin1)
    tree2 = generate_tree(maefile2, origin2)

    # Index of atom origin of stereomatic network
    retArr1, retArr2 = [], []
    get_nodes_by_level(tree1, level, retArr1)
    get_nodes_by_level(tree2, level, retArr2)
    arr1_packed, arr2_packed = pack_array(retArr1, retArr2)
    difference = calculate_overlap(arr1_packed, arr2_packed)
    overlap = np.exp(-difference)
    print(overlap)

    if args.debug:
        print('nodes from tree1: --------------------')
        print_node(retArr1)
        print('nodes from tree2: --------------------')
        print_node(retArr2)

if __name__ == "__main__":
    start_time = time.time()
    tracemalloc.start()
    main()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    print("--- %s seconds ---" % (time.time() - start_time))


