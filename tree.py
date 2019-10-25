import random
import sys
import itertools
import numpy as np
from schrodinger.infra import mm
from schrodinger.structure import StructureReader
from generate_stereomatic_step1 import stereomatic_descriptor

def get_bond_order(st, at1, at2):
    
    if at1.atomic_number < at2.atomic_number:
        atoms_pair = '%s_%s'%(at1.element, at2.element)
    else:
        atoms_pair = '%s_%s'%(at2.element, at1.element)
    distance = st.measure(at1, at2)
    
    return stereomatic_descriptor(atoms_pair, distance)


class Node:

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

    if root is None:
        return
    if k == 0:
        # print(root)
        array.append(root)
    else:
        for child in root.children:
            get_nodes_by_level(child, k-1, array)


def get_stereomatic_desc(st, tree):

    origin = tree.idx
    # print(tree)
    # for at in st.atom:
    for at in sort_atom_by_atomic(st):
        value = get_bond_order(st, st.atom[origin], at)
        if value > 0.5:
            if at.index not in tree.visited:
#                print('adding ', at.index, at.element)
                tree.insert(value, at.index, at.element, at.partial_charge, origin, tree.visited)

    for child in tree.children:
#        print(f'child {child.idx} of {origin} \n', child)
        get_stereomatic_desc(st, child)

def sum_bonded_atomic(atom):

    sum = 0
    for bonded_atom in atom.bonded_atoms:
        sum += bonded_atom.atomic_number

    return sum

def sort_atom_by_atomic(st):

    retArr = [at for at in st.atom]
    
    return sorted(retArr, key=lambda atom: (atom.atomic_number, sum_bonded_atomic(atom)))

def calculate_overlap(arr1, arr2):
    """
    first ensure two array has the same length after packing

    arr1: [node1, node2, node3, etc]
    arr2: [node1, node2, node3, etc]
    """
    sum = 0
    for node1, node2 in zip(arr1, arr2):
        sum += (node1.value - node2.value) * (node1.value - node2.value)

    return sum

def generate_tree(maefile, origin):

    print(f'Processing {maefile}')
    st = next(StructureReader(maefile))
    print(f'Using atom {origin} ({st.atom[origin].element}) as origin.\n')
    root = Node(None, origin, st.atom[origin].element, st.atom[origin].partial_charge, None, set())
    get_stereomatic_desc(st, root)

    return root

def print_node(node_arr):
    """
    helper function for debug 
    """

    for node in node_arr:
        print(node)

def pack_array(arr1, arr2):
    """
    A function to added node with zero values to the corresponding array
    """
    # for item1, item2 in itertools.zip_longest(arr1, arr2, fillvalue=Node(0, origin, st.atom[origin].element, None, set())): 

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

    maefile1 = sys.argv[1]
    origin1 = int(sys.argv[2])
    maefile2 = sys.argv[3]
    origin2 = int(sys.argv[4])
    level = int(sys.argv[5])

    tree1 = generate_tree(maefile1, origin1)
    tree2 = generate_tree(maefile2, origin2)

    # Index of atom origin of stereomatic network
    retArr1, retArr2 = [], []
    get_nodes_by_level(tree1, level, retArr1)
    get_nodes_by_level(tree2, level, retArr2)

    print('arr1: -------------')
    print_node(retArr1)
    print('arr2: -------------')
    print_node(retArr2)

    arr1_packed, arr2_packed = pack_array(retArr1, retArr2)

    # print('arr1_packed: ', print_node(arr1_packed))
    # print('arr2_packed: ', print_node(arr2_packed))

    difference = calculate_overlap(arr1_packed, arr2_packed)
    overlap = np.exp(-difference)
    print(overlap)
    # print(tree1)
    # print(tree1.children)
    # child = root.children[0]
    # print(child)
#    grandchild1 = child.children[0]
#    print(grandchild1)
#    grandchild2 = child.children[1]
#    print(grandchild2)

if __name__ == "__main__":
    main()

