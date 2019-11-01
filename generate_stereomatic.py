from scipy import linalg
from scipy.special import expit
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import json

try:
    with open("/Users/hyu/Documents/Projects/Jaguar_stereomatic/Stereomatic_Tree/new_bond_data.json", 'r') as fh:
        new_bond_data_dict = json.load(fh)
except:
    with open("new_bond_data.json", 'r') as fh:
        new_bond_data_dict = json.load(fh)

def sigmoid_function(c, s, shift, x):
    """
    sigmoid function

    c: center of the function
    s: shape of the function
    shift: shift up by shift
    x: varible

    return f(x) + shift
    """

    return expit(s * (c -x)) + shift

def box_function(width_left, width_right, shift, sigma, x):
    """
    a function equals one at (shift - width_left, shift + width_right)

    sigma: how fast it goes from 1 to zero
    """

    prefactor = 2.0 * 0.25
    left = erf( (1.0/width_left * x + 1.0/width_left * shift  + 1.0) / ( sigma * math.sqrt(2.0)) )
    right = erf( (1.0/width_right * x + 1.0/width_right * shift  - 1.0) / ( sigma * math.sqrt(2.0)) )

    return prefactor * (left - right)

def plot_stereomatic(name, database):
    """
    A function to generate new LAD plot

    :type  name: str
    :param name: the name of the plot

    :type  database: dict
    :param database: the database that store all information of a given atoms pair
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(16,12)
    x = [ xx * 0.0005 for xx in range(-10000,10000)]
    y = []
    for xx in x:
        y.append(stereomatic_descriptor(name, xx, database))

    ax.scatter(x, y, label='%s'%name, color='k', s=10)
    plt.xlim((0,4))
    plt.ylim((0,5))
    x_new_ticks = np.linspace(0,4,21)
    y_new_ticks = np.linspace(0,5,11)
    plt.xticks(x_new_ticks, fontsize=10)
    plt.yticks(y_new_ticks, fontsize=10)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('y', fontsize=10)
    plt.title('stereometic Function', fontsize=10, y=1.05)
    plt.legend(loc='best', fontsize=10)
    # plt.show()
    plt.savefig('%s.png'%name)
    plt.close(fig)

def stereomatic_descriptor(atoms_pair, x, database=new_bond_data_dict):
    """
    A function to return the stereomatic value for a type of atom pairs at distance of x

    :type  atoms_pair: string
    :param atoms_pair: the name of atoms pair eg: C_C, C_O

    :type  x: float
    :param x: the bond distance between two atoms

    :type  database: dict
    :param database: the database that store all information of a given atoms pair

    return the new CN value
    """
    
    try:
        data_to_import = database[atoms_pair]
    except KeyError:
        return 0
    bond_length_average, bond_orders, bond_range_list = prepare_data(data_to_import)
    
    ret_value = 0
    for i in range(len(bond_length_average)):
        ret_value += sigmoid_function(bond_length_average[i], 100, bond_orders[i]-1, x) * box_function(bond_range_list[i][0], bond_range_list[i][1], -bond_length_average[i], 0.0001, x)
    
    return ret_value

def prepare_data(to_import_data):

    """
    :type  to_import_data: list
    :param to_import_data: bonding information of a certain type of atom pairs

    return bond_length_average, bond_orders, bond_range_list
    """

    if(len(to_import_data) == 3):
        triple_bond, double_bond, single_bond = to_import_data[:]
        single_bond_min, single_bond_max = round(single_bond['bond_min'], 2), round(single_bond['bond_max'],2)
        double_bond_min, double_bond_max = round(double_bond['bond_min'], 2), round(double_bond['bond_max'],2)
        triple_bond_min, triple_bond_max = round(triple_bond['bond_min'], 2), round(triple_bond['bond_max'],2)
        
        triple_bond_center = (triple_bond_max + double_bond_min)/2
        double_bond_center = (double_bond_max + single_bond_min)/2
        single_bond_center = single_bond_max + 0.1
        bond_length_average = [triple_bond_center, double_bond_center, single_bond_center]
        bond_orders = [3, 2, 1]

        triple_bond_left_width, triple_bond_right_width = triple_bond_center, (double_bond_max + double_bond_min)/2 - triple_bond_center
        double_bond_left_width, double_bond_right_width = double_bond_center - (double_bond_max + double_bond_min)/2, (single_bond_max + single_bond_min)/2 - double_bond_center
        single_bond_left_width, single_bond_right_width = single_bond_center - (single_bond_max + single_bond_min)/2, 0.1
        bond_range_list = [(triple_bond_left_width, triple_bond_right_width),(double_bond_left_width, double_bond_right_width),(single_bond_left_width, single_bond_right_width)]

    elif(len(to_import_data) == 2):
        double_bond, single_bond = to_import_data[:]
        single_bond_min, single_bond_max = round(single_bond['bond_min'], 2), round(single_bond['bond_max'],2)
        double_bond_min, double_bond_max = round(double_bond['bond_min'], 2), round(double_bond['bond_max'],2)

        double_bond_center = (double_bond_max + single_bond_min)/2
        single_bond_center = single_bond_max + 0.1
        bond_length_average = [double_bond_center, single_bond_center]
        bond_orders = [2, 1]
        double_bond_left_width, double_bond_right_width = double_bond_center, (single_bond_max + single_bond_min)/2 - double_bond_center
        single_bond_left_width, single_bond_right_width = single_bond_center - (single_bond_max + single_bond_min)/2, 0.1
        bond_range_list = [(double_bond_left_width, double_bond_right_width),(single_bond_left_width, single_bond_right_width)]

    elif(len(to_import_data) == 1):
        single_bond = to_import_data[0]
        single_bond_min, single_bond_max = round(single_bond['bond_min'], 2), round(single_bond['bond_max'],2)
        single_bond_center = single_bond_max + 0.1
        bond_length_average = [single_bond_center]
        bond_orders = [1]
        single_bond_left_width, single_bond_right_width = single_bond_center, 0.1
        bond_range_list = [(single_bond_left_width, single_bond_right_width)]

    return bond_length_average, bond_orders, bond_range_list

def main():

    with open('new_bond_data.json', 'r') as fh:
        data = json.load(fh)
        for key in data.keys():
            print('working on %s'%(key))
            plot_stereomatic(key, data)
            # combined_function([1.275, 1.4, 1.70], [3,2,1], [[1.275, 1.35-1.275],[1.40-1.35, 1.52-1.40],[1.70-1.52, 1.8-1.7]], 'C_C')

if __name__ == "__main__":
    main()