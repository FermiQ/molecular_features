"""
Information and Functions using only python. 

Note: All functions are supposed to work out of the box without any dependencies, i.e. do not depend on each other.

@author: Patrick Reiser, 
"""

def element_list_to_value(elem_list, replace_dict):
    """
    Translate list of atoms as string to a list of values according to a dictionary.
    This is recursive and should also work for nested lists.
    
    Args:
        elem_list (list): List of elements like ['H','C','O']
        replace_dict (dict): python dictionary of atom label and value e.g. {'H':1,...}
    
    Returns:
        list of values for each atom.
    """
    if isinstance(elem_list,str):
        return replace_dict[elem_list]
    elif isinstance(elem_list,list):
        outlist = []
        for i in range(len(elem_list)):
            if isinstance(elem_list[i],list):
                outlist.append(element_list_to_value(elem_list[i],replace_dict))
            elif isinstance(elem_list[i],str):
                outlist.append(replace_dict[elem_list[i]])
        return outlist
    

def element_list_to_proton(elem_list):
    pass

