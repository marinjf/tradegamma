from typing import Any, List
import pickle

def update_dict_of_list(key:Any,value:Any,dictionnary:dict[Any,List])->dict:
    if key not in list(dictionnary.keys()): 
        dictionnary[key] = [value]
    else: dictionnary[key].append(value)
    return dictionnary

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


