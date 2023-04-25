import typing as t

import numpy as np


"""
This is just a regular dict type with a loose semi-determinate structure. These dicts will be the results 
when loading a "visual graph dataset". The metadata dict will contain things such as the name of the 
element it's index in the dataset, the path to the image file visualizing that particular element...

It also contains all of the additional metadata information contained within the JSON file that represents 
that element. The contents of that JSON file are not fixed and may dynamically contain all sorts of 
additional information. Some information it should always contain however is: The graph representation of 
the element as a dictionary & the target ground truth value associated with the element
"""
MetaDict = t.Dict[str, t.Union[str, int, dict]]


"""
This is just a regular dict type with a loose semi-determinate structure. This kind of dict is used to 
represent a graph internally in the program. There are some fields such dict absolutely has to have to 
qualify as a valid graph and some fields are optional which will be checked for at the various places 
where they will be used. 

Dicts of this type should not be nested, meaning they do not themselves contain other dicts as values. 
Rather, all the values are supposed to by either numpy arrays or lists of numeric values which represent 
the various tensors used to describe a graph.
"""
GraphDict = t.Dict[str, t.Union[np.ndarray, list]]
