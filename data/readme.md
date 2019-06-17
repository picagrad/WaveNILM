Place measurement file here in a python pickle format (so that it can be loaded with pickle.load)
Some example files:

Active Power Input & Output:
https://researchdata.sfu.ca/pydio_public/in_p_out_p

Current; Active, Reactive, and Apparent Power, as input, Active Power as output:
https://researchdata.sfu.ca/pydio_public/in_all_4_out_p

In the pickle file, there must be a 3-D numpy array so that the first axis represents different times; the 2nd axis represetns different measurement (such as aggregate, fridge, microwave, etc); and the third axis represents different feaures (such as current, power, voltage etc.) 
