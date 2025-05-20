Another weird design is the combination of Architecture and Weighting. Weighting does not explicitly has the attributes like `rep_grad`
or `get_share_params` but we use it explicitly in the class.
* For that reason, I cannot have a MGDA instance in PSMGD as I have to manually set all these implicit features myself. 

* weighting is dependent on architecture as it should be. but this dependence is reflected in a double inherited class `<Architecture, Weighting>`; not explicitly.