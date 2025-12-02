import inspect
class ChemicalReaction:
  #  Chemical Reaction Attributes:
  #  stoich_in = dictionary of reactants
  #  stoich_out = dictionary of products
  #  stoich_net = Net stoichiometry vector(list) which is (stoich_out-stoich_in)
  #  rate_law = object indicating rate law

  # Constructor
  def __init__(self, stoich_in, stoich_out, rate_law="massAction", params=None, 
             concentrations=None):
    self.stoich_in = stoich_in
    self.stoich_out = stoich_out
    self.rate_law = rate_law

    #self.stoich_net = [
    #  out - inn for inn, out in zip(stoich_in, stoich_out)
    #  ]
    self.params = params if params is not None else {}

    # Concentrations array
    self.concentrations = concentrations
  
  # Calculate reaction rate based on rate law
  def calculate_rate(self):
    # assumed list/array of concentrations
    conc = self.concentrations
    p = self.params 
      
    if self.rate_law=="massAction":
      rate = p["k"]
      for i in p["reactants"]:
        rate *= conc[i]
      return rate
      
    if self.rate_law=="michaelisMenten":
      # requires: p["Vmax"], p["Km"], p["S"] (substrate index)
      S = conc[p["S"]]
      return p["Vmax"] * S / (p["Km"] + S)
      
    if self.rate_law=="hillA":
      # Hill activation
      # requires: p["K"], p["n"], p["k"], p["S"] (substrate index)
      S = conc[p["S"]]
      return p["k"] * (S**p["n"]) / (p["K"]**p["n"] + S**p["n"])
      
    if self.rate_law=="hillR":
      # Hill repression
      # requires: p["K"], p["n"], p["k"], p["S"] (substrate index)
      S = conc[p["S"]]
      return p["k"] * (p["K"]**p["n"]) / (p["K"]**p["n"] + S**p["n"])
    else: #case where rate_law is a function
      return self.rate_law

  # Builder method to turn dictionaries into vectors
  def toVectors(self, species_order):

    # Precompute index map for speed
    index_map = {s: i for i, s in enumerate(species_order)}

    def dictToVector(stoich_dict):
        vec = [0] * len(species_order)
        for species, coeff in stoich_dict.items():
            vec[index_map[species]] = coeff
        return vec

    # Only convert if they are dicts
    if isinstance(self.stoich_in, dict):
        self.stoich_in_vec = dictToVector(self.stoich_in)
    else:
        self.stoich_in_vec = self.stoich_in

    if isinstance(self.stoich_out, dict):
        self.stoich_out_vec = dictToVector(self.stoich_out)
    else:
        self.stoich_out_vec = self.stoich_out

    # Compute net vector
    self.stoich_net_vec = [
        out - inn for inn, out in zip(self.stoich_in_vec, self.stoich_out_vec)
    ]
      
    
    
    
  
  
