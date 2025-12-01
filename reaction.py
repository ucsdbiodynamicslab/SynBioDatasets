import inspect

Class ChemicalReaction: 
  #  Chemical Reaction Attributes:
  #  stoich_out = vector(list) of products
  #  stoich_net = Net stoichiometry vector(list) which is (stoich_out-stoich_in)
  #  rate_law = object indicating rate law

  def _init_(self, stoich_in, stoich_out, rate_law="massAction"):
    self.stoich_in = stoich_in
    self.stoich_out = stoich_out
    self.rate_law = rate_law

    self.stoich_net = [
            out - inn for inn, out in zip(stoich_in, stoich_out)
        ]

    def calculate_rate(self):
      
      if self.rate_law=="massAction":
        
      if self.rate_law=="michaelisMenten":
        
      if self.rate_law=="hillA":
        
      if self.rate_law=="hillR":
        
      else: #case where rate_law is a function
        return rate_law

      
    
    
    
  
  
