import numpy as np
# ovo je kao pojedin neuron - single neuron - ja sam mu dao input sa kojim radi zato je output ~0.75

inputs = [1.0,0.7,0.0] # znacenje da li je suncano/toplo/kisa pada
weights = [0.8,0.5,-1.5] # njihove tezine 
bias = -0.4 # raspolozenje
output = bias
for x,w in zip(inputs,weights): # for ~ .dot
    output = output + x*w

print(output)
print(np.dot(inputs,weights)+bias)