import numpy as np # https://numpy.org/

inputs = np.array([1.0, 0.7, 0.0]) # suncano,toplo,pada kisa
weights = np.random.uniform(-1, 1, 3)  # 3 random broja između -1 i 1 , biraju se njihove tezine
bias = np.random.uniform(-1, 1) # bias pomera prag 

output = np.dot(inputs, weights) + bias # optimizovano u smislu ne potrebna petlja jer .dot radi isto sto i for petlja

print("weights:", weights)
print("bias:", bias)
print("Output:", output)
print("Odluka:", "treba voziti bicikl" if output > 0 else "ne treba voziti bicikl")