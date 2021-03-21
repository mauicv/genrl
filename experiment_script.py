from examples.bip_walker_NEAT import neat_bipedal_walker
from examples.bip_walker_RES import bipedal_walker_RES
from examples.bip_walker_ADRES import bipedal_walker_ADRES
from time import time

print('----------------')
print()
start = time()
print('bipedal_walker_RES')
bipedal_walker_RES()
end = time()
print('time taken: ', end - start)
print()
print('----------------')
print()
start = time()
print('bipedal_walker_ADRES')
bipedal_walker_ADRES()
end = time()
print('time taken: ', end - start)
print()
print('----------------')
print()
start = time()
print('neat_bipedal_walker')
neat_bipedal_walker()
end = time()
print('time taken: ', end - start)
print()

