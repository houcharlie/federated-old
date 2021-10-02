import numpy as np

stages_needed = np.ceil(-np.log2(0.1))
stages_needed = np.ceil(-np.log2(0.01))
print('stages needed', stages_needed)
print('grid', np.linspace(0.002, 0.007, num=5))
initial_ratio = 0.007 * 500
ratio_sums = 0
for i in range(int(stages_needed)):
    print("curr val", initial_ratio*2.**float(i))
    ratio_sums += initial_ratio*2.**float(i) + 1

print(ratio_sums/500)