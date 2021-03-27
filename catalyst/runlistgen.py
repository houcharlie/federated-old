import numpy as np
'''
with open('txtfiles/client_momentum.txt', 'w+') as f:
    for lc in np.logspace(-3, 1, 9):
        for ls in np.logspace(-3,1,9):
            f.write("87313 {0} {1}\n60548 {0} {1}\n74407 {0} {1}\n".format(
                    round(ls,4),
                    round(lc,4)
                ))
'''

with open('txtfiles/no_drift.txt', 'w+') as f:
    for ls in np.logspace(-3,1,9):
        f.write("87313 {0} {1}\n60548 {0} {1}\n74407 {0} {1}\n".format(
                round(ls,4),
                1
            ))