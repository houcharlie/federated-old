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
'''
with open('txtfiles/no_drift.txt', 'w+') as f:
    for ls in np.logspace(-3,1,9):
        f.write("87313 {0} {1}\n60548 {0} {1}\n74407 {0} {1}\n".format(
                round(ls,4),
                1
            ))
'''
'''
with open('txtfiles/gan_search.txt', 'w+') as f:
    for lr_factor in np.logspace(-3,1,9):
        for tau in np.logspace(-3,1,9):
            f.write("{control} {lr_factor} {tau} {control}_{lr_factor}_{tau}\n".format(
                    control=1,
                    lr_factor=round(lr_factor,4),
                    tau=round(tau,4)
                ))
        f.write("{control} {lr_factor} {tau} {control}_{lr_factor}_{tau}\n".format(
                    control=0,
                    lr_factor=round(lr_factor,4),
                    tau=0
                ))   
'''
with open('txtfiles/gan_search2.txt', 'w+') as f:
    for lr_factor in np.logspace(-3,1,9):
        f.write("{control} {lr_factor} {tau} {control}_{lr_factor}_{tau}\n".format(
                    control=1,
                    lr_factor=round(lr_factor,4),
                    tau=0
                ))   