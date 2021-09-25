import numpy as np

with open('txtfiles/switch.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.1, 0.3, num=5):
        for switch in np.linspace(0.1, 0.9, num=5):
            for control in [0,1]:
                f.write("{0} {1} {2}\n".format(
                        lr,
                        switch,
                        control
                    ))
        f.write("{0} 200 0\n".format(lr))
        f.write("{0} 200 1\n".format(lr))

with open('txtfiles/minibatch.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.1, 0.5, num=5):
        for multistage in [0,1]:
            f.write("{0} {1}\n".format(lr, multistage))

with open('txtfiles/multistage.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.1, 0.5, num=5):
        for control in [0,1]:
            for switch in [1e-2, 10**(-1.625), 10**(-1.25), 10**(-0.875), 10**(-0.5)]:
                f.write("{0} {1} {2} {3}\n".format(
                        lr,
                        switch,
                        control,
                        1
                    ))
            f.write("{0} {1} {2} {3}\n".format(
                        lr,
                        200,
                        control,
                        0
                    ))