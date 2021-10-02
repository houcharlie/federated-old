import numpy as np

with open('txtfiles/switch.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.05, 0.25, num=5):
        for switch in np.linspace(0.1, 0.9, num=5):
            for control in [0,1]:
                f.write("{0} {1} {2}\n".format(
                        round(lr,4),
                        round(switch,4),
                        control
                    ))
        f.write("{0} 200 0\n".format(round(lr,4)))
        f.write("{0} 200 1\n".format(round(lr,4)))

with open('txtfiles/minibatch.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.1, 0.5, num=5):
        for switch in np.logspace(-2, -0.5, num=5):
            f.write("{0} {1} 1\n".format(round(lr,4), round(switch,4)))
        f.write("{0} 200 0\n".format(round(lr,4), round(switch,4)))

with open('txtfiles/multistage.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.05, 0.25, num=5):
        for control in [0,1]:
            for switch in np.linspace(0.002, 0.007, num=5):
                for allow_swap in [0,1]:
                    f.write("{0} {1} {2} {3}\n".format(
                            round(lr,4),
                            round(switch,4),
                            control,
                            allow_swap
                        ))

# with open('txtfiles/constantstage.txt', 'w+') as f:
#     f.write("\n")
#     for lr in np.linspace(0.05, 0.25, num=5):
#         for control in [0,1]:
#             for switch in [0.05, 0.1, 0.15, 0.2, 0.3]:
#                 for allow_swap in [0,1]:
#                     f.write("{0} {1} {2} {3}\n".format(
#                             lr,
#                             switch,
#                             control,
#                             allow_swap
#                         ))
