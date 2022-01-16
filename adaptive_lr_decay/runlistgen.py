import numpy as np

with open('txtfiles/switch.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.05, 0.25, num=5):
        for switch in np.linspace(0.1, 0.9, num=5):
            for control in [0]:
                f.write("{0} {1} {2}\n".format(
                        round(lr,4),
                        round(switch,4),
                        control
                    ))
        f.write("{0} 200 0\n".format(round(lr,4)))
        #f.write("{0} 200 1\n".format(round(lr,4)))

with open('txtfiles/switch_emnist.txt', 'w+') as f:
    f.write("\n")
    for i in range(4):
        f.write(f"0.2 200 0 {i}\n")
        f.write(f"0.1 200 1 {i}\n")
        f.write(f"0.1 0.7000000000000001 0 {i}\n")
        f.write(f"0.1 0.7000000000000001 1 {i}\n")

with open('txtfiles/switch_cifar.txt', 'w+') as f:
    f.write("\n")
    for i in range(4):
        f.write(f"0.15 200 0 {i}\n")
        f.write(f"0.15 0.7 0 {i}\n")

with open('txtfiles/minibatch.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.1, 0.5, num=5):
        for switch in np.logspace(-2, -0.5, num=5):
            f.write("{0} {1} 1\n".format(round(lr,4), round(switch,4)))
        f.write("{0} 200 0\n".format(round(lr,4), round(switch,4)))

with open('txtfiles/minibatch_emnist.txt', 'w+') as f:
    f.write("\n")
    for i in range(4):
        f.write(f"0.2 200 0 {i}\n")
        f.write(f"0.4 0.31622776601683794 1 {i}\n")

with open('txtfiles/minibatch_cifar.txt', 'w+') as f:
    f.write("\n")
    for i in range(4):
        f.write(f"0.1 200 0 {i}\n")
        f.write(f"0.3 0.1334 0 {i}\n")

with open('txtfiles/multistage.txt', 'w+') as f:
    f.write("\n")
    for lr in np.linspace(0.1, 0.5, num=5):
        for control in [0]:
            for switch in np.linspace(0.1, 0.45, num=3):
                for swap_round in [0.5, 0.7, 0.9, 200]:
                    f.write("{0} {1} {2} {3}\n".format(
                            round(lr,4),
                            round(switch,4),
                            control,
                            swap_round
                        ))

with open('txtfiles/multistage_emnist.txt', 'w+') as f:
    f.write("\n")
    for i in range(4):
        f.write(f"0.3 0.1 0 200 {i}\n")
        f.write(f"0.2 0.275 1 200 {i}\n")
        f.write(f"0.1 0.45 0 0.7 {i}\n")
        f.write(f"0.1 0.45 1 0.7 {i}\n")

with open('txtfiles/multistage_cifar.txt', 'w+') as f:
    f.write("\n")
    for i in range(4):
        f.write(f"0.5 0.45 0 0.7 {i}\n")
        f.write(f"0.4 0.275 0 200 {i}\n")
