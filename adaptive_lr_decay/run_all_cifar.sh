cd /jet/home/houc/multistage/federated/adaptive_lr_decay
sh runlist.sh cifar_scripts/runbazel.sh txtfiles/switch.txt cifar_long
sleep 5
sh runlist.sh cifar_scripts/runbazel_minibatch.sh txtfiles/minibatch.txt cifar_long
sleep 5
sh runlist.sh cifar_scripts/runbazel_multistage.sh txtfiles/multistage.txt cifar_long
