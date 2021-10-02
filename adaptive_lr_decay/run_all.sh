cd /jet/home/houc/multistage/federated/adaptive_lr_decay
sh runlist.sh emnist_scripts/runbazel.sh txtfiles/switch.txt
sh runlist.sh emnist_scripts/runbazel_minibatch.sh txtfiles/minibatch.txt 
sh runlist.sh emnist_scripts/runbazel_multistage.sh txtfiles/multistage.txt
