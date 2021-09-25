cd /jet/home/houc/multistage/federated/adaptive_lr_decay
sh runlist.sh runbazel.sh txtfiles/switch.txt
sh runlist.sh runbazel_minibatch.sh txtfiles/minibatch.txt 
sh runlist.sh runbazel_multistage.sh txtfiles/multistage.txt