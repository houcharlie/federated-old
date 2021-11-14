cd /jet/home/houc/multistage/federated/adaptive_lr_decay
sh runlist.sh emnist_scripts/runbazel.sh txtfiles/switch_emnist.txt
sleep 10
sh runlist.sh emnist_scripts/runbazel_minibatch.sh txtfiles/minibatch_emnist.txt 
sleep 10
sh runlist.sh emnist_scripts/runbazel_multistage.sh txtfiles/multistage_emnist.txt
