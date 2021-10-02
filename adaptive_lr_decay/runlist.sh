#!/bin/bash
# script to generate a list of submit files and submit them to condor
EXEC=$1
runlist=$2
jobname=$3



# set up results directory
dir=$PWD/runfiles/$jobname/runlist_`date '+%y%m%d_%H.%M.%S'`
echo "Setting up results directory: $dir"
mkdir $PWD/runfiles/$jobname
mkdir $dir
# preamble

i=0

while read p; do
  if [ $i -lt 1 ]
  then
    i=$((i+1))
    continue
  fi
  echo "$EXEC $p"
  echo "sh $EXEC $p" >> $dir/runlist_$i.sh
  echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=1
#SBATCH -n 5
#SBATCH -t 48:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH -o $dir/output_$i.out
singularity exec --nv /ocean/projects/iri180031p/houc/nightly.sif /bin/sh $dir/runlist_$i.sh" >> $dir/runlist_$i.job
  sbatch $dir/runlist_$i.job 
  i=$((i+1))
done <$runlist
#submit to condor