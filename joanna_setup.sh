# source /opt/conda/etc/profile.d/conda.sh
cd environment
export PYTHONPATH=`readlink -f /Users/joanna/ccc_mpi4py/libs`:$PYTHONPATH
cd ..
cd nbs
export PYTHONPATH=`readlink -f /Users/joanna/ccc_mpi4py/libs`:$PYTHONPATH
echo "Ran Joanna setup steps"