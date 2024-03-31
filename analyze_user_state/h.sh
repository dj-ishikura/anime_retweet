#PBS -q wLrchq
#PBS -l select=1:ncpus=1
#PBS -v DOCKER_IMAGE=prg-env:latest

cd $PBS_O_WORKDIR

echo "Hello, World!"