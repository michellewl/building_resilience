mkdir ~/PYTHON
export PYTHONPATH="$HOME/PYTHON"  # <- add to ~/.bashrc
cd $PYTHONPATH
git clone https://github.com/scott-hosking/baspy.git
cp -r ../building_resilience/data/bias_correction/jasmin/downloader ./
mkdir ERA
mkdir -p bjobs_run/model bjobs_run/era
cd PYTHON
