mkdir ~/PYTHON
export PYTHONPATH="$HOME/PYTHON"  # <- add to ~/.bashrc
cd $PYTHONPATH
git clone https://github.com/scott-hosking/baspy.git
cp ~/building_resilience/data/bias_correction/bias_correction.py ./
cp ~/building_resilience/data/bias_correction/jasmin/thresholds.py ./
cp ~/building_resilience/data/bias_correction/jasmin/downloader/dataprocessing.py ./
cp ~/building_resilience/data/bias_correction/jasmin/downloader/helper.py ./
mkdir ERA
mkdir NC
cd PYTHON
