mkdir ~/PYTHON
export PYTHONPATH="$HOME/PYTHON"  # <- add to ~/.bashrc
cd $PYTHONPATH
git clone https://github.com/scott-hosking/baspy.git
mv building_resilience/data/bias_correction/jasmin/downloader/dataprocessing.py ./
mv building_resilience/data/bias_correction/jasmin/downloader/helper.py ./
mkdir ERA
