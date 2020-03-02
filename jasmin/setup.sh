mkdir ~/PYTHON
export PYTHONPATH="$HOME/PYTHON"  # <- add to ~/.bashrc
cd $PYTHONPATH
git clone https://github.com/scott-hosking/baspy.git
mv building_resilience/risa_preprocessing/notebooks/downloader/dataprocessing.py ./
mv building_resilience/risa_preprocessing/notebooks/downloader/helper.py ./
mkdir ERA