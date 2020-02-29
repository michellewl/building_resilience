mkdir ~/PYTHON
export PYTHONPATH="$HOME/PYTHON"  # <- add to ~/.bashrc
cd $PYTHONPATH
git clone https://github.com/scott-hosking/baspy.git
cp building_resilience/risa_preprocessing/notebooks/downloader/dataprocessing.py ./
cp building_resilience/risa_preprocessing/notebooks/downloader/helper.py ./