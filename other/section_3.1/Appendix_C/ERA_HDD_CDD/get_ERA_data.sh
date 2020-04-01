echo("Loading modules")
module load python3
module use /g/data3/hh5/public/modules
module load conda/analysis3

echo("Retrieving data")
python get_T2_2018.py