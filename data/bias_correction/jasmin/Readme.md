- many thanks for @Risa Ueno for the downloader folder contents 
### Directory to access Jasmin and analyze data

To access Jasmin from any network, run from terminal:

```
ssh -X crsid@sirius.esc.cam.ac.uk
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa_jasmin --> insert password
ssh -A user_name@jasmin-login1.ceda.ac.uk
### Choose a server (sci1-sci6)
ssh -X jasmin-sci2
```
After login, run:

```
git clone https://github.com/michellewl/building_resilience.git
```

To setup enviornment run from Jasmin terminal:

```

sh ~/building_resilience/data/bias_correction/jasmin/setup/setup.sh
```

Then run (this step has to be run each time you login to Jasmin server):

```
module load jaspy
```
** Discover what is Jaspy for [here](https://help.jasmin.ac.uk/article/4729-jaspy-envs)

### ERA - reanalysis data

0. For creating csv files for the ERA data with threshold of 24c˚ run (this step takes ~4 hours):

```
sh ~/building_resilience/data/bias_correction/jasmin/bjobs/era_bjob.sh
```
* note that currently there is the need to change the path manually:
1. go to 
```
 ~/building_resilience/data/bias_correction/jasmin/bjobs/era/
```
2. eneter each era file and change path to /home/users/your_username/PYTHON
3. run (0)
- The csv files produced will have to grid point with its associated Cooling Degree Days (CDD)

To check the status of the job:

```
bjobs -a
```

After the jobs are done run: 

```
sh ~/building_resilience/data/bias_correction/jasmin/setup/mv_era_files.sh
```

### Climate model correction 

First, let's create folders for the different models: 
```
sh ~/building_resilience/data/bias_correction/jasmin/setup/create_directories_climate_models.sh
```
To run the bias correction (this step takes ~8 hours): 
```
sh ~/building_resilience/data/bias_correction/jasmin/bjobs/model_bjob.sh
```



