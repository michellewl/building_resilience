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
cd building_resilience/data/bias_correction/jasmin/setup

sh setup.sh
```

Then run (this step has to be run each time you login to Jasmin server):

```
module load jaspy
```
** Discover what is Jaspy for [here](https://help.jasmin.ac.uk/article/4729-jaspy-envs)


Run:

```
sh ../building_resilience/data/bias_correction/jasmin/setup/create_directories_climate_models.sh
```


For creating csv files for the ERA data with threshold of 24c˚ run:

```
sh ../building_resilience/data/bias_correction/jasmin/bjobs/era_bjob.sh
```  
- The csv files produced will have to grid point with its associated Cooling Degree Days (CDD)

To check the status of the job:

```
bjobs -a
```

After the jobs are done run: 

```
sh ../building_resilience/data/bias_correction/jasmin/setup/mv_era_files.sh
```


