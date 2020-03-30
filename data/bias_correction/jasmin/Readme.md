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
git clone git@github.com:michellewl/building_resilience.git
```
* note that you will have to ensure you have an access to github (ssh key)
to do so you can run from terminal: 

```
ssh-keygen -t rsa -b 4096 -C "your_email_on_github@blabla.com"
cat ~/.ssh/id_rsa.pub
```
Copy the output on the screen and paste it on your github/settings/ssh keys


To setup enviornment run from Jasmin terminal:

```
cd building_building_resilience/data/bias_correction/jasmin

sh setup.sh
```

Then run (this step has to be run each time you login to Jasmin server):

```
module load jaspy
cd PYTHON
```
** Discover what is Jaspy for [here](https://help.jasmin.ac.uk/article/4729-jaspy-envs)


Run:

```
sh create_directories_climate_models.sh
```


For creating csv files for the ERA data with varying thresholds run:

```
sh era_bjob.sh
```  

To check the status of the job:

```
bjobs -a

```

After the jobs are done run: 

```
sh mv_era_files.sh
```


