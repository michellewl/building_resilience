### Directory to access Jasmin and analyze data

Steps to begin with Jasmin: 
(these are just bullet points for me - extended explanation soon)

```
create a PYTHON directory  
pip install baspy
module load jaspy
```



To access Jasmin from any network, run from terminal:

```
ssh -X crsid@sirius.esc.cam.ac.uk
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa_jasmin --> insert password
ssh -A user_name@jasmin-login1.ceda.ac.uk
### Choose a machine (sci1-sci6)
ssh -X jasmin-sci2
```

