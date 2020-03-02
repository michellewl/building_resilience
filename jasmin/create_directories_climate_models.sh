mkdir -p HadGEM2-CC/{historical_r1, historical_r2, historical_r3, piControl_r1, rcp45_r1, rcp85_r1, rcp85_r2, rcp85_r3}
mkdir -p bcc-csm1-1-m/{historical_r1, historical_r2, historical_r3, piControl_r1, rcp26_r1, rcp45_r1, rcp85_r1}
mkdir -p CanESM2/{historical_r1, historical_r2, historical_r3, piControl_r1, rcp26_r1, rcp26_r2, rcp26_r3, rcp45_r1, rcp85_r1}
mkdir -p BNU-ESM/{historical_r1, piControl_r1, rcp26_r1, rcp45_r1, rcp85_r1}
mkdir ERA

cd ~/PYTHON/bcc-csm1-1-m/rcp26_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/bcc-csm1-1-m/rcp85_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/bcc-csm1-1-m/rcp45_r1/
mkdir mean_bias delta ecdf


cd ~/PYTHON/HadGEM2-CC/rcp45_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/HadGEM2-CC/rcp85_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/HadGEM2-CC/rcp85_r2/
mkdir mean_bias delta ecdf
cd ~/PYTHON/HadGEM2-CC/rcp85_r3/
mkdir mean_bias delta ecdf

cd ~/PYTHON/CanESM2/rcp45_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/CanESM2/rcp85_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/CanESM2/rcp26_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/CanESM2/rcp26_r2/
mkdir mean_bias delta ecdf
cd ~/PYTHON/CanESM2/rcp26_r3/
mkdir mean_bias delta ecdf

cd ~/PYTHON/BNU-ESM/rcp45_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/BNU-ESM/rcp85_r1/
mkdir mean_bias delta ecdf
cd ~/PYTHON/BNU-ESM/rcp26_r1/
mkdir mean_bias delta ecdf