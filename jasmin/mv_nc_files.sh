find . -type f -name "bc_mean_bcc-csm1-1-m_rcp85_r1*.csv" -exec cp -n {} bcc-csm1-1-m/rcp85_r1/ \;
find . -type f -name "bc_mean_bcc-csm1-1-m_rcp26_r1*.csv" -exec cp -n {} bcc-csm1-1-m/rcp26_r1/ \;
find . -type f -name "HadGEM2-CC_rcp85_r3*.csv" -exec cp -n {} HadGEM2-CC/historical_r3i1p1/r3 \;
find . -type f -name "bcc-csm1-1-m_r1i1p1_rcp26*mean_bc.nc" -exec cp -n {} bcc-csm1-1-m/rcp26_r1/mean_bias/ \;
find . -type f -name "bcc-csm1-1-m_r1i1p1_rcp85*mean_bc.nc" -exec cp -n {} bcc-csm1-1-m/rcp85_r1/mean_bias/ \;