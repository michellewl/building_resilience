find . -type f -name "__*.csv" -exec cp -n {} ERA \;
rm __*.csv
cd ERA
cat *.csv > era_merged.csv
