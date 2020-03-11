find . -type f -name "__*.csv" -exec mv -n {} ERA \;
cd ERA
cat *.csv > era_merged.csv
