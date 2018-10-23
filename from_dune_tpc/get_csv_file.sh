cd ../input_files
rm -rf training/*.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Training/*_w.csv training/
rm -rf testing/*.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Testing/*_w.csv testing/
rm -rf validation/*.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Validation/*_w.csv validation/
