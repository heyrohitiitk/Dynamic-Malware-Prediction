echo "starting data collection"
cuckoo submit $1
python3 transfer.py
python3 datacsv.py
cuckoo clean
echo "dataset ready"
