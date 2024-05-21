echo "Data preprocessing"
python Tools/Dataset.py

echo "Begin to fit Cluster model..."
python Method/Cluster.py