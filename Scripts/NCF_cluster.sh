# echo "Data preprocessing"
# python Tools/Dataset.py

echo "Begin to train NCF + cluster model..."
python Method/NCF_train_cluster.py

echo "Recommending by NCF + cluster method..."
python Method/NCF_recommand_cluster.py