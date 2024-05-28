# echo "Data preprocessing"
# python Tools/Dataset.py

echo "Begin to train NCF model..."
python Method/NCF_train.py

echo "Recommending by NCF method..."
python Method/NCF_recommand.py