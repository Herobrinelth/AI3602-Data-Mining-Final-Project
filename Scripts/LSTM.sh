echo "Data preprocessing"
python Tools/Dataset.py

echo "Begin to train LSTM model..."
python Method/LSTM_train.py

echo "Recommending by LSTM method..."
python Method/LSTM_recommand.py