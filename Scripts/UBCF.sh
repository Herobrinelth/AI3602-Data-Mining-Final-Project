echo "Data preprocessing"
python Tools/Dataset.py

echo "Begin to fit UBCF model..."
python Method/UBCF.py