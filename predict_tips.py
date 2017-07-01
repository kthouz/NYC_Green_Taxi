import tip_predictor as tp 
import pandas as pd
import sys

def predict(filename):
	data = pd.read_csv(filename)
	tp.make_predictions(data)

if __name__ == '__main__':
	filename = sys.argv[1]
	predict(filename)