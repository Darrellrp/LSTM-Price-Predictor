import random
import quandl
import csv
import os.path
from time import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Dense, Input, LSTM, TimeDistributed, Dropout
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange

# Data Source
AV_API_KEY = 'OHL1UQBAF94QK9FH'
DIR_RAW_DATA = 'data/alpha-vantage/raw'
DIR_RAW_CLEAN = 'data/alpha-vantage/clean'
FILE_PATH_RAW_DATA = DIR_RAW_DATA + '/{}_daily_full.csv'
FILE_PATH_CLEAN_DATA = DIR_RAW_CLEAN + '/{}_daily_full_{}.csv'
FETCH_COLLECTION = False

# Pre processing
TRAIN_TEST_SPLIT = 0.99
SAMPLE_SIZE = 2000

# LSTM
LSTM_HIDDEN_SIZE = 10
LSTM_CODE_SIZE = 6

# Training
EPOCHS = 10
BATCH_SIZE = 32
VERBOSE = 1

# Optimization Neural Network
LEARNING_RATE = 0.05
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = None
DECAY = 0.0
AMSGRAD = False

# Compilation Neural Network
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['accuracy']


# Tensorflow
FILE_PATH_TF_LOGS = 'logs/{}'

# Post Training
EVALUATE = False
PREDICT = True
NUMBER_SHOWN_OF_PREDICTIONS = 4


# Read market items from text file
market_items = open('market-items.txt').readline().split(', ')
# Select active market items
market_items = [i for i in market_items if '-' not in i]

# Remove newline character from market items
market_items = [m.replace('\n', '') for m in market_items]

data = None
targets = None

# Forex Data source TODO: Integrate Forex
# fx = ForeignExchange(key=AV_API_KEY, output_format='pandas')
# mmi = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='full')

# if clean data or targets doesn't exist, clean and store again
if (not os.path.exists('{}/{}'.format(DIR_RAW_CLEAN, 'data_collection.csv'))
		or not os.path.exists('{}/{}'.format(DIR_RAW_CLEAN, 'targets_collection.csv'))) or FETCH_COLLECTION:

	# ***************************************     Retrieve financial data     ***************************************
	for market_item_name in market_items:
		file_path_raw = FILE_PATH_RAW_DATA.format(market_item_name)

		if not os.path.exists(file_path_raw):
			print('Fetching Daily Time Series -csv (Alpha Vantage) - {}...'.format(market_item_name))
			ts = TimeSeries(key=AV_API_KEY, output_format='pandas', indexing_type='date')

			# Get json object with the intraday data and another with  the call's metadata
			market_item, meta_data = ts.get_daily(market_item_name, outputsize='full')

			print(market_item_name)
			print('Record size: {}'.format(market_item.shape[0]))
			print('')

			rename_columns = {'1. open': 'open', '2. high': 'high', '3. low': ' low',
							  '4. close': 'close', '5. volume': 'volume'}

			# Rename columns
			market_item.rename(columns=rename_columns, inplace=True)
			market_item.index.name = 'timestamp'

			# Store Market item data
			market_item.to_csv(file_path_raw, header=True)

		else:
			print('Loading Daily Time Series -csv (Alpha Vantage) - {}...'.format(market_item_name))
			market_item = pd.read_csv(file_path_raw, parse_dates=['timestamp'])
			market_item.drop('timestamp', axis=1, inplace=True)

		market_item['close'].plot()
		plt.title(market_item_name)
		# plt.show()

	# ***************************************     Pre-processing     ***************************************
		file_path_raw = FILE_PATH_RAW_DATA.format(market_item_name)
		file_path_clean_data = FILE_PATH_CLEAN_DATA.format(market_item_name, 'data')
		file_path_clean_targets = FILE_PATH_CLEAN_DATA.format(market_item_name, 'targets')

		# if clean data or targets doesn't exist, clean and store again
		if not os.path.exists(file_path_clean_data) or not os.path.exists(file_path_clean_targets):
			print('Pre-processing {}...'.format(market_item_name))
			print('')

			# Remove last row, Drop timestamp column
			market_item_data = market_item.drop(market_item.tail(1).index)

			# Select close columns, Drop First row (Next days close price)
			market_item_targets = market_item['close'].drop(market_item.head(1).index)

			# Convert to float32
			market_item_data = market_item_data.astype('float32')
			market_item_targets = market_item_targets.astype('float32')

			scaler = MinMaxScaler(feature_range=(0, 1))
			scaler = scaler.fit(market_item_data.values)

			# Normalize financial data
			market_item_data = scaler.transform(market_item_data.values)

			# Store cleaned financial data
			np.savetxt(FILE_PATH_CLEAN_DATA.format(market_item_name, 'data'), market_item_data, delimiter=',')
			np.savetxt(FILE_PATH_CLEAN_DATA.format(market_item_name, 'targets'), market_item_targets, delimiter=',')

		else:
			# Read cleaned financial data
			market_item_data = pd.read_csv(FILE_PATH_CLEAN_DATA.format(market_item_name, 'data'), header=None).values
			market_item_targets = pd.read_csv(FILE_PATH_CLEAN_DATA.format(market_item_name, 'targets'), header=None).values

		if data is None:
			data = market_item_data
			targets = market_item_targets
		else:
			# TODO: Concatenate columns grouped by timestamp
			data = np.concatenate((data, market_item_data))
			targets = np.concatenate((targets, market_item_targets))

	np.savetxt('{}/{}'.format(DIR_RAW_CLEAN, 'data_collection.csv'), data, delimiter=',')
	np.savetxt('{}/{}'.format(DIR_RAW_CLEAN, 'targets_collection.csv'), targets, delimiter=',')

else:
	print('Loading Collection -csv (Alpha Vantage)...')
	data = pd.read_csv('{}/{}'.format(DIR_RAW_CLEAN, 'data_collection.csv'), header=None).values
	targets = pd.read_csv('{}/{}'.format(DIR_RAW_CLEAN, 'targets_collection.csv'), header=None).values

print('')

# Specify number of records
if isinstance(SAMPLE_SIZE, int):
	data = data[:SAMPLE_SIZE]
	targets = targets[:SAMPLE_SIZE]

print('Retrieve Daily Time Series -csv (Alpha Vantage) - {}'.format(market_items))
print('')

# Train Test Splits
print('Generating Train Test split...')
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

print('Number of rows train data: {}'.format(len(x_train)))
print('Number of rows test data: {} '.format(len(x_test)))
print('')

input_dim = x_train.shape[1]

# Reshape 2D to 3D (samples, rows, columns)
x_train = x_train.reshape(1, x_train.shape[0], x_train.shape[1])
x_test = x_test.reshape(1,  x_test.shape[0], x_test.shape[1])
y_train = y_train.reshape(1, y_train.shape[0], 1)
y_test = y_test.reshape(1, y_test.shape[0], 1)

# ***************************************     Neural Network Architecture     ***************************************
model = Sequential()

# LSTM
model.add(LSTM(LSTM_HIDDEN_SIZE, return_sequences=True, input_shape=(None, input_dim)))
model.add((Dropout(0.2)))

model.add(LSTM(LSTM_HIDDEN_SIZE, return_sequences=True))
model.add((Dropout(0.2)))

model.add(LSTM(LSTM_HIDDEN_SIZE, return_sequences=True))
model.add((Dropout(0.2)))

model.add(LSTM(LSTM_HIDDEN_SIZE, return_sequences=True))
model.add((Dropout(0.2)))

model.add(LSTM(LSTM_HIDDEN_SIZE, return_sequences=True))
model.add((Dropout(0.2)))

# model.add(LSTM(LsSTM_HIDDEN_SIZE, return_sequences=True))
model.add(LSTM(input_dim, return_sequences=True))

model.add(TimeDistributed((Dense(1, activation='relu'))))

adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON, decay=DECAY, amsgrad=AMSGRAD)
# optimizers: 'adadelta' | loss: 'mape', 'binary_crossentropy'
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

tensorboard = TensorBoard(log_dir=FILE_PATH_TF_LOGS.format(time()))

# model.summary()

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
						verbose=VERBOSE, callbacks=[tensorboard], validation_data=(x_test, y_test))

# ***************************************     Post Training     ***************************************
if EVALUATE:
	score, accuracy = model.evaluate(y_test, y_test)
	print('Test score:', score)
	print('Test accuracy:', accuracy)

if PREDICT:
	reconstructed = model.predict(x_test)

	# Show a number of predictions
	for i in range(NUMBER_SHOWN_OF_PREDICTIONS):
		# Generate random index
		rand_i = random.randrange(0, x_test.shape[0])

		print('')

		# Print actual label
		print('Actual closing price: {}'.format(y_test[0, rand_i]))
		print('Predicted closing price: {}'.format(reconstructed[0, rand_i]))

