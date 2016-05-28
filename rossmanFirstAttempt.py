#! usr/env/bin python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv as csv

# raw data files location
trainFile = "/home/deadlytackler/Documents/python/kaggleRossman/data/train.csv"
testFile = "/home/deadlytackler/Documents/python/kaggleRossman/data/test.csv"
storeFile = "/home/deadlytackler/Documents/python/kaggleRossman/data/store.csv"

# read csv files

traindf = pd.read_csv(trainFile, low_memory=False)
testdf = pd.read_csv(testFile, low_memory=False)
storedf = pd.read_csv(storeFile, low_memory=False)

# print()
# print(traindf.head(5))

# print()
# print(traindf['Date'].unique())

# Date values are from 01/01/2013 to 31/07/2015
# Week starts from Monday.

# print()
# print(storedf.head())

# TO-DO
# combine CompetitionOpenSinceMonth & CompetitionOpenSinceYear in storedf.
# add storeType, Assortment columns in traindf from storedf.
# look at competitionDistance, if it's in mtr change to km

print()
storedf.info()
print(storedf['StoreType'].unique())
print(storedf['Assortment'].unique())
