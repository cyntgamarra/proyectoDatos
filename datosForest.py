# First let's import the dataset, using Pandas.
import pandas as pd
import numpy as np 

train = pd.read_csv('train10lines.csv')
test = pd.read_csv('test10lines.csv')

from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt

cols = ['Id','HelpfulnessNumerator','HelpfulnessDenominator'] 
colsRes = ['Prediction']

trainArr = train.as_matrix(cols) #training array
trainRes = train.as_matrix(colsRes) # training results

## Training!
rf = RandomForestClassifier(bootstrap=True, criterion='gini',
            max_depth=None, max_features='auto',
            min_samples_leaf=1, min_samples_split=2,
            n_estimators=200, n_jobs=1,
            oob_score=False, random_state=None, verbose=0)
rf.fit(trainArr, trainRes) # fit the data to the algorithm

testArr = test.as_matrix(cols)
results = rf.predict(testArr)
test['predictions'] = results
#savetxt('submission2.csv', results, delimiter=',', fmt='%f')
np.savetxt('sampleSubmissiondatosForest.csv', np.c_[range(1,len(test)+1),results], delimiter=',', header = 'Id,Prediction', comments = '', fmt='%f')
