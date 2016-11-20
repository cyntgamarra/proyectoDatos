import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt
import csv
import file_reading
test_Id = []

train = file_reading.leer_archivo('train10lines.csv')
test = file_reading.leer_archivo('test10lines.csv')

"""# CARGAR DATASET--------------------------------------------------------------
train = pd.read_csv('train10lines.csv')
test = pd.read_csv('test10lines.csv')
"""
# TRAIN y TEST----------------------------------------------------------------
columnas = ['Id','HelpfulnessNumerator','HelpfulnessDenominator','Summary','Text'] 
colResultado = ['Prediction']
trainArr = train.as_matrix(columnas) #training array
trainRes = train.as_matrix(colResultado) # training results
testArr = test.as_matrix(columnas)

# Preparando los datos :)-------------------------------------------------------
# Aplicar el tokenizado a los strings
for fil in range(0,(len(trainArr))):
	for col in range(0,5):
		if (col == 3):	
			trainArr[fil][col] = trainArr[fil][col].split()
		if (col == 4):	
			trainArr[fil][col] = trainArr[fil][col].split()

for fil in range(0,(len(testArr))):
	test_Id.append(test[fil]['Id'])
	for col in range(0,5):
		if (col == 3):	
			testArr[fil][col] = testArr[fil][col].split()
		if (col == 4):	
			testArr[fil][col] = testArr[fil][col].split()


# Entrenando-----------------------------------------------------------------
rf = RandomForestClassifier(bootstrap=True, criterion='gini',
            max_depth=None, max_features='auto',
            min_samples_leaf=1, min_samples_split=2,
            n_estimators=200, n_jobs=1,
            oob_score=False, random_state=None, verbose=0)

rf.fit(trainArr, trainRes) # fit the data to the algorithm

# PREDICCION -----------------------------------------------------------------
results = rf.predict(testArr)
test['predictions'] = results
#Guardamos el archivo para el submission--------------------------------------
"""np.savetxt('sampleSubmissionDATOSFOREST.csv', np.c_[range(1,len(test)+1),results], delimiter=',', header = 'Id,Prediction', comments = '', fmt='%f')"""

with open("rf_benchmark.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow(["Id","Prediction"])
    for i, predicted_digit in enumerate(results):
        writer.writerow((test_Id[i],predicted_digit))
