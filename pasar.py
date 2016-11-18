import csv
import random
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier as RFC

train = open('train.csv')
archivo_csv = csv.reader(train, delimiter=",")
test = open('test.csv')
test_csv = csv.reader(test, delimiter=",")
archivo_csv.next()
test_csv.next()

rfc = RFC()
rfc.fit(archivo_csv)
preds=rfc.predict(test_csv)
rfc.score(test_csv)

"""setDeDatos = []
productosID=set()
userID=set()
contadorReviews=0
promedio=0
total=0
for label in archivo_csv:
	setDeDatos.append(label)
	for columna in range(0,9):
		if columna == 1:
			if setDeDatos[contadorReviews][columna] not in productosID:
					productosID.add(setDeDatos[contadorReviews][columna])
		if columna == 2:
			if setDeDatos[contadorReviews][columna] not in userID:
					userID.add(setDeDatos[contadorReviews][columna])
	contadorReviews +=1
"""

train.close()
test.close()
