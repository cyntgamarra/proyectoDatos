import csv
import random
import matplotlib.pyplot as plt
import math
import operator
from collections import defaultdict

train = open('train.csv')
archivo_csv = csv.reader(train, delimiter=",")
archivo_csv.next()
setDeDatos = []
productosID={}
textID={}
userID=set()
contadorReviews=0
promedio=0
resultado=()
for label in archivo_csv:
	setDeDatos.append(label)
	(idlabel,productID,userID,profileName,helpfullnessN,helpfullnessD,pred,time,summ,text) = label
	if setDeDatos[contadorReviews][1] not in textID:
		textID[setDeDatos[contadorReviews][1]]= list()
		textID[setDeDatos[contadorReviews][1]].append(setDeDatos[contadorReviews][8])
	else:
		textID[setDeDatos[contadorReviews][1]].append(setDeDatos[contadorReviews][8])
	for columna in range(0,9):
		if columna == 1:
			if setDeDatos[contadorReviews][columna] not in productosID:
				productosID[setDeDatos[contadorReviews][columna]] = 1
			else:
				productosID[setDeDatos[contadorReviews][columna]] += 1		
		if columna == 2:
			if setDeDatos[contadorReviews][columna] not in userID:
					userID.add(setDeDatos[contadorReviews][columna])
		if columna == 6:
			promedio += float(setDeDatos[contadorReviews][columna])
	contadorReviews += 1

#for clave in textID:
#	print clave
#	print textID[clave]

resultado = sorted(productosID.items(), key=operator.itemgetter(1))
resultado.reverse()
print "La cantidad de reviews: %d" %(len(setDeDatos)-1)
print "La cantidad de productos: %d" %(len(productosID))
print "La cantidad de usuarios: %d" %(len(userID))
print "El promedio del puntaje: %f" %(promedio / (len(setDeDatos)-1))
print "El producto mas vendido: %s" %(resultado[0][0])
print textID[resultado[0][0]]
print setDeDatos[0][9]
train.close()
