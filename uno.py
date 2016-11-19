import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
import pandas as pd
from datetime import datetime
start_time = datetime.now()
from numpy import savetxt

# CARGAR DATASET 
#---------------------------------------------------------------------------------------------
train = pd.read_csv('train10lines.csv')
test = pd.read_csv('test10lines.csv')
clase_name = 'Prediction' # nombre de variable a predecir
headers    = train.columns.values.tolist()
headers.remove(clase_name)


# TRAIN y TEST
#---------------------------------------------------------------------------------------------
cols = ['Id','HelpfulnessNumerator','HelpfulnessDenominator'] 
colsRes = ['Prediction']
#m_train     = np.random.rand(len(data)) < 0.7
data_train  = train.as_matrix(cols)
data_test   = test.as_matrix(cols)
clase_train = train.as_matrix(colsRes)
clase_test  = test.as_matrix(colsRes)


# CONVIERTE EN NUMPY.MATRIX. Para mejor performance
# -----------------------------------------------------------------------------------------------
data_train = np.matrix(data_train)
data_test  = np.matrix(data_test) 


# MODELO
#---------------------------------------------------------------------------------------------
modelo = RandomForestClassifier(
 random_state      = 1,   # semilla inicial de aleatoriedad del algoritmo
 n_estimators      = 10, # cantidad de arboles a crear
 min_samples_split = 2,   # cantidad minima de observaciones para dividir un nodo
 min_samples_leaf  = 1,   # observaciones minimas que puede tener una hoja del arbol
 n_jobs            = 1    # tareas en paralelo. para todos los cores disponibles usar -1
 )
modelo.fit(X = data_train, y = clase_train)


# PREDICCION
#---------------------------------------------------------------------------------------------
prediccion = modelo.predict(data_test)

#Guardamos el archivo para el submission
#---------------------------------------------------------------------------------------------
np.savetxt('sampleSubmissionUNO.csv', np.c_[range(1,len(test)+1),prediccion], delimiter=',', header = 'Id,Prediction', comments = '', fmt='%f')
