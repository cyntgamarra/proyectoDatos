from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt
import csv
import file_reading
test_Id = []

train = file_reading.leer_archivo('train10lines.csv')
test = file_reading.leer_archivo('test10lines.csv')

COLUMNAS = 9
FILAS = 9
matrix = []
matrix1 = []
vector_Prediction = []

for i in range(FILAS):
	matrix.append([])
        for j in range(COLUMNAS):
            matrix[i].append(0)

for indice in range(FILAS):
        vector_Prediction.append((float(train[indice]['Prediction'])))
        texto = train[indice]['Text'].split()
for j in range(len(texto)):
        hash_val = hash(texto[j]) % COLUMNAS
        matrix[indice][hash_val] += 1

rf = RandomForestClassifier(bootstrap=True, criterion='gini',
            max_depth=None, max_features='auto',
            min_samples_leaf=1, min_samples_split=2,
            n_estimators=200, n_jobs=1,
            oob_score=False, random_state=None, verbose=0)
rf.fit(matrix, vector_Prediction)

###### Test

test_Id = []
salida_predicciones = []
COLUMNAS2 = COLUMNAS #7919
FILAS2 = len(test) #568454

for i in range(FILAS2):
        matrix1.append([])
        for j in range(COLUMNAS2):
            matrix1[i].append(0)

for indice in range(FILAS2):
        test_Id.append(test[indice]['Id'])
        texto = test[indice]['Text'].split()
        for j in range(len(texto)):
            hash_val = hash(texto[j]) % COLUMNAS2
            matrix1[indice][hash_val] += 1
mat_aux = rf.predict(matrix1)
for i in range(FILAS2):
        prediccion = mat_aux[i]
        salida_predicciones.append({'Id': test_Id[i], 'Prediction': prediccion})

file_reading.generar_archivo(salida_predicciones)


