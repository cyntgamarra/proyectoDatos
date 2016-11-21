from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import read_and_process

test_Id = []
train = read_and_process.leer_archivo('train10lines.csv', 'train')
test = read_and_process.leer_archivo('test10lines.csv', 'test')

COLUMNAS = 9
FILAS = 9
matrix = []
matrix1 = []
vector_Prediction = []

for i in range(FILAS):
	matrix.append([])
        for j in range(COLUMNAS):
            matrix[i].append(0)

for indice in range(0,FILAS):
        vector_Prediction.append((float(train['Prediction'][indice])))
        texto = train['Text'][indice].split()
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
        test_Id.append(test['Id'][indice])
        texto = test['Text'][indice].split()
        for j in range(len(texto)):
            hash_val = hash(texto[j]) % COLUMNAS2
            matrix1[indice][hash_val] += 1
mat_aux = rf.predict(matrix1)
for i in range(FILAS2):
        prediccion = mat_aux[i]
        salida_predicciones.append({'Id': test_Id[i], 'Prediction': prediccion})
df = pd.DataFrame(salida_predicciones)
read_and_process.generar_archivo_submission(df)
