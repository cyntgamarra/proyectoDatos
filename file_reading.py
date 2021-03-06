import csv

import data_frame_processing


# Genera un arreglo donde cada elemento es un par {clave, valor}
# nombreArchivo = String con el path del archivo a leer
# header = las columnas que posee el archivo.

def leer_archivo(nombre_archivo):
    data_frame_vec = []
    with open(nombre_archivo, 'rt') as archivo:
        data_frame = csv.DictReader(archivo)
        for frame in data_frame:
            result_with_no_stopwords = data_frame_processing.pre_procesar_frame(frame)
            frame['Text'] = ''.join(result_with_no_stopwords)
            if 'Prediction' in frame:
                frame['Prediction'] = float(frame['Prediction'])
            data_frame_vec.append(frame)
    archivo.close()
    return data_frame_vec

#Metodo
def generar_archivo(data):
    with open('grupo16.csv', 'w') as outfile:
        fieldnames = ['Id', 'Prediction']
        submission_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        submission_writer.writeheader()
        submission_writer.writerows(data)
        
    outfile.close()


#Metodo
def test_generar_Archivo():

    dict_array = [{'Id': 1, 'Prediction': 1}, {'Id': 8, 'Prediction': 3}, {'Id': 9, 'Prediction': 2.5}, {'Id': 10, 'Prediction': 4.5}]
    generar_archivo(dict_array)
    archivo = open("submission.csv", "r")
    contenido = archivo.read()

    print(contenido)
