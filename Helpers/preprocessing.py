import os
import json
import numpy as np
import pandas as pd

def extract_keypoints(json_data):
    if not json_data['people']:
        return None, None, None  # No hay personas detectadas

    person = json_data['people'][0]
    keypoints = []

    # Lista de índices de keypoints del cuerpo a considerar (MediaPipe indices)
    body_keypoint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # Extraer y filtrar los keypoints del cuerpo
    if 'pose_keypoints_2d' in person:
        pose_keypoints = person['pose_keypoints_2d']
        body_keypoints_filtered = []
        for k in body_keypoint_indices:
            idx = k * 3
            if idx + 2 < len(pose_keypoints):
                x = pose_keypoints[idx]
                y = pose_keypoints[idx + 1]
                c = pose_keypoints[idx + 2]
                body_keypoints_filtered.extend([x, y, c])
            else:
                # Si el índice no existe, asignar ceros
                body_keypoints_filtered.extend([0, 0, 0])
    else:
        # Si no hay 'pose_keypoints_2d', asignar ceros
        body_keypoints_filtered = [0, 0, 0] * len(body_keypoint_indices)

    # Añadir los keypoints filtrados del cuerpo a la lista principal
    keypoints.extend(body_keypoints_filtered)

    # Añadir todos los keypoints de las manos sin filtrar
    for key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
        if key in person:
            hand_keypoints = person[key]
            if len(hand_keypoints) == 0:
                # Si los keypoints están vacíos, asignar ceros
                # Cada mano tiene 21 landmarks en MediaPipe
                keypoints.extend([0, 0, 0] * 21)
            else:
                keypoints.extend(hand_keypoints)
        else:
            # Si no hay keypoints de la mano, asignar ceros
            # Cada mano tiene 21 landmarks en MediaPipe
            keypoints.extend([0, 0, 0] * 21)
            
    return np.array(keypoints)

def obtener_numero_de_frames(json_path):
    total_frames = os.listdir(os.path.join(json_path, "json"))
    return len(total_frames)

def reduce(n, l):
    z = int(l / (n - 1))
    y = int(np.floor((l - z * (n - 1)) / 2))
    return [y + i * z for i in range(n)], z, y

def sampling(n, l):
    #Calcular la longitud promedio
    z = int(l / (n - 1))
    #print(f"z: {z}")
    y = int(np.floor((l - z * (n - 1)) / 2))
    #print(f"y: {y}")

    # Sequencia base Y
    Y = [y + i * z for i in range(n)]
    #print(f'sequencia base: {Y}')
    # Sequencia de numeros aleatorios R
    R = np.random.randint(1, z + 1, size=n)
    #print(R)
    # Nueva sequencia
    return np.array([min(Y[i] + R[i], l - 1) for i in range(n)]), z

def arrayToString(val):
    return " ".join(str(x) for x in val)

def stringToArray(val):
    return [int(x) for x in val.split()]

def getNSamples(N:int, n:int, l:int):
    N_samples = []
    N_samples_cath = []
    for i in range(N):
        #print(i)
        y_new, z = sampling(n, l)
        y_new = arrayToString(y_new)
        N_samples_cath.append(y_new)
        #if y_new not in N_samples:
        N_samples.append(y_new)
    if len(N_samples) < N:
        print(f"Solo se obtuvieron: {len(N_samples)} originales")
        print(f"Debio a que n:{n}, l:{l}, z:{z}")
    return N_samples, N_samples_cath

def getNSamplesArray(N_samples):
    return [stringToArray(sample) for sample in N_samples]

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error leyendo el archivo {file_path}: {e}")
        return None  # Retorna None si ocurre un erro

def getFrames(archivos, indices):
    indices_set = set(indices)
    resultado = []
    for archivo in archivos:
        # Ejemplo de archivo: "frame_000000000000_keypoints.json"
        # Separar el número entre "frame_" y "_keypoints.json"
        prefix = "frame_"
        suffix = "_keypoints.json"

        if archivo.startswith(prefix) and archivo.endswith(suffix):
            numero_str = archivo[len(prefix):-len(suffix)]
            # Convertir a entero
            try:
                numero = int(numero_str)
            except ValueError:
                # Si no se puede convertir a entero, continuar con el siguiente archivo
                continue
            # Si el número está en indices, agregar al resultado
            if numero in indices_set:
                resultado.append(archivo)

    return resultado

def normalize_vectors(X, Y):
    # Calcular media y desviación estándar de X
    mean_X = np.mean(X)
    std_X = np.std(X)
    
    # Calcular media y desviación estándar de Y
    mean_Y = np.mean(Y)
    std_Y = np.std(Y)
    
    # Normalizar X e Y
    X_normalizado = (X - mean_X) / std_X if std_X != 0 else X - mean_X
    Y_normalizado = (Y - mean_Y) / std_Y if std_Y != 0 else Y - mean_Y
    
    return X_normalizado, Y_normalizado

def make_a_row(label, subject, json_path, frames, indices, normalize=True):
    frames_filtrados = getFrames(frames, indices)
    sequencia = []
    for frame in frames_filtrados:
        file_path = os.path.join(json_path, frame)
        json_data = read_json_file(file_path)
        keypoints = extract_keypoints(json_data=json_data)
        #Separo los valores para eliminar el valor z (que es inservible)
        x_val = keypoints[0::3]
        y_val = keypoints[1::3]
        if normalize:
            x_val, y_val = normalize_vectors(X=x_val, Y=y_val)
        keypoints = np.empty((114,))
        keypoints[0::2] = x_val
        keypoints[1::2] = y_val
        sequencia.append(keypoints)
    sequencia = np.array(sequencia)
    sequencia = sequencia.flatten()
    return [label, subject] + sequencia.tolist()