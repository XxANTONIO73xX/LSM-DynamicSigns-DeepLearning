import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, df=None):
        """
        df: DataFrame con columnas:
            - 'subject'
            - 'class'
            - (15 * 114) columnas de features aplanadas (1710).
        """
        if df is None:
            df = pd.DataFrame()
        self.df = df

        # Nombres de columnas que identifican el sujeto y la clase (glosa)
        self.subject_col = 'subject'
        self.class_col   = 'class'

        # Listas de sujetos para experimentos
        self.train_subjects = [1, 3, 4, 5, 6, 7, 8, 9, 1]
        self.val_subjects   = [2]
        self.test_subjects  = [0, 11]

        # Asumimos que todo lo demás (excepto subject y class) son features
        self.feature_cols = [col for col in self.df.columns 
                             if col not in [self.subject_col, self.class_col]]

        # Verificamos la dimensión esperada (15 frames * 114 features = 1710)
        self.n_frames = 15
        self.n_features = 114
        expected_feats = self.n_frames * self.n_features
        if len(self.feature_cols) != expected_feats:
            print(f"ADVERTENCIA: Se esperaban {expected_feats} columnas de features, "
                  f"pero se encontraron {len(self.feature_cols)}.")

    def _row_to_dict(self, row: pd.Series) -> dict:
        """
        Convierte una fila del DataFrame (pd.Series) en un dict con:
         - 'subject'
         - 'glosa'  (renombra 'class')
         - 'secuencia' (np.array de shape (15,114))
        """
        subject_value = str(row[self.subject_col])
        glosa_value   = str(row[self.class_col])

        # Features aplanadas: shape (1710,)
        feat_values = row[self.feature_cols].values.astype(float)

        # Reshape a (15, 114)
        secuencia = feat_values.reshape(self.n_frames, self.n_features)

        return {
            "subject": subject_value,
            "glosa":   glosa_value,
            "secuencia": secuencia
        }

    def LeaveOneOutExp1(self, subject):
        """
        Ejemplo de leave-one-out:
          - train = (train_subjects + val_subjects) excepto 'subject'
          - test  = subject
        """
        all_subjects = self.train_subjects + self.val_subjects
        train_subjects = [s for s in all_subjects if s != subject]

        df_train = self.df[self.df[self.subject_col].isin(train_subjects)]
        df_test  = self.df[self.subject_col].isin([subject])

        train_data = df_train.apply(self._row_to_dict, axis=1).tolist()
        test_data  = self.df[df_test].apply(self._row_to_dict, axis=1).tolist()
        return train_data, test_data

    # -------------------------------------------------------------------------
    # SPLIT X e Y, y codificación de etiquetas
    # -------------------------------------------------------------------------
    def SplitXandY(self, dataset):
        """
        dataset: lista de dicts (cada dict con 'subject','glosa','secuencia')
        Retorna (X, y):
         - X: shape (num_samples, 15, 114)
         - y: array shape (num_samples,)
        """
        X = [item['secuencia'] for item in dataset]
        y = [item['glosa'] for item in dataset]

        X = np.stack(X)      # (num_samples, 15, 114)
        y = np.array(y)      # (num_samples,)
        return X, y

    def EncodeLabels(self, y_list):
        """
        Recibe una lista de listas (o arrays) de etiquetas: [y_train, y_val, ...]
        Devuelve:
          - splits codificados one-hot
          - label_encoder (LabelEncoder entrenado)
        """
        le = LabelEncoder()
        all_labels = np.concatenate(y_list)
        le.fit(all_labels)

        encoded_splits = []
        n_classes = len(le.classes_)
        for labels in y_list:
            encoded = le.transform(labels)
            encoded_1hot = to_categorical(encoded, num_classes=n_classes)
            encoded_splits.append(encoded_1hot)

        return encoded_splits, le
