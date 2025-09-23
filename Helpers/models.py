import tensorflow as tf
from tensorflow import keras

class Models():
    
    def __init__(self) -> None:
        pass
    
    def GergesLSTM(self, input_shape):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        model.add(keras.layers.LSTM(64, return_sequences=True, name='LSTM_1'))
        model.add(keras.layers.LSTM(128, return_sequences=True, name='LSTM_2'))
        model.add(keras.layers.LSTM(64, name='LSTM_3'))
        return model
    
    def BiLSTM(self, input_shape):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True), name='bidirectional'))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True), name='bidirectional_1'))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(128), name='bidirectional_2'))
        return model
    
    def GRU(self, input_shape):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        model.add(keras.layers.GRU(64, return_sequences=True, name='GRU_1'))
        model.add(keras.layers.GRU(128, return_sequences=True, name='GRU_2'))
        model.add(keras.layers.GRU(64, name='GRU_3'))
        return model

    def ResNet1D(self, input_shape):
        def residual_block(X, kernels, stride):
            out = keras.layers.Conv1D(kernels, stride, padding='same')(X)
            out = keras.layers.BatchNormalization()(out)
            out = keras.layers.ReLU()(out)
            
            out = keras.layers.Conv1D(kernels, stride, padding='same')(out)
            out = keras.layers.add([X, out])
            out = keras.layers.BatchNormalization()(out)
            out = keras.layers.ReLU()(out)
            out = keras.layers.MaxPool1D(2, 2)(out)
            return out
        
        kernels = 128
        stride = 2
        inputs = keras.layers.Input(shape=(input_shape))
        X = keras.layers.Conv1D(kernels, stride)(inputs)
        X = residual_block(X, kernels, stride)
        X = residual_block(X, kernels, stride)
        X = residual_block(X, kernels, stride)
        X = keras.layers.Flatten()(X)
        return X, inputs
    
    def SimpleRnnLSTM(self, input_shape, units):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(units=units, activation='relu', input_shape=input_shape))
        return model
    
    def SimpleRNN(self, input_shape, units):
        model = keras.Sequential()
        model.add(keras.layers.SimpleRNN(units=units, input_shape=input_shape))
        return model
    
    def AddClassificationLayer(self, base_model, inputs=None, n_classes=2, dropout=None, denses=[128], sequential=True):
        if dropout is None:
            dropout = []
        
        if sequential:
            # `base_model` será un modelo Sequential
            for i in range(len(denses)):
                base_model.add(keras.layers.Dense(units=denses[i], activation='relu'))
                if i < len(dropout) and dropout[i] != 0:
                    base_model.add(keras.layers.Dropout(dropout[i]))
            base_model.add(keras.layers.Dense(units=n_classes, activation='softmax', name="output_layer"))
            return base_model
        else:
            # `base_model` es un tensor o "bloque" funcional (por ejemplo, X en la red
            # residual). Añadimos capas de forma funcional
            for i in range(len(denses)):
                base_model = keras.layers.Dense(units=denses[i], activation='relu')(base_model)
                if i < len(dropout) and dropout[i] != 0:
                    base_model = keras.layers.Dropout(dropout[i])(base_model)
            base_model = keras.layers.Dense(units=n_classes, activation='softmax', name="output_layer")(base_model)
            # Construimos el modelo funcional final
            model = keras.models.Model(inputs=inputs, outputs=base_model)
            return model

    
    def TrainModel(self, 
                   model=None, X_train=None, X_val=None, y_train=None,
                   y_val=None, dirpath=None,
                   epochs=1, batch_size=32, factor=0.1, patience=1, reduce_lr=False):

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        checkpoin_callback_acc = keras.callbacks.ModelCheckpoint(
            filepath = f'{dirpath}/best_model_acc.h5',
            monitor = 'val_accuracy',
            verbose = 0,
            save_best_only = True,
            mode = 'max',
            save_weights_only = False,
            save_freq="epoch",
            initial_value_threshold=None
        )
        checkpoin_callback_loss = keras.callbacks.ModelCheckpoint(
            filepath = f'{dirpath}/best_model_loss.h5',
            monitor = 'val_loss',
            verbose = 0,
            save_best_only = True,
            mode = 'min',
            save_weights_only = False,
            save_freq="epoch",
            initial_value_threshold=None
        )
        reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor,
                      patience=patience, min_lr=0.0000001)
        
        csv_logger = keras.callbacks.CSVLogger(f"{dirpath}/history_epochs({epochs})_batch_size({batch_size}).csv", separator=",", append=False)

        callbacks = [checkpoin_callback_acc, checkpoin_callback_loss, csv_logger]
        
        if reduce_lr:
            callbacks.append(reduce_lr_on_plateau)        
    
        model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs,
            batch_size=batch_size, 
            verbose=0, 
            callbacks =callbacks
        )
        return model 
