import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv')

print(df.info())

print(df.describe())

#Eliminamos los registros donde el precio unitario sea <= que 0 ya que son ajustes o regalos de promoción que no 
# nos sirven para entrenar el modelo y hacemos lo mismo con las devoluciones ya que para este modelo solo predeciremos 
# las ventas sin tener en cuenta si se devuelven productos, y eliminamos también outliers.
print(df[df['UnitPrice'] <=0 ])
df = df.drop(df[(df['UnitPrice'] <= 0) | (df['UnitPrice'] > 1000)].index)
df = df.drop(df[(df['Quantity'] <= 0) | (df['Quantity'] > 70000)].index)



#Eliminamos todos las filas duplicadas conservando la primera con keep=first
duplicados = df.duplicated().sum()
df = df.drop_duplicates(keep='first')
print(f"Se han eliminado {duplicados} filas duplicadas")

#Creamos una nueva columna con los datos de ventas para cada día
#Transformamos las fechas a formato datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Sales'] = df['Quantity'] * df['UnitPrice']
df_ventas_diarias = df.groupby(df['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
df_ventas_diarias.columns = ['InvoiceDate', 'TotalSales']  # Renombramos las columnas
print(df.head())
print(df_ventas_diarias)

#Separamos el dataset en la parte de entrenamiento, validación y test
df_ventas_diarias['InvoiceDate'] = pd.to_datetime(df_ventas_diarias['InvoiceDate'])
data_entrenamiento = df_ventas_diarias[(df_ventas_diarias['InvoiceDate'] >= '2010-12-01') & (df_ventas_diarias['InvoiceDate'] <= '2011-10-08')]
data_validacion = df_ventas_diarias[(df_ventas_diarias['InvoiceDate'] >= '2011-10-09') & (df_ventas_diarias['InvoiceDate'] <= '2011-11-08')]
data_test = df_ventas_diarias[(df_ventas_diarias['InvoiceDate'] >= '2011-11-09') & (df_ventas_diarias['InvoiceDate'] <= '2011-12-09')]
print(data_test['TotalSales'])
print(f"Rango de fechas del conjunto de test: ", data_test)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])  # Ventana de tiempo
        y.append(data[i+window_size])  # Día siguiente
    return np.array(X), np.array(y)

window_size = 5  # Número de días previos para predecir

# Extraemos la columna de ventas, se transforman los dataframe en arrays
data_entrenamiento = data_entrenamiento['TotalSales'].values
data_validacion = data_validacion['TotalSales'].values
data_test = data_test['TotalSales'].values

X_train, y_train = create_sequences(data_entrenamiento, window_size)
X_val, y_val = create_sequences(data_validacion, window_size)
X_test, y_test = create_sequences(data_test, window_size)


# Redimensionamos para LSTM 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Escaladores para entradas y salidas
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Escalamos X manteniendo la estructura correcta
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[1])).reshape(X_val.shape)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
print(f"Tamaño original del test: {len(data_test)}")
print(f"Tamaño después de crear secuencias: {X_test.shape[0]}")

# Creamos las capas del modelo
model = Sequential()
model.add(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
model.add(LSTM(units=64, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamos el modelo
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=10, 
    batch_size=32, 
    validation_data=(X_val_scaled, y_val_scaled)
)

test_loss = model.evaluate(X_test_scaled, y_test_scaled)
print(f"Pérdida en el conjunto de test: {test_loss}")