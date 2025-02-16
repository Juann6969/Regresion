import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

print(df.info())
print(df.describe())

df = df.drop(df[(df['UnitPrice'] <= 0) | (df['UnitPrice'] > 100)].index)
df = df.drop(df[(df['Quantity'] <= 0) | (df['Quantity'] > 100)].index)
descripciones_a_eliminar = ["AMAZON FEE", "Manual", "Adjust bad debt", "POSTAGE",
                            "DOTCOM POSTAGE", "CRUK Commission", "Bank Charges", "SAMPLES"]

duplicados = df.duplicated().sum()
df = df.drop_duplicates(keep='first')
print(f"Se han eliminado {duplicados} filas duplicadas")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Sales'] = df['Quantity'] * df['UnitPrice']
df_ventas_diarias = df.groupby(df['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
df_ventas_diarias.columns = ['InvoiceDate', 'TotalSales'] 

df_ventas_diarias['InvoiceDate'] = pd.to_datetime(df_ventas_diarias['InvoiceDate'])
data_entrenamiento = df_ventas_diarias[(df_ventas_diarias['InvoiceDate'] >= '2010-12-01') & (df_ventas_diarias['InvoiceDate'] <= '2011-10-08')]
data_validacion = df_ventas_diarias[(df_ventas_diarias['InvoiceDate'] >= '2011-10-09') & (df_ventas_diarias['InvoiceDate'] <= '2011-11-08')]
data_test = df_ventas_diarias[(df_ventas_diarias['InvoiceDate'] >= '2011-11-09') & (df_ventas_diarias['InvoiceDate'] <= '2011-12-09')]

X_train = data_entrenamiento['TotalSales'].values[:-1].reshape(-1, 1) 
y_train = data_entrenamiento['TotalSales'].values[1:]  

X_val = data_validacion['TotalSales'].values[:-1].reshape(-1, 1)
y_val = data_validacion['TotalSales'].values[1:]

X_test = data_test['TotalSales'].values[:-1].reshape(-1, 1)
y_test = data_test['TotalSales'].values[1:]

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)


predictions_rf = model_rf.predict(X_test)

mape_rf = np.mean(np.abs((y_test - predictions_rf) / y_test)) * 100
print(f"MAPE: {mape_rf}%")


plt.figure(figsize=(14, 7))
plt.plot(y_test, color='blue', label='Ventas Reales')
plt.plot(predictions_rf, color='orange', label='Predicciones Random Forest')
plt.title('Predicciones vs Ventas Reales (Random Forest)')
plt.xlabel('Días')
plt.ylabel('Ventas')
plt.legend()
plt.show()

predictions_train_rf = model_rf.predict(X_train)

plt.figure(figsize=(14, 7))
plt.plot(y_train, color='blue', label='Ventas Reales (Entrenamiento)')
plt.plot(predictions_train_rf, color='orange', label='Predicciones Random Forest (Entrenamiento)')
plt.title('Predicciones vs Ventas Reales (Random Forest) - Entrenamiento')
plt.xlabel('Días')
plt.ylabel('Ventas')
plt.legend()
plt.show()