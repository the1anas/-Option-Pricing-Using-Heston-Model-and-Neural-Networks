import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def build_model(trial):
    model = Sequential()
    model.add(Dense(trial.suggest_int('layer_size', 64, 512), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(trial.suggest_float('dropout_rate', 0.1, 0.5)))
    model.add(Dense(5))  # 5 parameters of the Heston model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def objective(trial):
    model = build_model(trial)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)],
        verbose=0
    )
    return min(history.history['val_loss'])

# Load data
data = pd.read_csv('processed_historical_data.csv')
X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return', 'Volatility']].values
y = np.array([[2.0, 0.01, 0.1, -0.5, 0.01]] * len(data))  # Dummy Heston parameters for example

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Optimize model
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Train best model
best_trial = study.best_trial
best_model = build_model(optuna.trial.FixedTrial(best_trial.params))
history = best_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)],
    verbose=1
)

# Save best model
best_model.save('best_stock_prediction_model.keras')

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.show()

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)
    
    return mse, r2, mae

mse, r2, mae = evaluate_model(best_model, X_val, y_val)
print(f"MSE: {mse}, R^2: {r2}, MAE: {mae}")

# Plot predictions vs actual values
predictions = best_model.predict(X_val)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_val)), y_val[:, 0], label='Actual')
plt.scatter(range(len(predictions)), predictions[:, 0], label='Predicted')
plt.title('Predictions vs Actual')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.savefig('predictions_vs_actual.png')
plt.show()
