# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
from google.colab import files
import io

print("Загрузите файл train.csv с вашего компьютера")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
train_df = pd.read_csv(io.BytesIO(uploaded[file_name]))

print("Данные успешно загружены!")
print(f"Размер тренировочных данных: {train_df.shape}")

# Базовый анализ
print("Распределение целевой переменной:")
print(train_df['y'].value_counts(normalize=True))

# Простая и эффективная предобработка
def smart_preprocessing(df):
    df_processed = df.copy()
    
    # Удаляем ID
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)
    
    # Анализируем и преобразуем категориальные переменные
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Специфичная обработка для каждого категориального признака
    for col in categorical_cols:
        if col == 'poutcome':
            # poutcome - очень важный признак, кодируем осмысленно
            mapping = {'nonexistent': 0, 'failure': 1, 'success': 2}
            df_processed[col] = df_processed[col].map(mapping)
        elif col == 'month':
            # Месяц кодируем как порядковый признак
            month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            df_processed[col] = df_processed[col].map(month_map)
        elif col == 'day_of_week':
            # День недели
            day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
            df_processed[col] = df_processed[col].map(day_map)
        elif col in ['default', 'housing', 'loan']:
            # Бинарные признаки
            df_processed[col] = df_processed[col].map({'no': 0, 'yes': 1, 'unknown': 0.5})
        else:
            # Для остальных - frequency encoding
            freq_encoding = df_processed[col].value_counts().to_dict()
            df_processed[col] = df_processed[col].map(freq_encoding)
    
    # Создаем несколько простых и осмысленных новых признаков
    df_processed['previous_contact'] = (df_processed['pdays'] != 999).astype(int)
    df_processed['contact_density'] = df_processed['campaign'] / (df_processed['previous'] + 1)
    df_processed['age_group'] = pd.cut(df_processed['age'], bins=[0, 30, 40, 50, 60, 100], 
                                      labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Экономические индикаторы
    df_processed['economic_index'] = df_processed['cons.price.idx'] * df_processed['cons.conf.idx']
    
    # Удаляем исходные колонки, которые были преобразованы
    df_processed = df_processed.drop(['pdays'], axis=1)
    
    return df_processed

# Применяем предобработку
train_processed = smart_preprocessing(train_df)

# Разделение на features и target
X = train_processed.drop('y', axis=1)
y = train_processed['y']

print(f"Features shape: {X.shape}")

# Анализ корреляций
plt.figure(figsize=(12, 10))
correlation_matrix = X.corrwith(y).sort_values(ascending=False)
sns.barplot(x=correlation_matrix.values, y=correlation_matrix.index)
plt.title('Корреляция признаков с целевой переменной')
plt.tight_layout()
plt.show()

print("Топ-10 признаков по корреляции с целевой переменной:")
print(correlation_matrix.head(10))

# Выбираем топ-15 признаков по корреляции
selected_features = correlation_matrix.head(15).index.tolist()
X_selected = X[selected_features]

print(f"Выбрано {len(selected_features)} признаков")

# Стратифицированное разделение
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Простые и проверенные модели
models = {
    'Logistic Regression': LogisticRegression(C=0.1, random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=False)
}

# Оценка моделей
results = {}
print("\n" + "="*50)
print("ОЦЕНКА ПРОСТЫХ МОДЕЛЕЙ")
print("="*50)

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Метрики
    accuracy = accuracy_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'model': model
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    # Кросс-валидация
    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc')
    print(f"CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Сравнение моделей
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results],
    'AUC-ROC': [results[name]['auc_roc'] for name in results]
}).sort_values('AUC-ROC', ascending=False)

print("\nСравнение моделей:")
print(comparison_df)

# Визуализация
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(data=comparison_df, x='Accuracy', y='Model')
plt.title('Accuracy')
plt.subplot(1, 2, 2)
sns.barplot(data=comparison_df, x='AUC-ROC', y='Model')
plt.title('AUC-ROC')
plt.tight_layout()
plt.show()

# Выбор лучшей модели
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\nЛучшая модель: {best_model_name}")

# Обучение на всех данных
print("\nОбучение финальной модели на всех данных...")
final_model = best_model
final_model.fit(X_selected, y)

# Загрузка тестовых данных
print("\nЗагрузите файл test.csv с вашего компьютера")
uploaded_test = files.upload()
test_file_name = list(uploaded_test.keys())[0]
test_df = pd.read_csv(io.BytesIO(uploaded_test[test_file_name]))

# Предобработка тестовых данных
test_processed = smart_preprocessing(test_df)

# Выравнивание признаков
missing_cols = set(selected_features) - set(test_processed.columns)
for col in missing_cols:
    test_processed[col] = 0

extra_cols = set(test_processed.columns) - set(selected_features)
test_processed = test_processed.drop(columns=extra_cols, errors='ignore')

test_processed = test_processed[selected_features]

# Предсказание
test_predictions = final_model.predict(test_processed)
test_predictions_proba = final_model.predict_proba(test_processed)[:, 1]

# Анализ распределения предсказаний
print("\nАнализ предсказаний:")
print(f"Процент положительных классов: {test_predictions.mean():.2%}")
print(f"Распределение вероятностей:")
print(pd.Series(test_predictions_proba).describe())

# Создание нескольких submission файлов с разными подходами

# 1. Базовые предсказания
submission_basic = pd.DataFrame({
    'id': test_df['id'],
    'y': test_predictions
})

# 2. Предсказания с оптимизированным порогом
# Находим оптимальный порог на валидации
from sklearn.metrics import f1_score

y_val_proba = best_model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_val_pred = (y_val_proba >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Оптимальный порог: {best_threshold:.2f}")

submission_optimized = pd.DataFrame({
    'id': test_df['id'],
    'y': (test_predictions_proba >= best_threshold).astype(int)
})

# 3. Предсказания с учетом дисбаланса (увеличиваем порог для редкого класса)
submission_balanced = pd.DataFrame({
    'id': test_df['id'],
    'y': (test_predictions_proba >= 0.7).astype(int)
})

# 4. Вероятности для анализа
submission_proba = pd.DataFrame({
    'id': test_df['id'],
    'y': test_predictions_proba
})

# Сохранение файлов
submission_basic.to_csv('submission1_basic.csv', index=False)
submission_optimized.to_csv('submission2_optimized.csv', index=False)
submission_balanced.to_csv('submission3_balanced.csv', index=False)
submission_proba.to_csv('submission4_probabilities.csv', index=False)

print("\nСозданы submission файлы:")
print("1. submission1_basic.csv - базовые предсказания")
print("2. submission2_optimized.csv - с оптимизированным порогом")
print("3. submission3_balanced.csv - с учетом дисбаланса")
print("4. submission4_probabilities.csv - вероятности")

# Визуализация предсказаний
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(data=submission_basic, x='y')
plt.title('Базовые предсказания')

plt.subplot(2, 2, 2)
sns.countplot(data=submission_optimized, x='y')
plt.title('Оптимизированные предсказания')

plt.subplot(2, 2, 3)
sns.countplot(data=submission_balanced, x='y')
plt.title('Сбалансированные предсказания')

plt.subplot(2, 2, 4)
sns.histplot(data=submission_proba, x='y', bins=50)
plt.title('Распределение вероятностей')

plt.tight_layout()
plt.show()

# Анализ важности признаков для лучшей модели
if hasattr(final_model, 'feature_importances_'):
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Важность признаков')
    plt.tight_layout()
    plt.show()

# Скачивание файлов
files.download('submission1_basic.csv')
files.download('submission2_optimized.csv')
files.download('submission3_balanced.csv')
files.download('submission4_probabilities.csv')

print("\n" + "="*50)
print("РЕКОМЕНДАЦИИ:")
print("1. Сначала попробуйте submission2_optimized.csv")
print("2. Если не сработает - submission1_basic.csv") 
print("3. Анализируйте submission4_probabilities.csv для понимания модели")
print("4. Основные признаки:", selected_features[:5])
print("="*50)