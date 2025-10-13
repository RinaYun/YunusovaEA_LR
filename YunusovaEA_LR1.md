import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (18, 10)
data = pd.read_csv('/content/sample_data/adult_data.csv')
print(data)
male_count = data[data['sex'] == 'Male'].shape[0]
female_count = data[data['sex'] == 'Female'].shape[0]

print(f"Количество мужчин: {male_count}")
print(f"Количество женщин: {female_count}")
male_data = data[data['sex'] == 'Male']
average_age_male = male_data['age'].mean()

print(f"Средний возраст мужчин: {average_age_male:.2f} лет")
total_people = data.shape[0]

# количествj людей из Тайваня
taiwan_people = data[data['native-country'] == 'Taiwan'].shape[0]

# процент
taiwan_percentage = (taiwan_people / total_people) * 100

print(f"Общее количество людей в датасете: {total_people}")
print(f"Количество людей из Тайваня: {taiwan_people}")
print(f"Процент людей из Тайваня: {taiwan_percentage:.2f}%")
high_income_df = data[data['salary'] == '>50K']
average_age_high_income = high_income_df['age'].mean()

print(f"Средний возраст людей с зарплатой >50K: {average_age_high_income:.2f} лет")
high_income_data = data[data['salary'] == '>50K']
std_age_high_income = high_income_df['age'].std()

print(f"Стандартное отклонение возраста людей с зарплатой >50K: {std_age_high_income:.2f} лет")
high_income_df = data[data['salary'] == '>50K']

# образования уровня Bachelors
bachelors_plus = ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']

# все ли люди с >50K имеют образование Bachelors+
all_bachelors_plus = high_income_data['education'].isin(bachelors_plus).all()

# люди с разными уровнями образования среди тех кто зарабатывает >50K
education_counts = high_income_data['education'].value_counts()

print(f"Все люди с зарплатой >50K имеют образование Bachelors+: {all_bachelors_plus}")
print("\nРаспределение образования среди людей с зарплатой >50K:")
print(education_counts)

# процент людей с образованием ниже Bachelors среди тех кто зарабатывает >50K
non_bachelors_count = high_income_data[~high_income_data['education'].isin(bachelors_plus)].shape[0]
total_high_income = high_income_data.shape[0]
percentage_non_bachelors = (non_bachelors_count / total_high_income) * 100

print(f"\nКоличество людей с зарплатой >50K без образования Bachelors+: {non_bachelors_count}")
print(f"Общее количество людей с зарплатой >50K: {total_high_income}")
print(f"Процент людей с зарплатой >50K без образования Bachelors+: {percentage_non_bachelors:.2f}%")
filtered_data = data[
    (data['hours-per-week'] == 40) &
    (data['native-country'] == 'United-States') &
    (data['salary'] == '<=50K')
]

# количество людей
count_people = filtered_data.shape[0]

print(f"Количество людей, работающих 40 часов, граждан США и зарабатывающих <=50K: {count_people}")
max_hours = data['hours-per-week'].max()

# Фильтруем людей, которые работают максимальное количество часов и зарабатывают <=50K
max_hours_low_income = data[
    (data['hours-per-week'] == max_hours) &
    (data['salary'] == '<=50K')
]

# Подсчитываем количество
count_people = max_hours_low_income.shape[0]

print(f"Максимальное количество часов в неделю: {max_hours}")
print(f"Количество людей, работающих {max_hours} часов в неделю и зарабатывающих <=50K: {count_people}")
import matplotlib.pyplot as plt
import seaborn as sns

# стили для лучшего отображения
plt.style.use('default')
sns.set_palette("husl")

# Подсчёт количества людей по уровням образования
education_counts = data['education'].value_counts()

# Создание графика
plt.figure(figsize=(14, 8))
bars = plt.bar(education_counts.index, education_counts.values, color='skyblue', edgecolor='navy', alpha=0.7)

# Настройка внешнего вида
plt.title('Распределение людей по уровням образования в датасете', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Уровень образования', fontsize=12)
plt.ylabel('Количество людей', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Добавление значений на столбцы
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Вывод ответа
most_common_education = education_counts.index[0]
most_common_count = education_counts.values[0]

print("=" * 60)
print(f"ОТВЕТ: Больше всего в датасете представлены люди с образованием '{most_common_education}'")
print(f"Количество: {most_common_count} человек")
print("=" * 60)
print("\nПолное распределение по образованиям:")
for i, (edu, count) in enumerate(education_counts.items(), 1):
    print(f"{i:2d}. {edu:<15} - {count:4d} человек")
plt.figure(figsize=(14, 8))

# Построение гистограммы с стандартными настройками seaborn
hist = sns.histplot(data=data, x='age', bins='auto', kde=True, color='lightcoral', alpha=0.7)

plt.title('Распределение возрастов в датасете', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Возраст', fontsize=12)
plt.ylabel('Количество людей', fontsize=12)
plt.grid(alpha=0.3)

# Получение данных о столбцах гистограммы
counts, bin_edges = np.histogram(data['age'], bins='auto')

# Подсчёт отрезков с более чем 1600 значениями
segments_above_1600 = np.sum(counts > 1600)

print("=" * 70)
print("Анализ расапределенич возрастов:")
print("=" * 70)
print(f"Общее количество возрастных отрезков: {len(counts)}")
print(f"Отрезков с более чем 1600 значений: {segments_above_1600}")
print(f"Максимальное количество значений в одном отрезке: {counts.max()}")
print(f"Минимальное количество значений в одном отрезке: {counts.min()}")
print(f"Среднее количество значений в отрезке: {counts.mean():.1f}")

print("\nДетальная информация по отрезкам с более чем 1600 значений:")
for i, (count, left_edge, right_edge) in enumerate(zip(counts, bin_edges[:-1], bin_edges[1:])):
    if count > 1600:
        print(f"Отрезок {i+1}: возраст {left_edge:.1f}-{right_edge:.1f} лет - {count} человек")

plt.tight_layout()
plt.show()