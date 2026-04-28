import pandas as pd


# ─── Самописные статистические функции ────────────────────────────────────────

def meanvalue(data):
    return sum(data) / len(data)


def meandeviation(data):
    mu = meanvalue(data)
    variance = sum((x - mu) ** 2 for x in data) / len(data)
    return variance ** 0.5


def mean_absolute_error(actual, predicted):
    n = len(actual)
    total_error = sum(abs(actual[i] - predicted[i]) for i in range(n))
    return total_error / n


def correlation(dataset1, dataset2):
    n = len(dataset1)
    mu1 = sum(dataset1) / n
    mu2 = sum(dataset2) / n
    numerator = 0
    sum_sq_diff1 = 0
    sum_sq_diff2 = 0
    for i in range(n):
        diff1 = dataset1[i] - mu1
        diff2 = dataset2[i] - mu2
        numerator += diff1 * diff2
        sum_sq_diff1 += diff1 ** 2
        sum_sq_diff2 += diff2 ** 2
    denominator = (sum_sq_diff1 * sum_sq_diff2) ** 0.5
    return numerator / denominator if denominator != 0 else 0


# ─── Загрузка данных ───────────────────────────────────────────────────────────

marks = pd.read_csv('EdStatsCountry.csv')

print("Колонки в датасете:")
print(marks.columns.tolist())
print(f"\nВсего строк: {len(marks)}, колонок: {len(marks.columns)}")

# ─── Только числовые колонки (без пустых) ─────────────────────────────────────

numeric_marks = marks.select_dtypes(include=['number'])

# Убираем колонки, где менее 10 непустых значений (бесполезны для анализа)
numeric_marks = numeric_marks.dropna(axis=1, thresh=10)

cols = numeric_marks.columns.tolist()
print(f"\nЧисловые колонки ({len(cols)}):", cols)

# ─── Корреляции между всеми парами числовых колонок ───────────────────────────

print("\n" + "=" * 70)
print("КОРРЕЛЯЦИИ МЕЖДУ ПАРАМИ КОЛОНОК")
print("(считаются только по строкам без NaN в обеих колонках)")
print("=" * 70)

for i in cols:
    for j in cols:
        if i < j:
            # Берём только строки, где обе колонки не NaN
            pair = numeric_marks[[i, j]].dropna()
            if len(pair) < 2:
                print(f"{i} vs {j}: недостаточно данных")
                continue
            d1 = pair[i].tolist()
            d2 = pair[j].tolist()
            result = correlation(d1, d2)
            print(f"{i}  vs  {j}")
            print(f"  Корреляция: {result:.4f}  (по {len(pair)} наблюдениям)")

# ─── Среднее, стандартное отклонение и MAE для каждой колонки ─────────────────

print("\n" + "=" * 70)
print("СТАТИСТИКА ПО КАЖДОЙ ЧИСЛОВОЙ КОЛОНКЕ")
print("=" * 70)

for col in cols:
    clean = numeric_marks[col].dropna().tolist()
    if len(clean) < 2:
        print(f"\nКолонка: {col.upper()}")
        print("  Недостаточно данных для анализа")
        continue

    avg_val = meanvalue(clean)
    dev_val = meandeviation(clean)
    baseline_predictions = [avg_val] * len(clean)
    mae_val = mean_absolute_error(clean, baseline_predictions)

    print(f"\nКолонка: {col.upper()}")
    print(f"  - Непустых значений:   {len(clean)}")
    print(f"  - Среднее значение:    {avg_val:.4f}")
    print(f"  - Среднеквадр. откл.:  {dev_val:.4f}")
    print(f"  - Базовый MAE:         {mae_val:.4f}")
    print("-" * 50)

# ─── Краткий анализ категориальных колонок ────────────────────────────────────

print("\n" + "=" * 70)
print("ТОП-3 ЗНАЧЕНИЙ В КАТЕГОРИАЛЬНЫХ КОЛОНКАХ")
print("=" * 70)

categorical_cols = ['Region', 'Income Group', 'Lending category', 'System of trade']
for col in categorical_cols:
    if col in marks.columns:
        top = marks[col].value_counts().head(3)
        print(f"\n{col}:")
        for val, cnt in top.items():
            print(f"  {val}: {cnt} стран")