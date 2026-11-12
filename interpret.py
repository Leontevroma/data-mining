"""
Интерпретируемость модели (компонент по выбору): почему модель дала такой прогноз.

Объясняем Random Forest (ту же модель, что в train.py: те же признаки и зерно).
Четыре части, как требует методичка:
  1. Важность признаков двумя независимыми способами:
     - встроенная в лес (насколько признак помогает деревьям);
     - перестановочная (на сколько растёт ошибка, если перемешать признак).
  2. Объяснение одного прогноза: вклад каждого признака.
     Вклад = прогноз минус прогноз, в котором признак заменён на типичное значение (медиану).
  3. Зависимость прогноза от одного признака.
  4. Сравнение важности признаков в двух периодах.

Запуск:  python interpret.py   (после python features.py)
Эти же функции использует дашборд.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from features import CONFIG, TARGETS, split_by_time

# Понятные названия признаков для людей
FEATURE_NAMES = {
    "distance_km_lag1": "пробег вчера",
    "distance_km_lag7": "пробег неделю назад",
    "distance_km_mean7": "средний пробег за 7 дней",
    "fuel_l_lag1": "расход вчера",
    "fuel_l_lag7": "расход неделю назад",
    "fuel_l_mean7": "средний расход за 7 дней",
    "mean_speed_lag1": "средняя скорость вчера",
    "mean_speed_mean7": "средняя скорость за 7 дней",
    "max_speed_lag1": "макс. скорость вчера",
    "max_speed_mean7": "макс. скорость за 7 дней",
    "n_alerts_lag1": "тревог вчера",
    "n_alerts_mean7": "тревог в день за 7 дней",
    "day_of_week": "день недели (0 = пн)",
    "is_weekend": "выходной",
    "month": "месяц",
}

# Короткие названия для подписей на графиках (длинные обрезаются)
SHORT_NAMES = {
    "distance_km_lag1": "пробег вчера",
    "distance_km_lag7": "пробег -7 дн.",
    "distance_km_mean7": "ср. пробег",
    "fuel_l_lag1": "расход вчера",
    "fuel_l_lag7": "расход -7 дн.",
    "fuel_l_mean7": "ср. расход",
    "mean_speed_lag1": "скор. вчера",
    "mean_speed_mean7": "ср. скорость",
    "max_speed_lag1": "макс. вчера",
    "max_speed_mean7": "макс. 7 дн.",
    "n_alerts_lag1": "тревоги вчера",
    "n_alerts_mean7": "тревоги 7 дн.",
    "day_of_week": "день недели",
    "is_weekend": "выходной",
    "month": "месяц",
}


def load_data():
    data = pd.read_csv(CONFIG["features_file"], parse_dates=["date"])
    with open(CONFIG["selected_features_file"], "r", encoding="utf-8") as f:
        selected = json.load(f)
    return data, selected


def train_model(train, features, target):
    """Та же модель, что в train.py: Random Forest, 200 деревьев, зерно из конфига."""
    model = RandomForestRegressor(n_estimators=200, random_state=CONFIG["random_seed"])
    model.fit(train[features], train[target])
    return model


# ---------------- 1. Важность признаков двумя способами ----------------
def importance_builtin(model, features):
    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)


def importance_permutation(model, valid, features, target):
    """Мешаем значения одного признака и смотрим, на сколько выросла ошибка (MAE)."""
    result = permutation_importance(
        model, valid[features], valid[target], n_repeats=10,
        random_state=CONFIG["random_seed"], scoring="neg_mean_absolute_error",
    )
    return pd.Series(result.importances_mean, index=features).sort_values(ascending=False)


# ---------------- 2. Объяснение одного прогноза ----------------
def explain_prediction(model, row, train, features):
    """
    row: одна строка таблицы признаков (DataFrame). Возвращает прогноз и вклад признаков.
    Положительный вклад: признак увеличил прогноз, отрицательный: уменьшил.
    """
    x = row[features]
    prediction = float(model.predict(x)[0])
    typical = train[features].median()
    contributions = {}
    for feature in features:
        changed = x.copy()
        changed[feature] = typical[feature]
        contributions[feature] = prediction - float(model.predict(changed)[0])
    result = pd.Series(contributions)
    return prediction, result.reindex(result.abs().sort_values(ascending=False).index)


# ---------------- 3. Зависимость прогноза от признака ----------------
def dependence(model, data, feature, features):
    """Подставляем значение признака во все строки и усредняем прогноз."""
    values = np.sort(data[feature].unique())
    if len(values) > 20:
        values = np.unique(np.quantile(data[feature], np.linspace(0.02, 0.98, 20)))
    averages = []
    for value in values:
        x = data[features].copy()
        x[feature] = value
        averages.append(float(model.predict(x).mean()))
    return pd.Series(averages, index=values)


# ---------------- 4. Важность в разных периодах ----------------
def importance_by_period(train, features, target):
    """Обучаем модель отдельно на первой и второй половине обучающих дней."""
    dates = sorted(train["date"].unique())
    middle = dates[len(dates) // 2]
    first = train[train["date"] < middle]
    second = train[train["date"] >= middle]
    day = pd.Timestamp(middle).date()
    table = pd.DataFrame({
        f"период 1 (до {day})": importance_builtin(train_model(first, features, target), features),
        f"период 2 (с {day})": importance_builtin(train_model(second, features, target), features),
    })
    corr = table.iloc[:, 0].corr(table.iloc[:, 1], method="spearman")
    return table.sort_values(table.columns[1], ascending=False), float(corr)


if __name__ == "__main__":
    data, selected = load_data()
    train, valid = split_by_time(data)
    rows = []
    for target in TARGETS:
        features = selected[target]
        model = train_model(train, features, target)
        builtin = importance_builtin(model, features)
        permutation = importance_permutation(model, valid, features, target)
        table, corr = importance_by_period(train, features, target)

        print(f"\n=== {target} ===")
        print("Важность (встроенная в лес):")
        print(builtin.round(3).to_string())
        print("Важность (перестановочная, рост MAE):")
        print(permutation.round(1).to_string())
        print("Важность в двух периодах:")
        print(table.round(3).to_string())
        print(f"Совпадение порядка важности (корреляция Спирмена): {corr:.2f}")

        for name, series in (("встроенная", builtin), ("перестановочная", permutation)):
            for feature, value in series.items():
                rows.append({"Цель": target, "Метод": name, "Признак": feature, "Важность": value})

    pd.DataFrame(rows).to_csv(CONFIG["importance_file"], index=False, encoding="utf-8")
    print(f"\nСохранено: {CONFIG['importance_file']}")
