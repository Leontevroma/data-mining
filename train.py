"""
Модельное ядро (компонент 3 методички, часть 1): обучение и сравнение моделей.

Сравниваем для каждой цели (пробег distance_km и расход топлива fuel_l):
  - две базовые модели: наивный прогноз "как вчера" и "как неделю назад";
  - Ridge (линейная регрессия);
  - Random Forest (ансамбль деревьев, усреднение);
  - Gradient Boosting (ансамбль деревьев, последовательное улучшение).

Схема проверки одна для всех моделей и учитывает порядок времени:
  1. Кросс-проверка на обучающей части: окно обучения растёт, проверка всегда
     на более поздних днях (TimeSeriesSplit по датам).
  2. Итоговая проверка на последних valid_days днях, которых модель не видела.

Метрики: MAE (средняя ошибка в км или литрах) и RMSE (сильнее штрафует большие промахи).
Для пользователя ошибка переводится в проценты от типичного суточного значения.

Запуск:  python train.py   (после python features.py)
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import CONFIG, TARGETS, split_by_time

NAIVE_DAY = "Наивный: как вчера"
NAIVE_WEEK = "Наивный: как неделю назад"
BASELINE = NAIVE_DAY          # главная базовая модель, с ней сравниваем остальные


def new_models():
    """Модели с фиксированным зерном (воспроизводимость). None = наивная модель."""
    seed = CONFIG["random_seed"]
    return {
        NAIVE_DAY: None,
        NAIVE_WEEK: None,
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
    }


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def predict(name, model, train_part, test_part, target, features):
    """Прогноз модели. Наивные модели берут готовый лаг, остальные обучаются."""
    if name == NAIVE_DAY:
        return test_part[f"{target}_lag1"].to_numpy()
    if name == NAIVE_WEEK:
        return test_part[f"{target}_lag7"].to_numpy()
    fitted = clone(model)
    fitted.fit(train_part[features], train_part[target])
    return fitted.predict(test_part[features])


def time_folds(train, n_splits=4):
    """Делим по ДАТАМ: обучение всегда раньше проверки, перемешивания нет."""
    dates = np.array(sorted(train["date"].unique()))
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(dates):
        train_part = train[train["date"].isin(dates[train_idx])]
        test_part = train[train["date"].isin(dates[test_idx])]
        yield train_part, test_part


def evaluate_target(target, train, valid, features):
    rows = []
    predictions = valid[["date", "truck_id"]].copy()
    predictions["target"] = target
    predictions["actual"] = valid[target].to_numpy()

    for name, model in new_models().items():
        # 1. Кросс-проверка во времени на обучающей части
        cv_mae, cv_rmse = [], []
        for train_part, test_part in time_folds(train):
            pred = predict(name, model, train_part, test_part, target, features)
            cv_mae.append(mean_absolute_error(test_part[target], pred))
            cv_rmse.append(rmse(test_part[target], pred))

        # 2. Итоговая проверка на последних днях
        pred = predict(name, model, train, valid, target, features)
        predictions[name] = pred
        rows.append({
            "Цель": target,
            "Модель": name,
            "MAE (кросс-проверка)": np.mean(cv_mae),
            "RMSE (кросс-проверка)": np.mean(cv_rmse),
            "MAE (проверка)": mean_absolute_error(valid[target], pred),
            "RMSE (проверка)": rmse(valid[target], pred),
        })

    results = pd.DataFrame(rows)
    # Понятные пользователю величины
    typical = valid[target].mean()
    results["Ошибка, % от среднего"] = results["MAE (проверка)"] / typical * 100
    base_mae = results.loc[results["Модель"] == BASELINE, "MAE (проверка)"].iloc[0]
    results["Точнее базовой, %"] = (base_mae - results["MAE (проверка)"]) / base_mae * 100
    return results, predictions


if __name__ == "__main__":
    data = pd.read_csv(CONFIG["features_file"], parse_dates=["date"])
    with open(CONFIG["selected_features_file"], "r", encoding="utf-8") as f:
        selected = json.load(f)

    train, valid = split_by_time(data)
    print(f"Обучение: {len(train)} строк, проверка: {len(valid)} строк "
          f"(с {valid['date'].min().date()} по {valid['date'].max().date()})")

    all_results, all_predictions = [], []
    for target in TARGETS:
        results, predictions = evaluate_target(target, train, valid, selected[target])
        all_results.append(results)
        all_predictions.append(predictions)

        unit = "км" if target == "distance_km" else "л"
        print(f"\n=== {target} (признаков в моделях: {len(selected[target])}, MAE в {unit}) ===")
        print(results.drop(columns="Цель").round(1).to_string(index=False))

    pd.concat(all_results).to_csv(CONFIG["results_file"], index=False, encoding="utf-8")
    pd.concat(all_predictions).to_csv(CONFIG["predictions_file"], index=False, encoding="utf-8")
    print(f"\nСохранено: {CONFIG['results_file']}, {CONFIG['predictions_file']}")
