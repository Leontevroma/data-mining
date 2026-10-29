"""
Обработка данных и формирование признаков (компонент 2 методички).

Что делает программа:
  1. читает сырые данные (JSON с датчиками и CSV с журналами событий);
  2. очищает их;
  3. собирает суточную таблицу: одна строка = один грузовик за один день;
  4. строит признаки (лаги, окна, календарь, кодирование, признаки из журналов);
  5. доказывает, что в признаках нет утечки целевой переменной;
  6. делит данные по времени и отбирает признаки;
  7. сохраняет готовую витрину в features.csv.

Целевые переменные (что прогнозируем): distance_km (пробег за сутки)
и fuel_l (расход топлива за сутки).

Запуск:  python features.py

Весь код расчёта признаков лежит в функции add_features(). Её же будем
вызывать и при обучении, и при построении прогноза (единый код).
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

TARGETS = ["distance_km", "fuel_l"]          # что прогнозируем
MIN_IMPORTANCE = 0.01                         # признак слабее 1% отбрасываем

# Столбцы суточной таблицы, от которых берём лаги и скользящее среднее
LAG_COLUMNS = ["distance_km", "fuel_l", "mean_speed", "max_speed", "n_alerts"]

# Признаки из журналов: слово в тексте -> название столбца
KEYWORDS = {
    "n_brake": "торможение",
    "n_speeding": "превышение",
    "n_load": "нагрузк",
    "n_low_fuel": "низкий уровень топлива",
    "n_overheat": "перегрев",
}

# Полный список признаков (имена столбцов), которые подаём в модель
FEATURE_COLUMNS = (
    [f"{c}_lag1" for c in LAG_COLUMNS]                 # значение за вчера
    + [f"{c}_mean7" for c in LAG_COLUMNS]              # среднее за 7 прошлых дней
    + [f"{c}_lag1" for c in KEYWORDS]                  # события в журнале за вчера
    + [f"{c}_lag7" for c in TARGETS]                   # значение неделю назад
    + ["day_of_week", "is_weekend", "month"]           # календарные
    + [f"truck_{t}" for t in CONFIG["trucks"]]         # номер грузовика
)


# ---------------------------------------------------------------
# 1. Чтение и очистка (приведение к единому виду)
# ---------------------------------------------------------------
def load_raw():
    with open(CONFIG["coords_file"], "r", encoding="utf-8") as f:
        sensors = pd.DataFrame(json.load(f))
    logs = pd.read_csv(CONFIG["logs_file"], encoding="utf-8")
    return sensors, logs


def clean_sensors(sensors):
    """
    Приводим датчики к единому виду, убираем пропуски и невозможные значения.
    Аномалии из генератора (скорость 85-110, резкое торможение) НЕ удаляем:
    это реальные события, они влияют на расход и несут информацию.
    Удаляем только физически невозможное: скорость < 0 или > max_speed_kmh,
    отрицательный путь или расход.
    """
    df = sensors.copy()
    n_before = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["speed_kmh", "distance_km", "fuel_l"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["timestamp", "truck_id"])
    df = df.dropna(subset=["timestamp", "truck_id", "speed_kmh", "distance_km", "fuel_l"])
    impossible = (
        (df["speed_kmh"] < 0) | (df["speed_kmh"] > CONFIG["max_speed_kmh"])
        | (df["distance_km"] < 0) | (df["fuel_l"] < 0)
    )
    df = df[~impossible]
    print(f"Очистка датчиков: было строк {n_before}, осталось {len(df)}")
    return df


def clean_logs(logs):
    df = logs.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["telemetry_log"] = df["telemetry_log"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["timestamp", "device_id"])
    df = df.dropna(subset=["timestamp", "device_id"])
    print(f"Очистка журналов: осталось строк {len(df)}")
    return df


# ---------------------------------------------------------------
# 2. Суточная таблица: одна строка = грузовик + день
# ---------------------------------------------------------------
def make_daily(sensors, logs):
    # Датчики -> суммы и средние за сутки
    sensors = sensors.copy()
    sensors["date"] = sensors["timestamp"].dt.normalize()
    daily = sensors.groupby(["date", "truck_id"]).agg(
        distance_km=("distance_km", "sum"),
        fuel_l=("fuel_l", "sum"),
        mean_speed=("speed_kmh", "mean"),
        max_speed=("speed_kmh", "max"),
        n_records=("speed_kmh", "count"),
    ).reset_index()

    # Оставляем только полные сутки (например, 144 записи при шаге 10 минут),
    # чтобы неполный день не выглядел как "мало проехал"
    records_per_day = 24 * 60 // CONFIG["step_minutes"]
    daily = daily[daily["n_records"] == records_per_day].drop(columns="n_records")

    # Журналы -> какие события случились за сутки (извлечение признаков из текста)
    device_to_truck = sensors[["device_id", "truck_id"]].drop_duplicates()
    logs = logs.merge(device_to_truck, on="device_id", how="left")
    logs = logs.dropna(subset=["truck_id"]).copy()
    logs["date"] = logs["timestamp"].dt.normalize()
    text = logs["telemetry_log"].str.lower()
    logs["n_alerts"] = logs["telemetry_log"].str.endswith("!").astype(int)
    for name, word in KEYWORDS.items():
        logs[name] = text.str.contains(word, regex=False).astype(int)
    log_cols = ["n_alerts"] + list(KEYWORDS)
    log_daily = logs.groupby(["date", "truck_id"])[log_cols].sum().reset_index()

    daily = daily.merge(log_daily, on=["date", "truck_id"], how="left")
    daily[log_cols] = daily[log_cols].fillna(0)   # нет записей = событий не было

    # Делаем непрерывный календарь: пропущенный день становится строкой с NaN.
    # Так сдвиг на "1 день назад" всегда означает именно вчера, а не "предыдущую строку".
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([dates, CONFIG["trucks"]], names=["date", "truck_id"])
    daily = daily.set_index(["date", "truck_id"]).reindex(grid).reset_index()
    return daily


# ---------------------------------------------------------------
# 3. Признаки (единый код для обучения и для прогноза)
# ---------------------------------------------------------------
def add_features(daily):
    """
    Все признаки считаются ТОЛЬКО из прошлых дней (shift), поэтому
    значения за сам день прогноза в признаки не попадают.
    При прогнозе на завтра в таблицу добавляется строка завтрашнего дня
    с пустыми значениями, а признаки считаются этой же функцией.
    """
    df = daily.sort_values(["truck_id", "date"]).reset_index(drop=True)
    g = df.groupby("truck_id")

    # Лаговые и оконные признаки
    for col in LAG_COLUMNS:
        df[f"{col}_lag1"] = g[col].shift(1)                                   # вчера
        df[f"{col}_mean7"] = g[col].transform(lambda s: s.shift(1).rolling(7).mean())
    for col in KEYWORDS:
        df[f"{col}_lag1"] = g[col].shift(1)
    for col in TARGETS:
        df[f"{col}_lag7"] = g[col].shift(7)                                   # неделю назад

    # Календарные признаки (дата известна заранее, утечки нет)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month

    # Кодирование грузовика: one-hot. Грузовиков всего 3, порядка между ними нет,
    # поэтому числа 1, 2, 3 ввели бы ложный порядок; столбцов получается мало.
    for t in CONFIG["trucks"]:
        df[f"truck_{t}"] = (df["truck_id"] == t).astype(int)
    return df


# ---------------------------------------------------------------
# 4. Проверка отсутствия утечки целевой переменной
# ---------------------------------------------------------------
def check_no_leakage(daily):
    """
    Идея проверки: портим данные начиная с дня X (умножаем на 10 и прибавляем 1000).
    Если признаки дня X не изменились, значит они не используют данные
    самого дня X и более поздних дней, то есть будущее в них не попало.
    Для контроля проверяем, что признаки следующего дня X+1 изменились
    (он использует день X), иначе проверка ничего не доказывает.
    """
    dates = sorted(daily["date"].dropna().unique())
    if len(dates) < 15:
        print("Проверка утечки пропущена: слишком мало дней данных")
        return
    test_date = pd.Timestamp(dates[len(dates) // 2])

    broken = daily.copy()
    value_cols = [c for c in daily.columns if c not in ("date", "truck_id")]
    future = broken["date"] >= test_date
    broken.loc[future, value_cols] = broken.loc[future, value_cols] * 10 + 1000

    a = add_features(daily)
    b = add_features(broken)

    def features_on(df, date):
        return df.loc[df["date"] == date, FEATURE_COLUMNS].to_numpy(dtype=float)

    same_day = np.allclose(features_on(a, test_date), features_on(b, test_date), equal_nan=True)
    next_day = test_date + pd.Timedelta(days=1)
    next_changed = not np.allclose(features_on(a, next_day), features_on(b, next_day), equal_nan=True)

    assert same_day, "УТЕЧКА: признаки дня зависят от данных этого же дня или будущего!"
    assert next_changed, "Проверка утечки неинформативна: признаки не реагируют на прошлое"
    print(f"Проверка утечки пройдена (контрольная дата {test_date.date()}): "
          f"признаки дня не зависят от этого дня и будущего")


# ---------------------------------------------------------------
# 5. Итоговая витрина и разбиение по времени
# ---------------------------------------------------------------
def build_dataset():
    sensors, logs = load_raw()
    sensors = clean_sensors(sensors)
    logs = clean_logs(logs)
    daily = make_daily(sensors, logs)
    check_no_leakage(daily)
    data = add_features(daily)
    # Первые 7 дней и дни рядом с пропусками не имеют полных признаков - убираем
    data = data.dropna(subset=FEATURE_COLUMNS + TARGETS)
    return data[["date", "truck_id"] + TARGETS + FEATURE_COLUMNS].reset_index(drop=True)


def split_by_time(data):
    """Обучение: старые дни. Проверка: последние valid_days дней. Перемешивания нет."""
    border = data["date"].max() - pd.Timedelta(days=CONFIG["valid_days"] - 1)
    train = data[data["date"] < border]
    valid = data[data["date"] >= border]
    return train, valid


# ---------------------------------------------------------------
# 6. Отбор признаков и оценка их влияния на качество
# ---------------------------------------------------------------
def select_features(train, valid, target):
    """
    Отбор делаем ТОЛЬКО по обучающей части, проверочная нужна лишь для оценки.
    Шаг 1: убираем константы и почти одинаковые признаки (корреляция > 0.95).
    Шаг 2: оставляем признаки, чья важность в случайном лесу не ниже MIN_IMPORTANCE.
    Затем сравниваем ошибку на проверочной выборке: все признаки и отобранные.
    """
    seed = CONFIG["random_seed"]

    candidates = [c for c in FEATURE_COLUMNS if train[c].nunique() > 1]
    corr = train[candidates].corr().abs()
    dropped = set()
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if a not in dropped and b not in dropped and corr.loc[a, b] > 0.95:
                dropped.add(b)
    candidates = [c for c in candidates if c not in dropped]

    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(train[candidates], train[target])
    importance = pd.Series(model.feature_importances_, index=candidates).sort_values(ascending=False)
    selected = list(importance[importance >= MIN_IMPORTANCE].index)

    mae_all = mean_absolute_error(valid[target], model.predict(valid[candidates]))
    model_sel = RandomForestRegressor(n_estimators=100, random_state=seed)
    model_sel.fit(train[selected], train[target])
    mae_sel = mean_absolute_error(valid[target], model_sel.predict(valid[selected]))
    mae_naive = mean_absolute_error(valid[target], valid[f"{target}_lag1"])   # "как вчера"

    print(f"\n--- Цель: {target} ---")
    print(f"Признаков всего: {len(FEATURE_COLUMNS)}, после очистки от дублей: {len(candidates)}, "
          f"отобрано: {len(selected)}")
    print("Важность признаков (топ-8):")
    print(importance.head(8).round(3).to_string())
    print(f"MAE на проверке, все признаки:       {mae_all:.3f}")
    print(f"MAE на проверке, отобранные признаки: {mae_sel:.3f}")
    print(f"MAE наивного прогноза (как вчера):    {mae_naive:.3f}")
    return selected


if __name__ == "__main__":
    data = build_dataset()
    data.to_csv(CONFIG["features_file"], index=False, encoding="utf-8")

    train, valid = split_by_time(data)
    print(f"\nВитрина признаков: {len(data)} строк, {len(FEATURE_COLUMNS)} признаков, "
          f"сохранена в {CONFIG['features_file']}")
    print(f"Обучение: {len(train)} строк (до {train['date'].max().date()}), "
          f"проверка: {len(valid)} строк (с {valid['date'].min().date()})")

    selected = {target: select_features(train, valid, target) for target in TARGETS}
    with open(CONFIG["selected_features_file"], "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=4)
