"""
Дашборд (компонент 3 методички, часть 2).

Для кого: диспетчер или логист автотранспортного предприятия.
Какие вопросы решает дашборд:
  1. Сколько проехал парк (или грузовик) и сколько прогнозирует модель?  -> график
  2. Насколько можно верить прогнозу?                                   -> метрики сверху
  3. Какая модель точнее?                                               -> таблица сравнения
  4. Почему модель дала такой прогноз?                                  -> блок «Почему модель так решила»

Данные берутся из predictions.csv (создаётся train.py) и перечитываются
каждую минуту, поэтому после нового запуска пайплайна дашборд обновляется сам.

Запуск:  streamlit run dashboard.py
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

from features import split_by_time
from interpret import (FEATURE_NAMES, SHORT_NAMES, dependence, explain_prediction, importance_builtin,
                       importance_by_period, importance_permutation, load_data, train_model)

BASELINE = "Наивный: как вчера"
ALL_TRUCKS = "Весь парк"
TARGETS = {
    "distance_km": ("Пробег за сутки", "км"),
    "fuel_l": ("Расход топлива за сутки", "л"),
}


@st.cache_data(ttl=60)
def load_predictions():
    return pd.read_csv(CONFIG["predictions_file"], parse_dates=["date"])


@st.cache_data(ttl=60)
def load_features():
    return load_data()


@st.cache_resource(ttl=600)
def get_forest(target):
    """Random Forest для объяснений (та же модель, что в train.py)."""
    data, selected = load_features()
    train, _ = split_by_time(data)
    return train_model(train, selected[target], target)


def quality_table(daily, models):
    """Качество каждой модели на выбранных данных: MAE, RMSE, ошибка в %, выигрыш у базовой."""
    rows = []
    for model in models:
        error = daily[model] - daily["actual"]
        rows.append({
            "Модель": model,
            "MAE": error.abs().mean(),
            "RMSE": float(np.sqrt((error ** 2).mean())),
        })
    table = pd.DataFrame(rows)
    table["Ошибка, % от среднего"] = table["MAE"] / daily["actual"].mean() * 100
    base_mae = table.loc[table["Модель"] == BASELINE, "MAE"].iloc[0]
    table["Точнее базовой, %"] = (base_mae - table["MAE"]) / base_mae * 100
    return table.sort_values("MAE").reset_index(drop=True)


st.set_page_config(page_title="Прогноз пробега и расхода топлива", layout="wide")
st.title("Прогноз пробега и расхода топлива автопарка")
st.caption("Сколько проедет парк и сколько топлива понадобится, и насколько можно верить прогнозу.")

if not os.path.exists(CONFIG["predictions_file"]):
    st.error("Нет файла predictions.csv. Сначала запустите: python train.py")
    st.stop()

data = load_predictions()
model_columns = [c for c in data.columns if c not in ("date", "truck_id", "target", "actual")]

# ---------------- Фильтры ----------------
st.sidebar.header("Фильтры")
target = st.sidebar.selectbox("Показатель", list(TARGETS), format_func=lambda t: TARGETS[t][0])
truck = st.sidebar.selectbox("Грузовик", [ALL_TRUCKS] + sorted(data["truck_id"].unique()))
all_dates = data["date"].dt.date
period = st.sidebar.date_input("Период", value=(all_dates.min(), all_dates.max()),
                               min_value=all_dates.min(), max_value=all_dates.max())
if len(period) != 2:
    st.info("Выберите конечную дату периода.")
    st.stop()
start, end = period

# ---------------- Данные по выбранным фильтрам ----------------
part = data[(data["target"] == target)
            & (data["date"].dt.date >= start) & (data["date"].dt.date <= end)]
if truck != ALL_TRUCKS:
    part = part[part["truck_id"] == truck]
if part.empty:
    st.warning("Нет данных за выбранный период.")
    st.stop()

daily = part.groupby("date")[["actual"] + model_columns].sum()   # для всего парка суммируем грузовики
table = quality_table(daily, model_columns)
model = st.sidebar.selectbox("Модель для прогноза", table["Модель"].tolist(), index=0)

title, unit = TARGETS[target]
row = table[table["Модель"] == model].iloc[0]

# ---------------- Насколько можно верить прогнозу ----------------
st.subheader("Насколько можно верить прогнозу")
col1, col2, col3 = st.columns(3)
col1.metric(f"Средняя ошибка, {unit} в сутки", f"{row['MAE']:.1f}")
col2.metric("Ошибка от среднего значения", f"{row['Ошибка, % от среднего']:.1f}%")
col3.metric("Точнее прогноза «как вчера»", f"на {row['Точнее базовой, %']:.0f}%")
st.write(f"В среднем прогноз модели «{model}» отличается от факта на {row['MAE']:.1f} {unit} в сутки "
         f"({row['Ошибка, % от среднего']:.1f}% от среднего значения).")

# ---------------- Факт и прогноз ----------------
st.subheader(f"{title}: факт и прогноз ({truck})")
chart = pd.DataFrame({"Факт": daily["actual"], f"Прогноз: {model}": daily[model]})
chart.index = chart.index.strftime("%Y-%m-%d")    # подписи по оси: даты, а не время
st.line_chart(chart, color=["#1f77b4", "#ff7f0e"])   # факт синим, прогноз оранжевым

# ---------------- Какая модель точнее ----------------
st.subheader("Какая модель точнее")
st.dataframe(table.round(1), hide_index=True)
st.caption(f"MAE и RMSE в {unit}. Чем меньше, тем точнее. Проверка на последних 30 днях, "
           "которых модели не видели при обучении.")


# ---------------- Почему модель так решила ----------------
st.header("Почему модель так решила")
if not (os.path.exists(CONFIG["features_file"]) and os.path.exists(CONFIG["selected_features_file"])):
    st.info("Для объяснений нужны файлы features.csv и selected_features.json. Запустите: python features.py")
    st.stop()

feature_data, selected = load_features()
train_part, valid_part = split_by_time(feature_data)
features = selected[target]
forest = get_forest(target)
st.caption(f"Объясняем модель Random Forest для показателя «{title}». "
           f"Признаков в модели: {len(features)}.")

tab1, tab2, tab3, tab4 = st.tabs(["Что важно", "Объяснить прогноз", "Зависимость от признака",
                                  "Стабильность по периодам"])

with tab1:
    st.write("Какие признаки сильнее всего влияют на прогноз. Два независимых способа "
             "должны давать похожую картину.")
    left, right = st.columns(2)
    left.caption("Встроенная важность: насколько признак помогает деревьям леса")
    left.bar_chart(importance_builtin(forest, features).rename(index=SHORT_NAMES))
    right.caption(f"Перестановочная важность: на сколько вырастет ошибка ({unit}), если перемешать признак")
    right.bar_chart(importance_permutation(forest, valid_part, features, target).rename(index=SHORT_NAMES))
    st.caption("Сокращения: «-7 дн.» — неделю назад, «7 дн.» и «ср.» — среднее за 7 дней.")

with tab2:
    st.write("Из чего сложился прогноз для выбранного грузовика и дня.")
    explain_truck = st.selectbox("Грузовик для объяснения", sorted(feature_data["truck_id"].unique()))
    truck_dates = sorted(feature_data.loc[feature_data["truck_id"] == explain_truck, "date"].dt.date.unique())
    day = st.selectbox("Дата", truck_dates, index=len(truck_dates) - 1)
    row = feature_data[(feature_data["truck_id"] == explain_truck) & (feature_data["date"].dt.date == day)]
    prediction, contributions = explain_prediction(forest, row, train_part, features)
    st.write(f"Прогноз: {prediction:.0f} {unit}. Фактическое значение: {row[target].iloc[0]:.0f} {unit}.")
    details = pd.DataFrame({
        "Значение признака": row[features].iloc[0],
        f"Вклад в прогноз, {unit}": contributions,
    }).loc[contributions.index].rename(index=FEATURE_NAMES)
    st.bar_chart(contributions.rename(index=SHORT_NAMES))
    st.dataframe(details.round(1))
    top_name = FEATURE_NAMES.get(contributions.index[0], contributions.index[0])
    top_value = contributions.iloc[0]
    st.write(f"Сильнее всего повлиял признак «{top_name}»: он "
             f"{'увеличил' if top_value > 0 else 'уменьшил'} прогноз примерно на {abs(top_value):.0f} {unit}.")
    st.caption("Вклад показывает, насколько изменился бы прогноз, если заменить признак на типичное "
               "значение (медиану). Это направление и сила влияния, вклады не обязаны точно "
               "складываться в итоговый прогноз.")

with tab3:
    st.write("Как меняется прогноз при изменении одного признака, когда остальные не трогаем.")
    feature = st.selectbox("Признак", features, format_func=lambda f: FEATURE_NAMES.get(f, f))
    curve = dependence(forest, feature_data, feature, features)
    if len(curve) <= 7:
        st.bar_chart(curve)
    else:
        st.line_chart(curve)
    st.caption(f"По горизонтали значение признака, по вертикали средний прогноз ({unit}).")

with tab4:
    st.write("Одинаково ли модель опирается на признаки в разные периоды. "
             "Если картина стабильна, модель держится на устойчивых закономерностях.")
    period_table, rank_corr = importance_by_period(train_part, features, target)
    st.dataframe(period_table.rename(index=FEATURE_NAMES).round(3))
    st.write(f"Совпадение порядка важности в двух периодах (корреляция Спирмена): {rank_corr:.2f}. "
             "Чем ближе к 1, тем стабильнее.")

