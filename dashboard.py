"""
Дашборд (компонент 3 методички, часть 2).

Для кого: диспетчер или логист автотранспортного предприятия.
Какие вопросы решает дашборд:
  1. Сколько проехал парк (или грузовик) и сколько прогнозирует модель?  -> график
  2. Насколько можно верить прогнозу?                                   -> метрики сверху
  3. Какая модель точнее?                                               -> таблица сравнения

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

BASELINE = "Наивный: как вчера"
ALL_TRUCKS = "Весь парк"
TARGETS = {
    "distance_km": ("Пробег за сутки", "км"),
    "fuel_l": ("Расход топлива за сутки", "л"),
}


@st.cache_data(ttl=60)
def load_predictions():
    return pd.read_csv(CONFIG["predictions_file"], parse_dates=["date"])


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
