"""
Сбор данных телеметрии автопарка (вариант 25: расход топлива и пробег парка).

Источник 1: telemetry_coords.json - датчики (координаты, скорость, пробег, топливо)
Источник 2: telemetry_logs.csv    - текстовые журналы событий (слабоструктурированные)

Запуск:
    python etl.py        # догрузить 1 день новых данных
    python etl.py 180    # догрузить 180 дней (первичная загрузка истории)
"""
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------
# 1. Конфигурация (из файла config.json, а не из кода)
# ---------------------------------------------------------------
def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()

# Три грузовика: начальные координаты и "фаза" маршрута
TRUCKS = {
    "TRUCK_001": {"lat": 55.7558, "lon": 37.6173, "device": "GPS_TRUCK_001", "phase": 0.0},
    "TRUCK_002": {"lat": 55.5701, "lon": 37.5612, "device": "GPS_TRUCK_002", "phase": 2.0},
    "TRUCK_003": {"lat": 55.8011, "lon": 37.4523, "device": "GPS_TRUCK_003", "phase": 4.0},
}

NORMAL_TEXTS = [
    "Сбоев не обнаружено",
    "Система работает штатно",
    "Режим круиз-контроля активирован",
    "Заправка завершена",
    "Остановка на разгрузку",
    "Плановая проверка пройдена",
]
ANOMALY_TEXTS = [
    "Резкое торможение зафиксировано!",
    "Превышение скоростного режима!",
    "Повышенная нагрузка на двигатель!",
    "Низкий уровень топлива!",
    "Перегрев двигателя!",
]


# ---------------------------------------------------------------
# 2. Журнал загрузок
# ---------------------------------------------------------------
def write_log(level, message):
    """Записывает строку в журнал загрузок: время, уровень, сообщение."""
    now = datetime.now().strftime(TIME_FORMAT)
    with open(CONFIG["manager_log"], "a", encoding="utf-8") as f:
        f.write(f"[{now}] [{level}] {message}\n")


# ---------------------------------------------------------------
# 3. Генерация данных (одинаковый результат при повторе)
# ---------------------------------------------------------------
def make_records(truck_id, current_time):
    """
    Создаёт запись для одного грузовика на один момент времени.
    Случайные числа зависят ТОЛЬКО от грузовика и времени, поэтому
    при повторном запуске за тот же период данные получатся такими же
    (это и есть идемпотентность).
    """
    truck = TRUCKS[truck_id]
    time_str = current_time.strftime(TIME_FORMAT)
    rng = random.Random(f"{truck_id}-{time_str}")

    # Номер шага от начала наблюдений - по нему считаем положение
    start = datetime.strptime(CONFIG["default_start_time"], TIME_FORMAT)
    step_no = int((current_time - start).total_seconds() / 60 / CONFIG["step_minutes"])
    lat = truck["lat"] + 0.02 * math.sin(step_no / 30 + truck["phase"])
    lon = truck["lon"] + 0.02 * math.cos(step_no / 30 + truck["phase"])

    # Обычный режим или аномалия (5%)
    if rng.random() < 0.05:
        log_text = rng.choice(ANOMALY_TEXTS)
        if "Превышение" in log_text:
            speed = rng.randint(85, 110)
        else:
            speed = rng.randint(10, 40)
        fuel_per_100km = rng.uniform(38, 45)   # при аномалии расход выше
    else:
        log_text = rng.choice(NORMAL_TEXTS)
        speed = rng.randint(50, 80)
        fuel_per_100km = rng.uniform(28, 34)

    # Пройденный путь и топливо за один шаг (по умолчанию 10 минут)
    distance_km = round(speed * CONFIG["step_minutes"] / 60, 2)
    fuel_l = round(distance_km * fuel_per_100km / 100, 2)

    json_row = {
        "timestamp": time_str,
        "device_id": truck["device"],
        "truck_id": truck_id,
        "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "speed_kmh": speed,
        "distance_km": distance_km,
        "fuel_l": fuel_l,
    }
    csv_row = [time_str, truck["device"], log_text]
    return json_row, csv_row


# ---------------------------------------------------------------
# 4. Инкрементальность: узнаём, до какого момента данные уже есть
# ---------------------------------------------------------------
def get_last_time_json():
    path = CONFIG["coords_file"]
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None
    return max(datetime.strptime(row["timestamp"], TIME_FORMAT) for row in data)


def get_last_time_csv():
    path = CONFIG["logs_file"]
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]          # пропускаем заголовок
    rows = [r for r in rows if r]               # пропускаем пустые строки
    if not rows:
        return None
    return max(datetime.strptime(r[0], TIME_FORMAT) for r in rows)


# ---------------------------------------------------------------
# 5. Безопасное сохранение (через временный файл)
# ---------------------------------------------------------------
def save_json(new_rows):
    """Читает старые данные, убирает дубли по (время, грузовик), добавляет новые."""
    path = CONFIG["coords_file"]
    tmp_path = path + ".tmp"
    old_rows = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            old_rows = json.load(f)

    new_keys = {(r["timestamp"], r["truck_id"]) for r in new_rows}
    kept = [r for r in old_rows if (r["timestamp"], r["truck_id"]) not in new_keys]
    all_rows = sorted(kept + new_rows, key=lambda r: (r["timestamp"], r["truck_id"]))

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, path)              # замена файла целиком
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_csv(new_rows):
    """То же самое для CSV: дубли определяются по (время, устройство)."""
    path = CONFIG["logs_file"]
    tmp_path = path + ".tmp"
    header = ["timestamp", "device_id", "telemetry_log"]
    old_rows = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            old_rows = [r for r in list(csv.reader(f))[1:] if r]

    new_keys = {(r[0], r[1]) for r in new_rows}
    kept = [r for r in old_rows if (r[0], r[1]) not in new_keys]
    all_rows = sorted(kept + new_rows, key=lambda r: (r[0], r[1]))

    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_rows)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------
# 6. Устойчивость к сбоям: повторные попытки подключения
# ---------------------------------------------------------------
def source_is_available(current_time):
    """Имитация недоступного источника. Пробуем несколько раз."""
    for attempt in range(1, CONFIG["max_retries"] + 1):
        if random.random() >= CONFIG["fail_chance"]:
            return True
        write_log("WARNING", f"Попытка {attempt}/{CONFIG['max_retries']}: "
                             f"источник недоступен на шаге {current_time}")
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------
# 7. Основной процесс
# ---------------------------------------------------------------
def run_etl(days=1):
    started = time.time()
    json_last = get_last_time_json()
    csv_last = get_last_time_csv()
    step = timedelta(minutes=CONFIG["step_minutes"])

    # С какого момента начинать: с самой ранней точки, где данных не хватает
    if json_last is None or csv_last is None:
        current_time = datetime.strptime(CONFIG["default_start_time"], TIME_FORMAT)
    else:
        current_time = min(json_last, csv_last) + step
    end_time = current_time + timedelta(days=days)

    write_log("INFO", f"Старт. Период: {current_time} - {end_time}, дней: {days}")

    new_json = []
    new_csv = []
    status = "SUCCESS"

    while current_time <= end_time:
        if not source_is_available(current_time):
            write_log("CRITICAL", f"Источник недоступен на шаге {current_time}. "
                                  f"Остановка, собранные данные сохраним.")
            status = "PARTIAL"
            break

        # Пишем в файл только то, чего в нём ещё нет
        need_json = json_last is None or current_time > json_last
        need_csv = csv_last is None or current_time > csv_last
        for truck_id in TRUCKS:
            json_row, csv_row = make_records(truck_id, current_time)
            if need_json:
                new_json.append(json_row)
            if need_csv:
                new_csv.append(csv_row)

        current_time += step

    try:
        if new_json:
            save_json(new_json)
        if new_csv:
            save_csv(new_csv)
    except Exception as error:
        write_log("CRITICAL", f"Ошибка записи, файлы не изменены: {error}")
        print(f"Ошибка записи: {error}")
        return

    seconds = round(time.time() - started, 1)
    message = (f"{status}. Записано строк JSON: {len(new_json)}, "
               f"CSV: {len(new_csv)}. Время работы: {seconds} с")
    write_log(status if status != "PARTIAL" else "WARNING", message)
    print(message)


if __name__ == "__main__":
    days_to_load = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_etl(days_to_load)

