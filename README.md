Структура данных
Убедитесь, что в директории data/raw/ находятся следующие файлы:

data/raw/
├── book_descriptions.csv
├── book_genres.csv
├── books.csv
├── genres.csv
├── sample_submission.csv
├── test.csv
├── train.csv
└── users.csv


Структура проекта
.
├── data/
│   ├── raw/              # Исходные CSV-файлы
│   ├── interim/          # Промежуточные данные (при необходимости)
│   └── processed/        # Обработанные данные с признаками (parquet)
├── output/
│   ├── models/           # Обученные модели и TF-IDF векторайзер
│   └── submissions/      # Файлы submission
├── src/baseline/
│   ├── config.py         # Конфигурация и параметры модели
│   ├── constants.py      # Константы проекта (имена файлов, колонок)
│   ├── data_processing.py # Загрузка и объединение raw данных
│   ├── features.py       # Feature engineering (агрегаты, жанры, TF-IDF, BERT)
│   ├── prepare_data.py   # Подготовка данных (загрузка, обработка, сохранение)
│   ├── temporal_split.py # Утилиты для корректного временного разделения данных
│   ├── train.py          # Обучение модели (использует prepared данные)
│   ├── predict.py        # Генерация предсказаний (использует prepared данные)
│   ├── validate.py       # Проверка формата submission
│   └── evaluate.py       # Оценка качества предсказаний (метрики)


Зависимости
Python >= 3.10
pandas, scikit-learn, lightgbm, joblib
transformers, torch, sentencepiece
ruff, pre-commit (dev)
