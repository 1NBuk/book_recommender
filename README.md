# Book Recommender System

Мультиисточниковая рекомендательная система для книжного онлайн-сервиса. Проект выполнен в рамках дисциплины «Введение в науку о данных», профиль «DevOps-инженерия».

## Обоснование выбора задачи

Методические указания предполагают задачу бинарной классификации. По согласованию с преподавателем выбрана задача рекомендательных систем как развитие бинарной классификации в прикладном контексте. Обоснование:

- Рекомендации напрямую влияют на бизнес-метрики онлайн-сервисов: конверсию, средний чек, удержание пользователей.
- Задача ранжирования сводится к серии бинарных решений «показывать / не показывать книгу конкретному пользователю»; метрики Precision@K, Recall@K и NDCG@K расширяют Accuracy, Precision и Recall из классической постановки.
- Задача позволяет полноценно задействовать три роли команды: Data Scientist, Data Engineer и Business Analyst.


## Данные

Используются три публичных датасета с платформы Kaggle:

- Book-Crossing (~1.15M оценок, 271K книг) — основной источник взаимодействий.
- Goodbooks-10k (~980K оценок, 10K книг) — метаданные и теги книг.
- Amazon Books Reviews (~1M отзывов) — тексты отзывов для эмбеддингов.

Датасеты объединяются по ISBN и по паре «название + автор». Пользователи из разных источников получают префиксы (`bc_`, `gb_`, `amazon_`) для исключения коллизий.

## Модели

Реализованы восемь моделей:

- Popularity — baseline на взвешенной популярности
- ALS — матричная факторизация (implicit feedback)
- ContentBased — TF-IDF профиль пользователя
- LightGBM — регрессия рейтинга
- LGBMGroupRanker — learning-to-rank с оптимизацией NDCG
- TwoTower — нейросетевая архитектура с текстовыми эмбеддингами
- Hybrid — взвешенный ансамбль базовых моделей
- TwoStageRanker — двухэтапная схема recall → rank

## Метрики

Ranking-метрики: Precision@K, Recall@K, NDCG@K, MAP@K, HitRate@K, MRR@K (K = 5, 10, 20).

Бизнес-метрики: Coverage, Diversity, Novelty, Personalization.

Основная метрика проекта — NDCG@10.

## Структура проекта

```
book_recommender/
├── configs/          # YAML-конфиги с наследованием
├── src/
│   ├── data/         # загрузка, предобработка, разбиение
│   ├── features/     # признаки пользователей, книг, текстов
│   ├── models/       # восемь моделей
│   ├── evaluation/   # метрики, кеш, оценщики
│   ├── pipeline/     # оркестратор, менеджер конфигов
│   └── analysis/     # EDA-отчёт
├── scripts/          # CLI-скрипты запуска
├── tests/            # pytest-тесты
├── feature_store/    # промежуточные артефакты признаков
├── models_saved/     # обученные модели
└── experiments/      # метрики, визуализации, кеш рекомендаций
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск пайплайна

Полный пайплайн одной командой:

```bash
python scripts/run_pipeline.py --config configs/multi_source.yaml --stage all
```

Запуск отдельных этапов:

```bash
python scripts/run_pipeline.py --config configs/multi_source.yaml --stage ingest
python scripts/run_pipeline.py --config configs/multi_source.yaml --stage features
python scripts/run_pipeline.py --config configs/multi_source.yaml --stage train
python scripts/run_pipeline.py --config configs/multi_source.yaml --stage evaluate
```

Принудительный пересчёт артефактов:

```bash
python scripts/run_pipeline.py --config configs/multi_source.yaml --stage all --force
```

Через DVC:

```bash
dvc repro
```

Через Makefile:

```bash
make pipeline
make test
make lint
```

## Тесты

```bash
pytest tests/ -v
```

Покрытие: ConfigManager (наследование, разрешение путей, валидация), Preprocessor (нормализация ISBN и текста, фильтрация), ранжирующие метрики (все шесть), базовые модели.

## Линтинг и форматирование

```bash
flake8 src/ scripts/ tests/
black src/ scripts/ tests/
isort src/ scripts/ tests/
```

## Воспроизводимость

- Все параметры вынесены в YAML-конфиги.
- Random seed фиксирован (42) в конфиге и во всех моделях.
- Промежуточные артефакты кешируются в `feature_store/`.
- DVC отслеживает версии данных, моделей и метрик.
- Пайплайн идемпотентен: повторный запуск пропускает готовые этапы.

## Результаты

Финальная сравнительная таблица всех моделей: `experiments/metrics/final_results_v2.csv`.

Визуализации EDA: `experiments/figures/eda/`. Визуализации результатов: `experiments/figures/results/` (сравнение моделей, heatmaps метрик, radar chart, сегментный анализ, feature importance).

## Лицензия

Учебный проект, данные используются из открытых источников (Kaggle, лицензии CC0 и CC-BY-SA).
