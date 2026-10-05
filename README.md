## Live Demo

[Открыть приложение в Streamlit](https://titanic-ai-survival-ca7titanic-ai-survivalpwmekuvemycvnsqp8df.streamlit.app/)
# 🚢 Titanic Survival Predictor

Учебный мини-проект по машинному обучению: модель прогнозирует вероятность выживания пассажира Titanic по выбранным пользователем параметрам.

## Что используется

- Python
- pandas
- scikit-learn
- Random Forest Classifier
- Streamlit
- Titanic dataset

## Признаки модели

Модель использует:

- `Pclass` — класс билета;
- `Sex` — пол;
- `Age` — возраст;
- `SibSp` — число братьев/сестёр и супругов на борту;
- `Parch` — число родителей/детей на борту;
- `Fare` — стоимость билета;
- `Embarked` — порт посадки.

Целевая переменная — `Survived`: `0` — не выжил, `1` — выжил.

## Как работает ML-часть

1. Загружается Titanic dataset.
2. Данные делятся на обучающую и тестовую выборки 80/20.
3. Пропуски числовых признаков заполняются медианой.
4. Пропуски категориальных признаков заполняются наиболее частым значением.
5. Категории кодируются через One-Hot Encoding.
6. Обучается `RandomForestClassifier`.
7. В Streamlit пользователь вводит параметры пассажира.
8. Модель выводит класс и вероятность выживания.

## Запуск на Windows

### Вариант 1 — одной кнопкой

Запустите:

```text
run.bat
```

### Вариант 2 — через терминал

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

После запуска Streamlit откроет приложение в браузере, обычно по адресу `http://localhost:8501`.

## Запуск на macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Датасет

По умолчанию приложение загружает Titanic CSV из публичного GitHub-репозитория DataScienceDojo.

При желании можно заранее сохранить датасет локально:

```bash
python download_data.py
```

Тогда появится файл `data/titanic.csv`, и приложение будет использовать его.

## Как загрузить проект на GitHub

Создайте новый пустой репозиторий, затем выполните в папке проекта:

```bash
git init
git add .
git commit -m "Titanic survival ML app"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Замените `USERNAME` и `REPOSITORY` на свои значения.

## Публикация приложения

Проект удобно развернуть через Streamlit Community Cloud:

1. Загрузить проект на GitHub.
2. Создать приложение в Streamlit Community Cloud.
3. Выбрать GitHub-репозиторий.
4. В качестве main file указать `app.py`.
5. Нажать Deploy.

## Структура проекта

```text
titanic_ai/
├── app.py
├── download_data.py
├── requirements.txt
├── README.md
├── DEFENSE.md
├── run.bat
├── run.sh
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
```
