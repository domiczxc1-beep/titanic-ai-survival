from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
LOCAL_DATA = Path("data/titanic.csv")
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load Titanic data from a local CSV when available, otherwise from GitHub."""
    if LOCAL_DATA.exists():
        return pd.read_csv(LOCAL_DATA)
    return pd.read_csv(DATA_URL)


@st.cache_resource
def train_model(data: pd.DataFrame):
    """Build and train the preprocessing + classification pipeline."""
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, len(X_train), len(X_test)


st.title("🚢 Titanic Survival Predictor")
st.write(
    "Небольшое ML-приложение, которое оценивает вероятность выживания пассажира "
    "Titanic по выбранным параметрам."
)

try:
    data = load_data()
    model, accuracy, train_size, test_size = train_model(data)
except Exception as exc:
    st.error(
        "Не удалось загрузить датасет или обучить модель. "
        "Проверьте подключение к интернету либо положите файл data/titanic.csv в проект."
    )
    st.exception(exc)
    st.stop()

with st.expander("О модели и датасете"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Строк в датасете", len(data))
    col2.metric("Обучающая выборка", train_size)
    col3.metric("Тестовая выборка", test_size)
    st.metric("Accuracy на тестовой выборке", f"{accuracy:.1%}")
    st.caption(
        "Алгоритм: Random Forest. Пропуски заполняются автоматически, "
        "категориальные признаки преобразуются One-Hot Encoding."
    )

st.subheader("Параметры пассажира")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass_label = st.selectbox(
            "Класс билета",
            ["1 — первый", "2 — второй", "3 — третий"],
            index=2,
        )
        sex_label = st.selectbox("Пол", ["Мужской", "Женский"])
        age = st.slider("Возраст", min_value=0.5, max_value=80.0, value=25.0, step=0.5)
        sibsp = st.number_input(
            "Братья/сёстры или супруг(а) на борту",
            min_value=0,
            max_value=8,
            value=0,
            step=1,
        )

    with col2:
        parch = st.number_input(
            "Родители/дети на борту",
            min_value=0,
            max_value=6,
            value=0,
            step=1,
        )
        fare = st.number_input(
            "Стоимость билета (£)",
            min_value=0.0,
            max_value=520.0,
            value=15.0,
            step=1.0,
        )
        embarked_label = st.selectbox(
            "Порт посадки",
            ["Southampton", "Cherbourg", "Queenstown"],
        )

    submitted = st.form_submit_button("Сделать прогноз", use_container_width=True)

if submitted:
    pclass = int(pclass_label[0])
    sex = "male" if sex_label == "Мужской" else "female"
    embarked_map = {
        "Southampton": "S",
        "Cherbourg": "C",
        "Queenstown": "Q",
    }

    passenger = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked_map[embarked_label],
            }
        ]
    )

    prediction = int(model.predict(passenger)[0])
    probability = float(model.predict_proba(passenger)[0][1])

    st.subheader("Результат")
    st.progress(int(round(probability * 100)))
    st.write(f"Вероятность выживания: **{probability:.1%}**")

    if prediction == 1:
        st.success("Прогноз модели: пассажир, скорее всего, выжил бы.")
    else:
        st.error("Прогноз модели: пассажир, скорее всего, не выжил бы.")

    st.caption(
        "Это учебный статистический прогноз по историческому датасету, "
        "а не утверждение о конкретном реальном человеке."
    )
