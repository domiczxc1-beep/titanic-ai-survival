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
    page_title="Titanic AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- DESIGN --------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: rgba(16, 28, 45, 0.86);
        --panel-2: rgba(21, 37, 58, 0.92);
        --line: rgba(148, 163, 184, 0.18);
        --text: #f8fafc;
        --muted: #9fb0c6;
        --blue: #38bdf8;
        --blue2: #2563eb;
        --green: #34d399;
        --red: #fb7185;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(37,99,235,.18), transparent 30%),
            radial-gradient(circle at 88% 16%, rgba(56,189,248,.12), transparent 26%),
            linear-gradient(180deg, #07111f 0%, #081525 55%, #06101d 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 34px 34px 30px;
        background: linear-gradient(135deg, rgba(15,31,52,.96), rgba(11,24,41,.86));
        box-shadow: 0 22px 60px rgba(0,0,0,.28);
        margin-bottom: 22px;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 310px;
        height: 310px;
        right: -90px;
        top: -120px;
        background: radial-gradient(circle, rgba(56,189,248,.28), transparent 67%);
        pointer-events: none;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid rgba(56,189,248,.28);
        background: rgba(56,189,248,.08);
        color: #bde9ff;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: 18px 0 8px;
        font-size: clamp(42px, 6vw, 72px);
        line-height: .98;
        letter-spacing: -.045em;
        color: white;
    }

    .hero p {
        margin: 0;
        max-width: 760px;
        color: var(--muted);
        font-size: 17px;
        line-height: 1.65;
    }

    .chip-row {
        margin-top: 20px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .chip {
        padding: 8px 11px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255,255,255,.035);
        color: #d7e5f4;
        font-size: 13px;
        font-weight: 600;
    }

    .section-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -.02em;
        margin: 4px 0 14px;
        color: white;
    }

    .glass {
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 22px;
        background: var(--panel);
        box-shadow: 0 16px 42px rgba(0,0,0,.22);
    }

    .metric-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px 18px;
        background: rgba(255,255,255,.035);
        min-height: 104px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 700;
    }

    .metric-value {
        color: white;
        font-size: 27px;
        font-weight: 850;
        margin-top: 7px;
    }

    .result-card {
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 24px;
        min-height: 420px;
        background:
            linear-gradient(180deg, rgba(18,33,52,.96), rgba(11,24,40,.92));
        box-shadow: 0 16px 42px rgba(0,0,0,.22);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .result-kicker {
        color: var(--muted);
        font-size: 12px;
        font-weight: 750;
        letter-spacing: .09em;
        text-transform: uppercase;
    }

    .prob {
        font-size: clamp(64px, 8vw, 98px);
        line-height: .95;
        font-weight: 900;
        letter-spacing: -.05em;
        margin: 12px 0 6px;
        color: white;
    }

    .result-title {
        font-size: 22px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
    }

    .result-text {
        color: var(--muted);
        line-height: 1.55;
        margin-bottom: 18px;
    }

    .bar {
        width: 100%;
        height: 12px;
        background: rgba(148,163,184,.15);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
    }

    .bar > div {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #38bdf8, #34d399);
    }

    .placeholder-icon {
        font-size: 54px;
        margin-bottom: 14px;
    }

    div[data-testid="stForm"] {
        border: 1px solid var(--line);
        background: var(--panel);
        border-radius: 24px;
        padding: 22px 22px 8px;
        box-shadow: 0 16px 42px rgba(0,0,0,.22);
    }

    div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: rgba(255,255,255,.045) !important;
        border-color: rgba(148,163,184,.18) !important;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        border: 0 !important;
        border-radius: 14px !important;
        min-height: 52px;
        font-weight: 800 !important;
        color: white !important;
        background: linear-gradient(90deg, #2563eb, #0ea5e9) !important;
        box-shadow: 0 10px 28px rgba(14,165,233,.2);
        transition: .2s ease;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 34px rgba(14,165,233,.27);
    }

    .stExpander {
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        overflow: hidden;
        background: rgba(255,255,255,.025);
    }

    label, .stMarkdown, p, span {
        color: inherit;
    }

    footer {visibility: hidden;}

    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        .hero {
            padding: 26px 22px;
            border-radius: 22px;
        }
        .hero h1 {
            font-size: 48px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- DATA & MODEL --------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if LOCAL_DATA.exists():
        return pd.read_csv(LOCAL_DATA)
    return pd.read_csv(DATA_URL)


@st.cache_resource
def train_model(data: pd.DataFrame):
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
        [("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
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
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, len(X_train), len(X_test)


try:
    data = load_data()
    model, accuracy, train_size, test_size = train_model(data)
except Exception as exc:
    st.error("Не удалось загрузить датасет или обучить модель.")
    st.exception(exc)
    st.stop()


# -------------------- HERO --------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">● Machine Learning Project</div>
        <h1>Titanic AI</h1>
        <p>
            Интерактивный прогноз выживаемости пассажира на основе исторического
            датасета Titanic. Выберите параметры — модель Random Forest оценит
            вероятность результата.
        </p>
        <div class="chip-row">
            <div class="chip">Python</div>
            <div class="chip">scikit-learn</div>
            <div class="chip">Random Forest</div>
            <div class="chip">Streamlit</div>
            <div class="chip">GitHub</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metrics
m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("Датасет", f"{len(data)} строк"),
    ("Train", f"{train_size}"),
    ("Test", f"{test_size}"),
    ("Accuracy", f"{accuracy:.1%}"),
]
for col, (label, value) in zip((m1, m2, m3, m4), metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

with st.expander("Как работает модель"):
    st.markdown(
        """
        **Pipeline:** заполнение пропусков → One-Hot Encoding → Random Forest.

        Модель обучается на 80% данных, а на оставшихся 20% оценивается точность.
        Целевая переменная `Survived`: `0` — не выжил, `1` — выжил.
        """
    )


# -------------------- MAIN AREA --------------------
left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown('<div class="section-title">Параметры пассажира</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            pclass_label = st.selectbox(
                "Класс билета",
                ["1 — первый класс", "2 — второй класс", "3 — третий класс"],
                index=2,
            )
            sex_label = st.selectbox("Пол", ["Мужской", "Женский"])
            age = st.slider(
                "Возраст",
                min_value=0.5,
                max_value=80.0,
                value=25.0,
                step=0.5,
            )
            sibsp = st.number_input(
                "Братья / сёстры / супруг(а)",
                min_value=0,
                max_value=8,
                value=0,
                step=1,
            )

        with c2:
            parch = st.number_input(
                "Родители / дети",
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

            st.markdown(
                """
                <div style="
                    margin-top: 14px;
                    padding: 14px 15px;
                    border-radius: 14px;
                    border: 1px solid rgba(148,163,184,.16);
                    background: rgba(255,255,255,.025);
                    color: #9fb0c6;
                    font-size: 13px;
                    line-height: 1.5;">
                    Прогноз носит учебный характер и показывает статистическую
                    вероятность по историческим данным.
                </div>
                """,
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button("Рассчитать вероятность")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None

if submitted:
    pclass = int(pclass_label[0])
    sex = "male" if sex_label == "Мужской" else "female"
    embarked_map = {
        "Southampton": "S",
        "Cherbourg": "C",
        "Queenstown": "Q",
    }

    passenger = pd.DataFrame(
        [{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked_map[embarked_label],
        }]
    )

    st.session_state.prediction = int(model.predict(passenger)[0])
    st.session_state.probability = float(model.predict_proba(passenger)[0][1])


with right:
    st.markdown('<div class="section-title">Результат</div>', unsafe_allow_html=True)

    if st.session_state.probability is None:
        st.markdown(
            """
            <div class="result-card">
                <div class="placeholder-icon">🧭</div>
                <div class="result-title">Готово к прогнозу</div>
                <div class="result-text">
                    Заполните параметры пассажира слева и нажмите
                    «Рассчитать вероятность».
                </div>
                <div class="bar"><div style="width: 0%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        probability = st.session_state.probability
        prediction = st.session_state.prediction
        pct = round(probability * 100)

        if prediction == 1:
            icon = "✅"
            title = "Высокая вероятность выживания"
            message = "По выбранным параметрам модель относит пассажира к классу «выжил»."
        else:
            icon = "⚠️"
            title = "Низкая вероятность выживания"
            message = "По выбранным параметрам модель относит пассажира к классу «не выжил»."

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-kicker">Прогноз модели</div>
                <div class="prob">{pct}%</div>
                <div class="result-title">{icon} {title}</div>
                <div class="result-text">{message}</div>
                <div class="bar"><div style="width: {pct}%"></div></div>
                <div style="margin-top:12px;color:#9fb0c6;font-size:13px;">
                    Вероятность класса «выжил»: {probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    """
    <div style="text-align:center;color:#708399;font-size:12px;padding:12px 0 4px;">
        Titanic AI • учебный ML-проект • Random Forest + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
