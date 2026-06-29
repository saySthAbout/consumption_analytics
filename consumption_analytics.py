import os
import glob
import warnings
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="경기도 소비 트렌드 분석 및 매출 예측 AI",
    layout="wide"
)

import matplotlib.font_manager as fm

# pkl 역직렬화를 위해 학습 때와 동일한 클래스를 모듈 최상단에 정의
try:
    import torch
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc   = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16),        nn.ReLU(),
                nn.Linear(16, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16), nn.ReLU(),
                nn.Linear(16, 32),         nn.ReLU(),
                nn.Linear(32, input_dim),  nn.Sigmoid()
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))
        def encode(self, x):
            return self.encoder(x)

    _torch_available = True
except ImportError:
    _torch_available = False

def _setup_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    found = [p for p in fm.findSystemFonts(fontext="ttf")
             if any(k in p for k in ("Nanum", "nanum", "Malgun", "malgun", "AppleGothic"))]
    path = next((p for p in candidates if os.path.exists(p)), None) or (found[0] if found else None)
    if path:
        fm.fontManager.addfont(path)
        font_name = fm.FontProperties(fname=path).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["axes.unicode_minus"] = False

_setup_korean_font()

def _apply_korean_font():
    _setup_korean_font()

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
ENCODER_DIR = os.path.join(BASE_DIR, "encoders")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

SALES_MODEL_PATH     = os.path.join(MODEL_DIR,   "sales_predict_model.pkl")
MODEL_INFO_PATH      = os.path.join(MODEL_DIR,   "model_info.pkl")
LABEL_ENCODER_PATH   = os.path.join(ENCODER_DIR, "label_encoders.pkl")
ONEHOT_ENCODER_PATH  = os.path.join(ENCODER_DIR, "onehot_encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ENCODER_DIR, "feature_columns.pkl")
LSTM_MODEL_PATH      = os.path.join(MODEL_DIR,   "lstm_model.pkl")
CLUSTER_MODEL_PATH   = os.path.join(MODEL_DIR,   "cluster_model.pkl")

GDRIVE_FILE_ID   = "1JnLZDcT0OY2bAQLcinH_eSXpzg9hS7RY"
GDRIVE_FILE_NAME = "tbsh_gyeonggi_day_202602_hwaseungsi.csv"

DATA_PATHS = [
    os.path.join(DATASET_DIR, GDRIVE_FILE_NAME),
    os.path.join(BASE_DIR,    GDRIVE_FILE_NAME),
]


def ensure_sales_data():
    """CSV가 없으면 Google Drive에서 자동 다운로드."""
    if any(os.path.exists(p) for p in DATA_PATHS):
        return
    import gdown
    os.makedirs(DATASET_DIR, exist_ok=True)
    dest = DATA_PATHS[0]
    url  = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    with st.spinner("매출 데이터를 Google Drive에서 다운로드 중입니다... (최초 1회)"):
        gdown.download(url, dest, quiet=False)

ADMIN_CODE_PATHS = [
    os.path.join(DATASET_DIR, "city_admin_code.csv"),
]

MODEL_FEATURES = ["age", "day", "hour", "month", "cnt",
                  "sex", "admi_cty_no", "card_tpbuz_nm_1", "card_tpbuz_nm_2"]
CAT_COLS  = ["sex", "admi_cty_no", "card_tpbuz_nm_1", "card_tpbuz_nm_2"]
NUM_COLS  = ["age", "day", "hour", "month", "cnt"]
LABEL_COLS  = ["age", "day", "hour", "month"]
ONEHOT_COLS = CAT_COLS
NUMERIC_COLS = ["cnt"]

AGE_MAP = {
    1: "0~9세",   2: "10~19세", 3: "20~29세", 4: "30~39세",
    5: "40~49세", 6: "50~59세", 7: "60~69세", 8: "70~79세",
    9: "80~89세", 10: "90~99세", 11: "100세 이상"
}
DAY_MAP = {
    1: "월요일", 2: "화요일", 3: "수요일", 4: "목요일",
    5: "금요일", 6: "토요일", 7: "일요일"
}
HOUR_MAP = {
    1: "00:00~06:59", 2: "07:00~08:59", 3: "09:00~10:59",
    4: "11:00~12:59", 5: "13:00~14:59", 6: "15:00~16:59",
    7: "17:00~18:59", 8: "19:00~20:59", 9: "21:00~22:59",
    10: "23:00~23:59"
}
SEX_MAP         = {"M": "남성", "F": "여성"}
SEX_REVERSE_MAP = {"남성": "M", "여성": "F"}


# =========================================================
# 1. 데이터 로드 함수
# =========================================================
def read_csv_auto(path_list):
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    for path in path_list:
        if os.path.exists(path):
            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    return df, enc, path
                except Exception:
                    continue
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다.\n확인 경로: {path_list}")


@st.cache_data
def load_sales_data():
    return read_csv_auto(DATA_PATHS)


def find_admin_path():
    for p in ADMIN_CODE_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "city_admin_code.csv 파일을 찾을 수 없습니다.\n"
        "dataset/city_admin_code.csv 위치를 확인해주세요."
    )


@st.cache_data
def load_admin_code_data(mtime):
    path      = find_admin_path()
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err  = None
    for enc in encodings:
        try:
            admin_df = pd.read_csv(path, encoding=enc)
            admin_df.columns = [c.strip() for c in admin_df.columns]
            for col in ["admi_cty_no", "admi_cty_name"]:
                if col not in admin_df.columns:
                    raise ValueError(f"'{col}' 컬럼이 없습니다. 컬럼: {admin_df.columns.tolist()}")
            admin_df = admin_df[["admi_cty_no", "admi_cty_name"]].dropna()
            admin_df["admi_cty_no"]   = admin_df["admi_cty_no"].astype(str).str.strip().astype(int)
            admin_df["admi_cty_name"] = admin_df["admi_cty_name"].astype(str).str.strip()
            admin_df = admin_df[admin_df["admi_cty_name"] != ""]
            admin_df = admin_df.drop_duplicates("admi_cty_no").sort_values("admi_cty_no")
            return admin_df, enc, path
        except Exception as e:
            last_err = e
    raise ValueError(f"city_admin_code.csv 읽기 실패: {last_err}")


DISTRICT_MAP = {
    "41111": "수원시 장안구", "41113": "수원시 권선구",
    "41115": "수원시 팔달구", "41117": "수원시 영통구",
    "41131": "성남시 수정구", "41133": "성남시 중원구", "41135": "성남시 분당구",
    "41150": "안양시 만안구",  # 41150 → 만안구 (41171도 안양시)
    "41171": "안양시 만안구", "41173": "안양시 동안구",
    "41210": "부천시",
    "41271": "광명시",
    "41273": "안산시 단원구",
    "41360": "남양주시",
    "41390": "시흥시",
    "41450": "하남시",
    "41461": "용인시 처인구", "41463": "용인시 기흥구", "41465": "용인시 수지구",
    "41480": "과천시",
    "41570": "의정부시",
    "41591": "화성시", "41593": "화성시", "41595": "화성시", "41597": "화성시",
    "41650": "파주시",
    "41670": "김포시",
    "41800": "여주시",
}

def build_admin_maps(admin_df):
    admin_df = admin_df.copy()
    admin_df["admi_cty_no"]   = admin_df["admi_cty_no"].astype(int)
    admin_df["admi_cty_name"] = admin_df["admi_cty_name"].astype(str).str.strip()
    admin_df["district"]      = admin_df["admi_cty_no"].astype(str).str[:5].map(DISTRICT_MAP).fillna("기타")
    name_to_code = dict(zip(admin_df["admi_cty_name"], admin_df["admi_cty_no"]))
    code_to_name = dict(zip(admin_df["admi_cty_no"],   admin_df["admi_cty_name"]))
    # district → dong 목록
    district_to_dongs = (
        admin_df.groupby("district")["admi_cty_name"]
        .apply(sorted).to_dict()
    )
    district_list = sorted(district_to_dongs.keys())
    return district_list, district_to_dongs, name_to_code, code_to_name


def preprocess_data(df):
    df = df.copy()
    if "ta_ymd" in df.columns:
        df["ta_ymd"] = pd.to_datetime(df["ta_ymd"].astype(str), format="%Y%m%d", errors="coerce")
        df["month"]  = df["ta_ymd"].dt.month
        df["date"]   = df["ta_ymd"].dt.date
    df["age_label"]  = df["age"].map(AGE_MAP)
    df["day_label"]  = df["day"].map(DAY_MAP)
    df["hour_label"] = df["hour"].map(HOUR_MAP)
    df["sex_label"]  = df["sex"].map(SEX_MAP)
    df["log_amt"]    = np.log1p(df["amt"])
    df["log_cnt"]    = np.log1p(df["cnt"])
    return df


# =========================================================
# 2. 인코딩 함수
# =========================================================
def create_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_and_save_encoders(X):
    X = X.copy()
    label_encoders = {}
    for col in LABEL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    ohe       = create_ohe()
    ohe_arr   = ohe.fit_transform(X[ONEHOT_COLS].astype(str))
    ohe_names = ohe.get_feature_names_out(ONEHOT_COLS).tolist()
    ohe_df    = pd.DataFrame(ohe_arr, columns=ohe_names, index=X.index)

    encoded_X       = pd.concat([X[NUM_COLS].reset_index(drop=True),
                                 ohe_df.reset_index(drop=True)], axis=1)
    feature_columns = encoded_X.columns.tolist()

    joblib.dump(label_encoders,  LABEL_ENCODER_PATH)
    joblib.dump(ohe,             ONEHOT_ENCODER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    return encoded_X


def load_encoders():
    label_encoders = joblib.load(LABEL_ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    if os.path.exists(ONEHOT_ENCODER_PATH):
        label_encoders["__ohe__"] = joblib.load(ONEHOT_ENCODER_PATH)
    return label_encoders, feature_columns


def transform_with_saved_encoders(X):
    X = X.copy()
    label_encoders, feature_columns = load_encoders()
    model_info = joblib.load(MODEL_INFO_PATH) if os.path.exists(MODEL_INFO_PATH) else {}
    encoding = model_info.get("encoding", "ohe")

    if encoding == "label":
        for col in CAT_COLS:
            le = label_encoders[col]
            val = str(X[col].iloc[0])
            X[col] = le.transform([val])[0] if val in le.classes_ else 0
        return X[feature_columns].astype(float)
    else:
        ohe = label_encoders.get("__ohe__")
        for col in LABEL_COLS:
            X[col] = X[col].astype(str)
            le = label_encoders[col]
            unknown = set(X[col].unique()) - set(le.classes_)
            if unknown:
                raise ValueError(f"'{col}' 컬럼에 학습되지 않은 값: {unknown}")
            X[col] = le.transform(X[col])
        if ohe is not None:
            ohe_arr   = ohe.transform(X[ONEHOT_COLS].astype(str))
            ohe_names = ohe.get_feature_names_out(ONEHOT_COLS).tolist()
            ohe_df    = pd.DataFrame(ohe_arr, columns=ohe_names, index=X.index)
            num_df    = X[NUMERIC_COLS].reset_index(drop=True)
            encoded_X = pd.concat([X[LABEL_COLS].reset_index(drop=True), num_df,
                                   ohe_df.reset_index(drop=True)], axis=1)
            return encoded_X.reindex(columns=feature_columns, fill_value=0)
        return X[feature_columns]


# =========================================================
# 3. 모델 학습 / 저장 / 예측
# =========================================================
def train_and_save_model(df, sample_size=100000, use_log_target=True, model_name="RandomForest",
                         remove_outliers=True):
    model_df = df[MODEL_FEATURES + ["amt", "log_amt"]].dropna().copy()
    if remove_outliers:
        upper = model_df["amt"].quantile(0.99)
        model_df = model_df[model_df["amt"] <= upper]
    if sample_size and len(model_df) > sample_size:
        model_df = model_df.sample(sample_size, random_state=42)

    X = model_df[MODEL_FEATURES]
    y = model_df["log_amt"] if use_log_target else model_df["amt"]

    encoded_X = fit_and_save_encoders(X)
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_X, y, test_size=0.2, random_state=42)

    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                   num_leaves=127, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=15,
                                      min_samples_leaf=5, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if use_log_target:
        y_test_real = np.expm1(y_test)
        pred_real   = np.expm1(pred)
    else:
        y_test_real = y_test
        pred_real   = pred
    pred_real = np.maximum(pred_real, 0)

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test_real, pred_real)),
        "MAE":  mean_absolute_error(y_test_real, pred_real),
        "R2":   r2_score(y_test_real, pred_real),
    }
    model_info = {
        "model_name":      model_name,
        "use_log_target":  use_log_target,
        "remove_outliers": remove_outliers,
        "sample_size":     sample_size,
        "features":        MODEL_FEATURES,
        "label_cols":      LABEL_COLS,
        "onehot_cols":     ONEHOT_COLS,
        "metrics":         metrics,
    }
    joblib.dump(model,      SALES_MODEL_PATH)
    joblib.dump(model_info, MODEL_INFO_PATH)
    return model, model_info, X_test, y_test_real, pred_real


def load_saved_model():
    return joblib.load(SALES_MODEL_PATH), joblib.load(MODEL_INFO_PATH)


def train_single_model_no_save(X_train, X_test, y_train, y_test, model_name, use_log_target):
    """인코딩된 데이터로 모델 하나만 학습 — 파일 저장 없이 metrics 반환."""
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                   num_leaves=127, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=15,
                                      min_samples_leaf=5, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if use_log_target:
        y_real  = np.expm1(y_test)
        p_real  = np.maximum(np.expm1(pred), 0)
    else:
        y_real  = y_test
        p_real  = np.maximum(pred, 0)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_real, p_real)),
        "MAE":  mean_absolute_error(y_real, p_real),
        "R2":   r2_score(y_real, p_real),
    }
    return model, metrics


def model_files_exist():
    return all(os.path.exists(p) for p in [
        SALES_MODEL_PATH, MODEL_INFO_PATH,
        LABEL_ENCODER_PATH, ONEHOT_ENCODER_PATH, FEATURE_COLUMNS_PATH
    ])


def predict_sales(input_df):
    model, model_info = load_saved_model()
    encoded = transform_with_saved_encoders(input_df)
    pred    = model.predict(encoded)[0]
    if model_info.get("use_log_target", True):
        pred = np.expm1(pred)
    return max(pred, 0)


# =========================================================
# 4. 시각화 함수
# =========================================================
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def plot_corr_heatmap(df):
    _apply_korean_font()
    cols = [c for c in ["cty_rgn_no","admi_cty_no","card_tpbuz_cd",
                        "hour","age","day","cnt","amt","log_amt"] if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("수치형 변수 상관관계 Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig, corr


def plot_age_hour_heatmap(df):
    _apply_korean_font()
    pivot = pd.pivot_table(df, index="age_label", columns="hour_label",
                           values="amt", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=list(AGE_MAP.values()), columns=list(HOUR_MAP.values()))
    p100m = pivot / 1e8
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(p100m, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.4, ax=ax)
    ax.set_title("연령대 × 시간대 총매출액 Heatmap (단위: 억 원)", fontsize=16, fontweight="bold")
    ax.set_xlabel("시간대"); ax.set_ylabel("연령대")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    return fig, p100m


def plot_day_hour_heatmap(df):
    _apply_korean_font()
    pivot = pd.pivot_table(df, index="day_label", columns="hour_label",
                           values="amt", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=list(DAY_MAP.values()), columns=list(HOUR_MAP.values()))
    p100m = pivot / 1e8
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(p100m, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.4, ax=ax)
    ax.set_title("요일 × 시간대 총매출액 Heatmap (단위: 억 원)", fontsize=16, fontweight="bold")
    ax.set_xlabel("시간대"); ax.set_ylabel("요일")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    return fig, p100m


def plot_biz_hour_heatmap(df, top_n=10):
    _apply_korean_font()
    top_biz = (df.groupby("card_tpbuz_nm_2")["amt"]
               .sum().sort_values(ascending=False).head(top_n).index)
    pivot = pd.pivot_table(df[df["card_tpbuz_nm_2"].isin(top_biz)],
                           index="card_tpbuz_nm_2", columns="hour_label",
                           values="amt", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=list(HOUR_MAP.values()))
    p100m = pivot / 1e8
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(p100m, annot=True, fmt=".1f", cmap="PuBuGn", linewidths=0.4, ax=ax)
    ax.set_title(f"업종 중분류 TOP {top_n} × 시간대 총매출액 Heatmap (단위: 억 원)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("시간대"); ax.set_ylabel("업종 중분류")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    return fig, p100m


def plot_histograms(df):
    _apply_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["amt"],     bins=50, ax=axes[0])
    axes[0].set_title("매출금액(amt) 분포"); axes[0].set_xlabel("매출금액")
    sns.histplot(df["log_amt"], bins=50, ax=axes[1])
    axes[1].set_title("로그 변환 매출금액(log_amt) 분포"); axes[1].set_xlabel("log1p(amt)")
    plt.tight_layout()
    return fig


def plot_actual_pred(y_true, y_pred):
    _apply_korean_font()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], "r--")
    ax.set_title("예측값 vs 실제값")
    ax.set_xlabel("실제 매출금액"); ax.set_ylabel("예측 매출금액")
    plt.tight_layout()
    return fig


# =========================================================
# 5. Streamlit 화면
# =========================================================
st.title("경기도 소비 트렌드 분석 및 매출 예측 AI")
st.caption("경기도 카드 소비 데이터 기반 · 소상공인을 위한 매출 예측 서비스")

# ── 사이드바 ─────────────────────────────────────────────
st.sidebar.header("설정")
if st.sidebar.button("Streamlit 캐시 초기화"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("캐시 초기화 완료. 새로고침해주세요.")

sample_size     = 100000
use_log_target  = True
remove_outliers = True
model_name      = "LightGBM"

# ── 매출 데이터 로드 ────────────────────────────────────
ensure_sales_data()
try:
    raw_df, sales_enc, sales_path = load_sales_data()
    df = preprocess_data(raw_df)
except Exception as e:
    st.error(f"매출 데이터 로드 실패: {e}")
    st.info("dataset 폴더 안의 매출 CSV 파일 위치를 확인해주세요.")
    st.stop()

# ── 행정동 코드 로드 (없어도 앱 계속 동작) ─────────────
try:
    admin_real_path    = find_admin_path()
    admin_mtime        = os.path.getmtime(admin_real_path)
    admin_df, admin_enc, admin_path = load_admin_code_data(admin_mtime)
    admin_district_list, admin_district_to_dongs, admin_name_to_code, _ = build_admin_maps(admin_df)
    admin_ok = True
except Exception:
    admin_ok                = False
    admin_df                = pd.DataFrame(columns=["admi_cty_no", "admi_cty_name"])
    admin_district_list     = []
    admin_district_to_dongs = {}
    admin_name_to_code      = {v: v for v in sorted(df["admi_cty_no"].dropna().astype(str).unique())}
    admin_enc               = "-"
    admin_path              = "-"

# ── 탭 (항상 생성) ───────────────────────────────────────
tab_pred, tab_hm, tab_eda, tab_lstm, tab_cluster, tab_ai, tab_ov = st.tabs([
    "💰 매출 예측",
    "⏰ 시간대·요일 분석",
    "📊 소비 트렌드",
    "📈 시계열 예측",
    "👥 고객 군집 분석",
    "📝 AI 리포트",
    "ℹ️ 데이터 정보",
])

_mi = joblib.load(MODEL_INFO_PATH) if os.path.exists(MODEL_INFO_PATH) else {}

# =====================================================
# 💰 매출 예측 (탭 1 - 사장님 하루 단위 예측)
# =====================================================
with tab_pred:
    st.subheader("오늘 하루 예상 매출은 얼마일까요?")
    st.caption("월·요일·동네·업종을 선택하면 하루 예상 매출을 알려드립니다.")

    p1, p2 = st.columns(2)

    with p1:
        sel_month = st.selectbox("월", list(range(1, 13)), index=0,
                                 format_func=lambda x: f"{x}월")
        sel_day_label = st.selectbox("요일", list(DAY_MAP.values()), index=4)
        sel_day = {v: k for k, v in DAY_MAP.items()}[sel_day_label]

    with p2:
        if admin_ok:
            sel_district  = st.selectbox("지역 (시/구)", admin_district_list)
            dong_opts     = admin_district_to_dongs.get(sel_district, [])
            sel_admi_name = st.selectbox("동네 선택", dong_opts)
            sel_admi      = admin_name_to_code.get(sel_admi_name, 0)
        else:
            st.warning("⚠️ 행정동 코드 파일 없음")
            fallback_opts = sorted(df["admi_cty_no"].dropna().astype(str).unique().tolist())
            sel_admi_name = st.selectbox("동네", fallback_opts)
            sel_admi      = int(sel_admi_name)

        sel_biz1  = st.selectbox("업종 대분류", sorted(df["card_tpbuz_nm_1"].dropna().unique()))
        biz2_opts = sorted(df[df["card_tpbuz_nm_1"] == sel_biz1]["card_tpbuz_nm_2"].dropna().unique())
        if not biz2_opts:
            st.warning("선택한 대분류에 해당하는 중분류가 없습니다.")
            sel_biz2 = None
        else:
            sel_biz2 = st.selectbox("업종 중분류", biz2_opts)

        avg_cnt = int(round(
            df[df["card_tpbuz_nm_2"] == sel_biz2]["cnt"].mean()
        )) if sel_biz2 and "cnt" in df.columns else 10
        sel_cnt = st.number_input(
            f"하루 예상 거래 건수  ※ {sel_biz2} 평균: {avg_cnt}건",
            min_value=1, value=avg_cnt, step=1
        )

    if sel_biz2 is not None and st.button("하루 예상 매출 계산하기", type="primary"):
        try:
            model, model_info = load_saved_model()

            # 선택 조건에 맞는 실제 데이터 필터링
            ref_mask = (
                (df["card_tpbuz_nm_2"] == sel_biz2) &
                (df["day"] == sel_day) &
                (df["month"] == sel_month)
            )
            ref = df[ref_mask] if ref_mask.sum() >= 10 else df[df["card_tpbuz_nm_2"] == sel_biz2]

            # 실제 데이터에 존재하는 시간대·성별·연령 조합 추출
            combos = (
                ref[["hour", "sex", "age"]]
                .drop_duplicates()
                .dropna()
            )
            active_hours = int(ref["hour"].nunique()) if not ref.empty else 10

            # 모든 (시간대 × 성별 × 연령) 조합으로 예측 후 평균 건당 매출 계산
            preds = []
            for _, row in combos.iterrows():
                input_row = pd.DataFrame([{
                    "sex": row["sex"], "age": int(row["age"]),
                    "day": sel_day, "hour": int(row["hour"]),
                    "month": sel_month, "admi_cty_no": sel_admi,
                    "card_tpbuz_nm_1": sel_biz1, "card_tpbuz_nm_2": sel_biz2,
                    "cnt": sel_cnt,
                }])
                try:
                    encoded  = transform_with_saved_encoders(input_row)
                    raw_pred = model.predict(encoded)[0]
                    p = max(np.expm1(raw_pred) if model_info.get("use_log_target", True) else raw_pred, 0)
                    preds.append(p)
                except Exception:
                    pass

            # 전체 조합의 평균 건당 매출
            avg_per_txn = float(np.mean(preds)) if preds else 0.0

            # 하루 예상 매출 = 평균 건당 매출 × 시간대당 거래건수 × 하루 활성 시간대수
            daily_total_cnt = sel_cnt * active_hours
            daily_pred      = avg_per_txn * daily_total_cnt

            # 비교 기준: 실제 데이터 해당 업종 평균 일매출 (전체 amt ÷ 전체 날짜 수)
            area_mask = df["card_tpbuz_nm_2"] == sel_biz2
            if area_mask.any():
                total_days = df.loc[area_mask, "ta_ymd"].nunique() if "ta_ymd" in df.columns else 30
                area_daily_avg = df.loc[area_mask, "amt"].sum() / max(total_days, 1)
            else:
                area_daily_avg = None

            # 결과 표시
            st.success(f"### 하루 예상 매출: **{daily_pred:,.0f}원**")
            c1, c2, c3 = st.columns(3)
            c1.metric("건당 평균 매출", f"{avg_per_txn:,.0f}원")
            c2.metric("하루 예상 거래건수", f"{daily_total_cnt}건 ({sel_cnt}건 × {active_hours}시간대)")
            c3.metric("분석 업종", sel_biz2)

            if area_daily_avg:
                diff_pct = (daily_pred - area_daily_avg) / area_daily_avg * 100
                arrow = "▲" if diff_pct >= 0 else "▼"
                color = "🟢" if diff_pct >= 0 else "🔴"
                st.info(f"{color} 해당 업종 평균 일매출({area_daily_avg:,.0f}원)보다 {arrow} {abs(diff_pct):.1f}% {'높은 수준입니다' if diff_pct >= 0 else '낮은 수준입니다'}")

            with st.expander("계산 과정 보기"):
                st.text(f"- 예측에 사용된 조합 수: {len(preds)}개 (시간대 × 성별 × 연령)")
                st.text(f"- 건당 평균 예측 매출: {avg_per_txn:,.0f}원")
                st.text(f"- 시간대당 거래건수: {sel_cnt}건")
                st.text(f"- 하루 활성 시간대: {active_hours}개 (실제 데이터 기준)")
                st.text(f"- 하루 예상 매출 = {avg_per_txn:,.0f}원 × {sel_cnt}건 × {active_hours}시간대 = {daily_pred:,.0f}원")

            st.caption(f"※ {sel_district} {sel_admi_name} · {sel_biz2} · {sel_month}월 {sel_day_label} 기준 예측")

            # ── 고객 세그먼트별 상세 분석 표 ──
            st.divider()
            st.subheader("고객 유형별 매출 분석")

            seg_rows = []
            total_ref_cnt = len(ref)
            for _, combo in combos.iterrows():
                h, s, a = int(combo["hour"]), combo["sex"], int(combo["age"])
                # 해당 조합의 실제 데이터 비중
                seg_mask = (ref["hour"] == h) & (ref["sex"] == s) & (ref["age"] == a)
                seg_cnt  = seg_mask.sum()
                seg_ratio = seg_cnt / total_ref_cnt * 100 if total_ref_cnt > 0 else 0

                # 예측
                input_row = pd.DataFrame([{
                    "sex": s, "age": a, "day": sel_day, "hour": h,
                    "month": sel_month, "admi_cty_no": sel_admi,
                    "card_tpbuz_nm_1": sel_biz1, "card_tpbuz_nm_2": sel_biz2,
                    "cnt": sel_cnt,
                }])
                try:
                    enc = transform_with_saved_encoders(input_row)
                    rp  = model.predict(enc)[0]
                    p   = max(np.expm1(rp) if model_info.get("use_log_target", True) else rp, 0)
                except Exception:
                    p = 0.0

                seg_rows.append({
                    "시간대":     HOUR_MAP.get(h, str(h)),
                    "성별":       "여성" if s == "F" else "남성",
                    "연령대":     AGE_MAP.get(a, f"{a}"),
                    "건당 예측 매출(원)": int(p),
                    "거래 건수":  sel_cnt,
                    "예상 매출(원)":      int(p * sel_cnt),
                    "데이터 비중(%)":     round(seg_ratio, 1),
                })

            if seg_rows:
                seg_df = (
                    pd.DataFrame(seg_rows)
                    .sort_values("데이터 비중(%)", ascending=False)
                    .reset_index(drop=True)
                )

                # 요약 피벗: 연령대 × 성별 건당 매출 평균
                st.markdown("**연령대 × 성별 건당 평균 예측 매출 (원)**")
                pivot = (
                    seg_df.groupby(["연령대", "성별"])["건당 예측 매출(원)"]
                    .mean()
                    .round(0)
                    .astype(int)
                    .unstack("성별")
                    .reindex([v for v in AGE_MAP.values() if v in seg_df["연령대"].values])
                )
                st.dataframe(
                    pivot.style.format("{:,}").background_gradient(cmap="Blues", axis=None),
                    use_container_width=True
                )

                # 전체 세그먼트 상세 표
                st.markdown("**시간대 × 성별 × 연령대 상세**")
                st.dataframe(
                    seg_df.style.format({
                        "건당 예측 매출(원)": "{:,}",
                        "예상 매출(원)": "{:,}",
                        "데이터 비중(%)": "{:.1f}%",
                    }).background_gradient(subset=["건당 예측 매출(원)"], cmap="Greens"),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"예측 오류: {e}")

# =====================================================
# ℹ️ 데이터 정보 (구 탭1)
# =====================================================
with tab_ov:
    st.subheader("학습 데이터 정보")
    n_files = _mi.get("n_files", len(glob.glob(os.path.join(DATASET_DIR, "tbsh_gyeonggi_day_*.csv"))))
    first_name = os.path.basename(sales_path)
    if n_files > 1:
        st.write(f"매출 데이터 파일: `{first_name}` 등 {n_files}개  /  인코딩: `{sales_enc}`")
    else:
        st.write(f"매출 데이터 파일: `{first_name}`  /  인코딩: `{sales_enc}`")
    data_start = _mi.get("data_start")
    data_end   = _mi.get("data_end")
    if data_start and data_end:
        st.write(f"데이터 기간: `{data_start}` ~ `{data_end}`")
    elif "ta_ymd" in df.columns and df["ta_ymd"].notna().any():
        date_min = df["ta_ymd"].min()
        date_max = df["ta_ymd"].max()
        st.write(f"데이터 기간: `{date_min.year}년 {date_min.month}월` ~ `{date_max.year}년 {date_max.month}월`")
    st.write(f"행정동 코드 파일: `{admin_path}`  /  인코딩: `{admin_enc}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 거래 건수",   f"{df.shape[0]:,}")
    c2.metric("데이터 항목 수", f"{df.shape[1]:,}")
    c3.metric("총 매출액",      f"{df['amt'].sum():,.0f}원")
    c4.metric("행정동 수",      f"{len(admin_name_to_code)}개")

    if not admin_ok:
        st.warning("⚠️ city_admin_code.csv 파일이 없어 데이터의 admi_cty_no 값을 그대로 사용합니다.")

    st.subheader("AI 모델 정보")
    enc_map = {c: "Label Encoding" for c in NUM_COLS}
    enc_map.update({c: "Label Encoding (LightGBM 카테고리)" for c in CAT_COLS})
    st.dataframe(pd.DataFrame({
        "변수명": MODEL_FEATURES,
        "인코딩 방식": [enc_map.get(c, "Label Encoding") for c in MODEL_FEATURES],
    }), width='stretch')

    st.subheader("행정동 목록")
    st.write(f"행정동 코드 CSV 로드 개수: {len(admin_df)}개  /  행정동 옵션 개수: {len(admin_name_to_code)}개")
    st.dataframe(admin_df, width='stretch')

    st.subheader("모델 파일 상태")
    paths_check = [SALES_MODEL_PATH, MODEL_INFO_PATH,
                   LABEL_ENCODER_PATH, ONEHOT_ENCODER_PATH, FEATURE_COLUMNS_PATH]
    st.dataframe(pd.DataFrame({
        "파일": paths_check,
        "존재 여부": [os.path.exists(p) for p in paths_check],
    }), width='stretch')

    st.subheader("원본 데이터 미리보기")
    st.dataframe(df.head(30), width='stretch')

# =====================================================
# 📊 소비 트렌드 (구 탭2)
# =====================================================
with tab_eda:
    st.subheader("소비 데이터 현황")
    missing_df = pd.DataFrame({
        "컬럼명":       df.columns,
        "결측치 개수":  df.isnull().sum().values,
        "결측 비율(%)": (df.isnull().mean().values * 100).round(2),
    })
    c1, c2 = st.columns(2)
    c1.metric("전체 결측치 개수", f"{int(df.isnull().sum().sum()):,}")
    c2.metric("중복 행 개수",     f"{int(df.duplicated().sum()):,}")
    st.dataframe(missing_df, width='stretch')

    st.subheader("주요 지표 통계")
    disp_cols = [c for c in ["amt","cnt","log_amt","log_cnt"] if c in df.columns]
    st.dataframe(df[disp_cols].describe().T, width='stretch')

    st.subheader("매출 분포")
    with st.spinner("히스토그램 생성 중..."):
        hist_fig = plot_histograms(df)
    st.pyplot(hist_fig)
    st.download_button("Histogram 이미지 다운로드", data=fig_to_bytes(hist_fig),
                       file_name="histogram_amt_log_amt.png", mime="image/png")

# =====================================================
# ⏰ 시간대·요일 분석 (구 탭3)
# =====================================================
with tab_hm:
    st.subheader("언제 매출이 높을까요?")
    with st.spinner("상관관계 Heatmap 생성 중..."):
        corr_fig, corr_df = plot_corr_heatmap(df)
    st.pyplot(corr_fig)
    st.download_button("상관관계 Heatmap 다운로드", data=fig_to_bytes(corr_fig),
                       file_name="correlation_heatmap.png", mime="image/png")
    st.dataframe(corr_df.round(3), width='stretch')

    st.subheader("연령대별 · 시간대별 소비 현황")
    with st.spinner("연령대 × 시간대 Heatmap 생성 중..."):
        ah_fig, ah_tbl = plot_age_hour_heatmap(df)
    st.pyplot(ah_fig)
    st.download_button("연령대_시간대 Heatmap 다운로드", data=fig_to_bytes(ah_fig),
                       file_name="age_hour_sales_heatmap.png", mime="image/png")
    st.dataframe(ah_tbl.round(2), width='stretch')

    st.subheader("요일 · 시간대별 소비 현황")
    with st.spinner("요일 × 시간대 Heatmap 생성 중..."):
        dh_fig, dh_tbl = plot_day_hour_heatmap(df)
    st.pyplot(dh_fig)
    st.download_button("요일_시간대 Heatmap 다운로드", data=fig_to_bytes(dh_fig),
                       file_name="day_hour_sales_heatmap.png", mime="image/png")
    st.dataframe(dh_tbl.round(2), width='stretch')

    st.subheader("업종 TOP 10 · 시간대별 매출 현황")
    with st.spinner("업종 × 시간대 Heatmap 생성 중..."):
        bh_fig, bh_tbl = plot_biz_hour_heatmap(df, top_n=10)
    st.pyplot(bh_fig)
    st.download_button("업종_시간대 Heatmap 다운로드", data=fig_to_bytes(bh_fig),
                       file_name="biz_hour_sales_heatmap.png", mime="image/png")
    st.dataframe(bh_tbl.round(2), width='stretch')

# =====================================================
# 📈 시계열 예측 (LSTM)
# =====================================================
with tab_lstm:
    st.subheader("날짜별 매출 추이 & 미래 예측")
    st.caption("조건을 선택하면 해당 지역·업종의 실제 매출 추이와 LSTM 미래 예측을 보여줍니다.")

    # ── 필터 ──
    lt1, lt2 = st.columns(2)
    with lt1:
        if admin_ok:
            lt_district  = st.selectbox("지역 (시/구)", ["전체"] + admin_district_list, key="lt_dist")
            lt_dong_opts = ["전체"] + admin_district_to_dongs.get(lt_district, []) if lt_district != "전체" else ["전체"]
            lt_admi_name = st.selectbox("동네 선택", lt_dong_opts, key="lt_dong")
        else:
            lt_district = lt_admi_name = "전체"
    with lt2:
        lt_biz1      = st.selectbox("업종 대분류", ["전체"] + sorted(df["card_tpbuz_nm_1"].dropna().unique()), key="lt_biz1")
        lt_biz2_opts = ["전체"] + sorted(df[df["card_tpbuz_nm_1"] == lt_biz1]["card_tpbuz_nm_2"].dropna().unique()) if lt_biz1 != "전체" else ["전체"]
        lt_biz2      = st.selectbox("업종 중분류", lt_biz2_opts, key="lt_biz2")

    # ── 필터 적용 ──
    lt_df = df.copy()
    if lt_district != "전체" and lt_admi_name != "전체" and admin_ok:
        lt_admi_code = admin_name_to_code.get(lt_admi_name)
        if lt_admi_code:
            lt_df = lt_df[lt_df["admi_cty_no"] == lt_admi_code]
    elif lt_district != "전체" and admin_ok:
        lt_codes = set(admin_name_to_code.get(d) for d in admin_district_to_dongs.get(lt_district, []))
        lt_df = lt_df[lt_df["admi_cty_no"].isin(lt_codes)]
    if lt_biz1 != "전체":
        lt_df = lt_df[lt_df["card_tpbuz_nm_1"] == lt_biz1]
    if lt_biz2 != "전체":
        lt_df = lt_df[lt_df["card_tpbuz_nm_2"] == lt_biz2]

    if lt_df.empty:
        st.warning("선택 조건에 해당하는 데이터가 없습니다.")
    else:
        # ta_ymd는 로드 시 이미 datetime으로 변환됨
        date_col = "ta_ymd" if "ta_ymd" in lt_df.columns else "date"
        lt_df["_date"] = pd.to_datetime(lt_df[date_col], errors="coerce")
        daily = lt_df.groupby("_date")["amt"].sum().reset_index().sort_values("_date")
        daily = daily.rename(columns={"_date": "date"})

        FORECAST_DAYS = st.slider("미래 예측 기간 (일)", 7, 60, 30)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"].astype(str), y=daily["amt"],
            name="실제 매출", line=dict(color="#58a6ff")
        ))

        # LSTM 미래 예측 (모델 있을 때만)
        if os.path.exists(LSTM_MODEL_PATH):
            try:
                import torch
                lstm_data    = joblib.load(LSTM_MODEL_PATH)
                model_lstm   = lstm_data["model"]
                scaler_lstm  = lstm_data["scaler"]
                seq_len      = lstm_data["seq_len"]

                vals = daily["amt"].values.astype("float32").reshape(-1, 1)
                if len(vals) >= seq_len:
                    scaled   = scaler_lstm.transform(vals)
                    last_seq = torch.tensor(scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)
                    preds = []
                    model_lstm.eval()
                    with torch.no_grad():
                        seq = last_seq.clone()
                        for _ in range(FORECAST_DAYS):
                            out = model_lstm(seq)
                            preds.append(out.item())
                            next_val = out.unsqueeze(1)
                            seq = torch.cat([seq[:,1:,:], next_val], dim=1)
                    pred_vals    = scaler_lstm.inverse_transform([[p] for p in preds])[:,0]
                    last_date    = daily["date"].iloc[-1]
                    future_dates = [str((last_date + pd.Timedelta(days=i+1)).date()) for i in range(FORECAST_DAYS)]

                    fig.add_trace(go.Scatter(
                        x=future_dates, y=pred_vals,
                        name="LSTM 예측", line=dict(color="#3fb950", dash="dash")
                    ))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("예측 평균 일매출", f"{pred_vals.mean():,.0f}원")
                    c2.metric("예측 최고 매출일", f"{pred_vals.max():,.0f}원")
                    c3.metric("예측 최저 매출일", f"{pred_vals.min():,.0f}원")
            except Exception as e:
                st.warning(f"LSTM 예측 오류: {e}")

        fig.update_layout(
            title=f"일별 매출 추이 ({lt_district} {lt_admi_name} · {lt_biz2})",
            xaxis_title="날짜", yaxis_title="매출액 (원)",
            hovermode="x unified", height=440
        )
        st.plotly_chart(fig, use_container_width=True)

        d1, d2 = st.columns(2)
        d1.metric("기간 평균 일매출", f"{daily['amt'].mean():,.0f}원")
        d2.metric("기간 총 매출",     f"{daily['amt'].sum():,.0f}원")

# =====================================================
# 👥 고객 군집 분석 (Autoencoder + KMeans)
# =====================================================

with tab_cluster:
    st.subheader("우리 동네 고객은 어떤 유형일까요?")
    st.caption("조건을 선택하면 해당 지역·업종·기간의 고객 소비 패턴을 분석합니다.")

    # ── 필터 ──
    cl1, cl2 = st.columns(2)
    with cl1:
        if admin_ok:
            cl_district  = st.selectbox("지역 (시/구)", ["전체"] + admin_district_list, key="cl_dist")
            cl_dong_opts = ["전체"] + admin_district_to_dongs.get(cl_district, []) if cl_district != "전체" else ["전체"]
            cl_admi_name = st.selectbox("동네 선택", cl_dong_opts, key="cl_dong")
        else:
            cl_district = cl_admi_name = "전체"
        cl_month = st.selectbox("월", ["전체"] + [f"{m}월" for m in range(1, 13)], key="cl_month")
        all_days  = list(DAY_MAP.values())
        cl_days   = st.multiselect("요일 (전체 선택 = 모든 요일)", all_days, default=all_days, key="cl_days")
    with cl2:
        cl_biz1      = st.selectbox("업종 대분류", ["전체"] + sorted(df["card_tpbuz_nm_1"].dropna().unique()), key="cl_biz1")
        cl_biz2_opts = ["전체"] + sorted(df[df["card_tpbuz_nm_1"] == cl_biz1]["card_tpbuz_nm_2"].dropna().unique()) if cl_biz1 != "전체" else ["전체"]
        cl_biz2      = st.selectbox("업종 중분류", cl_biz2_opts, key="cl_biz2")

    # ── 필터 적용 ──
    cl_df = df.copy()
    if cl_district != "전체" and cl_admi_name != "전체" and admin_ok:
        cl_code = admin_name_to_code.get(cl_admi_name)
        if cl_code:
            cl_df = cl_df[cl_df["admi_cty_no"] == cl_code]
    elif cl_district != "전체" and admin_ok:
        cl_codes = set(admin_name_to_code.get(d) for d in admin_district_to_dongs.get(cl_district, []))
        cl_df = cl_df[cl_df["admi_cty_no"].isin(cl_codes)]
    if cl_biz1 != "전체":
        cl_df = cl_df[cl_df["card_tpbuz_nm_1"] == cl_biz1]
    if cl_biz2 != "전체":
        cl_df = cl_df[cl_df["card_tpbuz_nm_2"] == cl_biz2]
    if cl_month != "전체":
        cl_m = int(cl_month.replace("월", ""))
        cl_df = cl_df[cl_df["month"] == cl_m]
    if cl_days and len(cl_days) < len(all_days):
        day_rev = {v: k for k, v in DAY_MAP.items()}
        cl_day_nums = [day_rev[d] for d in cl_days]
        cl_df = cl_df[cl_df["day"].isin(cl_day_nums)]

    if cl_df.empty:
        st.warning("선택 조건에 해당하는 데이터가 없습니다.")
    else:
        st.caption(f"분석 데이터: {len(cl_df):,}건")

        ch1, ch2 = st.columns(2)

        # 연령대별 매출 비중
        with ch1:
            age_grp = cl_df.groupby("age")["amt"].sum().reset_index()
            age_grp["연령대"] = age_grp["age"].map(AGE_MAP)
            fig_age = px.bar(age_grp, x="연령대", y="amt", title="연령대별 매출",
                             labels={"amt": "매출액 (원)"}, color="amt",
                             color_continuous_scale="Blues")
            fig_age.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_age, use_container_width=True)

        # 성별 매출 비중
        with ch2:
            sex_grp = cl_df.groupby("sex")["amt"].sum().reset_index()
            sex_grp["성별"] = sex_grp["sex"].map(SEX_MAP)
            fig_sex = px.pie(sex_grp, names="성별", values="amt", title="성별 매출 비중",
                             color_discrete_map={"여성": "#f472b6", "남성": "#60a5fa"})
            fig_sex.update_layout(height=320)
            st.plotly_chart(fig_sex, use_container_width=True)

        ch3, ch4 = st.columns(2)

        # 시간대별 매출
        with ch3:
            hour_grp = cl_df.groupby("hour")["amt"].sum().reset_index()
            hour_grp["시간대"] = hour_grp["hour"].map(HOUR_MAP)
            fig_hour = px.bar(hour_grp, x="시간대", y="amt", title="시간대별 매출",
                              labels={"amt": "매출액 (원)"}, color="amt",
                              color_continuous_scale="Greens")
            fig_hour.update_layout(showlegend=False, coloraxis_showscale=False, height=320,
                                   xaxis_tickangle=-30)
            st.plotly_chart(fig_hour, use_container_width=True)

        # 요일별 매출
        with ch4:
            day_grp = cl_df.groupby("day")["amt"].sum().reset_index()
            day_grp["요일"] = day_grp["day"].map(DAY_MAP)
            fig_day = px.bar(day_grp, x="요일", y="amt", title="요일별 매출",
                             labels={"amt": "매출액 (원)"}, color="amt",
                             color_continuous_scale="Oranges")
            fig_day.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_day, use_container_width=True)

        # 업종 TOP10
        st.subheader("업종별 매출 TOP 10")
        top_biz = (cl_df.groupby("card_tpbuz_nm_2")["amt"].sum()
                   .nlargest(10).reset_index()
                   .rename(columns={"card_tpbuz_nm_2": "업종", "amt": "매출액 (원)"}))
        fig_biz = px.bar(top_biz, x="매출액 (원)", y="업종", orientation="h",
                         color="매출액 (원)", color_continuous_scale="Purples")
        fig_biz.update_layout(coloraxis_showscale=False, height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_biz, use_container_width=True)

        # Autoencoder 군집 모델 요약 (있을 때만)
        if os.path.exists(CLUSTER_MODEL_PATH):
            st.divider()
            st.subheader("전체 고객 유형 분류 (Autoencoder 모델)")
            try:
                cluster_data  = joblib.load(CLUSTER_MODEL_PATH)
                cluster_stats = cluster_data["stats"]
                cluster_names = cluster_data.get("names", [])
                cluster_centers = cluster_data["centers"]
                ratios = cluster_data.get("ratios", [])
                st.dataframe(cluster_stats, use_container_width=True, hide_index=True)
                viz_df = pd.DataFrame(cluster_centers, columns=["x", "y"])
                viz_df["유형"] = cluster_names
                viz_df["비율(%)"] = ratios
                fig_c = px.scatter(viz_df, x="x", y="y", size="비율(%)", color="유형",
                                   text="유형", size_max=60, height=380,
                                   title="고객 유형 분포 (잠재 공간)")
                fig_c.update_traces(textposition="top center")
                st.plotly_chart(fig_c, use_container_width=True)
            except Exception as e:
                st.error(f"군집 모델 로드 오류: {e}")

# =====================================================
# 📝 AI 리포트 (OpenAI API)
# =====================================================
with tab_ai:
    st.subheader("AI가 분석한 우리 동네 소비 리포트")
    st.caption("데이터를 요약해 GPT-4o가 사장님을 위한 인사이트 리포트를 작성합니다.")

    # API 키: secrets 우선, 없으면 입력창
    api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password",
                                placeholder="sk-proj-...")

    # ── 필터 ──
    ai1, ai2 = st.columns(2)
    with ai1:
        if admin_ok:
            ai_district  = st.selectbox("지역 (시/구)", ["전체"] + admin_district_list, key="ai_dist")
            ai_dong_opts = ["전체"] + admin_district_to_dongs.get(ai_district, []) if ai_district != "전체" else ["전체"]
            ai_admi_name = st.selectbox("동네 선택", ai_dong_opts, key="ai_dong")
        else:
            ai_district = ai_admi_name = "전체"
    with ai2:
        ai_biz1      = st.selectbox("업종 대분류", ["전체"] + sorted(df["card_tpbuz_nm_1"].dropna().unique()), key="ai_biz1")
        ai_biz2_opts = ["전체"] + sorted(df[df["card_tpbuz_nm_1"] == ai_biz1]["card_tpbuz_nm_2"].dropna().unique()) if ai_biz1 != "전체" else ["전체"]
        ai_biz2      = st.selectbox("업종 중분류", ai_biz2_opts, key="ai_biz2")

    if st.button("📝 AI 리포트 생성", type="primary"):
        if not api_key:
            st.warning("OpenAI API Key를 입력해주세요.")
        else:
            with st.spinner("GPT-4o가 리포트를 작성 중입니다..."):
                try:
                    from openai import OpenAI

                    # ── 필터 적용 ──
                    filtered = df.copy()
                    if ai_district != "전체" and ai_admi_name != "전체" and admin_ok:
                        ai_code = admin_name_to_code.get(ai_admi_name)
                        if ai_code:
                            filtered = filtered[filtered["admi_cty_no"] == ai_code]
                    elif ai_district != "전체" and admin_ok:
                        ai_codes = set(admin_name_to_code.get(d) for d in admin_district_to_dongs.get(ai_district, []))
                        filtered = filtered[filtered["admi_cty_no"].isin(ai_codes)]
                    if ai_biz1 != "전체":
                        filtered = filtered[filtered["card_tpbuz_nm_1"] == ai_biz1]
                    if ai_biz2 != "전체":
                        filtered = filtered[filtered["card_tpbuz_nm_2"] == ai_biz2]

                    if filtered.empty:
                        st.warning("선택 조건에 해당하는 데이터가 없습니다.")
                    else:
                        top_biz2  = filtered.groupby("card_tpbuz_nm_2")["amt"].sum().nlargest(5)
                        top_hour  = filtered.groupby("hour")["amt"].sum().idxmax()
                        top_day   = filtered.groupby("day")["amt"].sum().idxmax()
                        top_age   = filtered.groupby("age")["amt"].sum().idxmax()
                        total_amt = filtered["amt"].sum()
                        avg_amt   = filtered["amt"].mean()

                        hour_label = HOUR_MAP.get(top_hour, str(top_hour))
                        day_label  = DAY_MAP.get(top_day, str(top_day))
                        age_label  = AGE_MAP.get(top_age, str(top_age))

                        loc_label = f"{ai_district} {ai_admi_name}".strip()
                        biz_label = f"{ai_biz1} > {ai_biz2}" if ai_biz2 != "전체" else ai_biz1

                        summary = f"""
경기도 카드 소비 데이터 분석 요약:
- 분석 지역: {loc_label if loc_label != "전체 전체" else "경기도 전체"}
- 분석 업종: {biz_label}
- 분석 기간: {_mi.get('data_start','알 수 없음')} ~ {_mi.get('data_end','알 수 없음')}
- 총 매출액: {total_amt:,.0f}원
- 건당 평균 매출: {avg_amt:,.0f}원
- 매출이 가장 높은 시간대: {hour_label}
- 매출이 가장 높은 요일: {day_label}
- 주요 소비 연령대: {age_label}
- 매출 상위 업종 중분류 TOP 5: {', '.join([f'{k}({v:,.0f}원)' for k, v in top_biz2.items()])}
"""

                        client = OpenAI(api_key=api_key)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            max_tokens=1024,
                            messages=[
                                {"role": "system", "content": "당신은 소상공인을 위한 경영 컨설턴트입니다."},
                                {"role": "user", "content": f"""아래 경기도 카드 소비 데이터 분석 결과를 바탕으로 사장님이 바로 활용할 수 있는 인사이트 리포트를 작성해주세요.

{summary}

다음 형식으로 작성해주세요:
1. 핵심 요약 (3줄 이내)
2. 주목할 소비 패턴 (2~3가지)
3. 사장님께 드리는 제안 (2~3가지 실용적인 조언)

전문 용어보다는 사장님이 바로 이해할 수 있는 쉬운 언어로 작성해주세요."""}
                            ]
                        )
                        report_text = response.choices[0].message.content
                        st.markdown(report_text)
                        st.download_button("리포트 저장 (텍스트)", data=report_text,
                                           file_name="ai_report.txt", mime="text/plain")

                        input_tokens  = response.usage.prompt_tokens
                        output_tokens = response.usage.completion_tokens
                        cost_usd = (input_tokens * 2.5 + output_tokens * 10) / 1_000_000
                        cost_krw = cost_usd * 1380
                        st.caption(
                            f"💸 이번 리포트 비용: **${cost_usd:.4f}** (약 {cost_krw:.1f}원) "
                            f"| 입력 {input_tokens:,}토큰 + 출력 {output_tokens:,}토큰"
                        )
                except Exception as e:
                    st.error(f"AI 리포트 생성 오류: {e}")
