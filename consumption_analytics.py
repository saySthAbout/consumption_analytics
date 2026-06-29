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
    "41591": "화성시", "41593": "화성시(동탄1)", "41595": "화성시(동탄2)", "41597": "화성시(동탄3)",
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
# 4. 매출 예측 (탭 1 - 사장님 첫 화면)
# =====================================================
with tab_pred:
    st.subheader("우리 가게 예상 매출은 얼마일까요?")
    st.caption("성별, 연령, 요일, 시간대, 동네, 업종을 선택하면 예상 매출액을 알려드립니다.")

    p1, p2 = st.columns(2)

    with p1:
        sel_sex_label  = st.selectbox("고객 성별",   ["남성", "여성"])
        sel_sex        = SEX_REVERSE_MAP[sel_sex_label]

        sel_age_label  = st.selectbox("고객 연령대", list(AGE_MAP.values()), index=3)
        sel_age        = {v: k for k, v in AGE_MAP.items()}[sel_age_label]

        sel_day_label  = st.selectbox("요일",        list(DAY_MAP.values()), index=4)
        sel_day        = {v: k for k, v in DAY_MAP.items()}[sel_day_label]

        sel_hour_label = st.selectbox("시간대",      list(HOUR_MAP.values()), index=4)
        sel_hour       = {v: k for k, v in HOUR_MAP.items()}[sel_hour_label]

        sel_month      = st.selectbox("월", list(range(1, 13)), index=0,
                                      format_func=lambda x: f"{x}월")

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

        sel_biz1 = st.selectbox("업종 대분류",
                                sorted(df["card_tpbuz_nm_1"].dropna().unique()))
        biz2_opts = sorted(df[df["card_tpbuz_nm_1"] == sel_biz1]["card_tpbuz_nm_2"]
                           .dropna().unique())
        if not biz2_opts:
            st.warning("선택한 대분류에 해당하는 중분류가 없습니다.")
            sel_biz2 = None
        else:
            sel_biz2 = st.selectbox("업종 중분류", biz2_opts)

        avg_cnt = int(round(
            df[df["card_tpbuz_nm_2"] == sel_biz2]["cnt"].mean()
        )) if sel_biz2 and "cnt" in df.columns else 10
        sel_cnt = st.number_input(
            f"예상 거래 건수  ※ {sel_biz2} 평균: {avg_cnt}건",
            min_value=1, value=avg_cnt, step=1
        )

        if sel_biz2 is not None:
            input_df = pd.DataFrame([{
                "sex": sel_sex, "age": sel_age, "day": sel_day, "hour": sel_hour,
                "month": sel_month, "admi_cty_no": sel_admi,
                "card_tpbuz_nm_1": sel_biz1, "card_tpbuz_nm_2": sel_biz2,
                "cnt": sel_cnt,
            }])

            st.subheader("입력 조건 확인")
            st.dataframe(pd.DataFrame([{
                "성별": sel_sex_label, "연령대": sel_age_label, "요일": sel_day_label,
                "시간대": sel_hour_label, "월": f"{sel_month}월",
                "동네": sel_admi_name,
                "업종 대분류": sel_biz1, "업종 중분류": sel_biz2, "거래건수": sel_cnt,
            }]), width='stretch')
        else:
            input_df = None

    if st.button("예상 매출액 예측하기", type="primary") and input_df is not None:
        try:
            model, model_info = load_saved_model()
            encoded = transform_with_saved_encoders(input_df)

            with st.expander("🔍 상세 디버그 정보"):
                st.write("원본 입력값:", input_df.to_dict(orient="records")[0])
                raw_pred = model.predict(encoded)[0]
                st.write(f"모델 raw 출력 (log 공간): {raw_pred:.6f}")
                st.write(f"use_log_target: {model_info.get('use_log_target')}")
                st.dataframe(encoded, width='stretch')

            pred    = np.expm1(raw_pred) if model_info.get("use_log_target", True) else raw_pred
            pred    = max(pred, 0)
            per_txn = pred / sel_cnt if sel_cnt > 0 else 0

            # 동네·업종 평균 매출과 비교
            mask = (df["card_tpbuz_nm_2"] == sel_biz2) if sel_biz2 else pd.Series([False]*len(df))
            area_avg = df.loc[mask, "amt"].mean() if mask.any() else None

            st.success(f"예상 매출액: **{pred:,.0f}원**  (건당 평균 {per_txn:,.0f}원 × {sel_cnt}건)")
            if area_avg:
                diff_pct = (pred - area_avg) / area_avg * 100
                arrow = "▲" if diff_pct >= 0 else "▼"
                color = "🟢" if diff_pct >= 0 else "🔴"
                st.info(f"{color} 해당 업종 전체 평균({area_avg:,.0f}원)보다 {arrow} {abs(diff_pct):.1f}% {'높습니다' if diff_pct >= 0 else '낮습니다'}")
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
    st.caption("LSTM 딥러닝 모델로 과거 매출 패턴을 학습해 향후 매출을 예측합니다.")

    if not os.path.exists(LSTM_MODEL_PATH):
        st.info("📦 LSTM 모델이 아직 없습니다. Kaggle에서 `kaggle_lstm.ipynb`를 학습한 뒤 `model/lstm_model.pkl`을 업로드해주세요.")
    else:
        try:
            import torch
            import plotly.graph_objects as go
            lstm_data = joblib.load(LSTM_MODEL_PATH)
            model_lstm   = lstm_data["model"]
            scaler_lstm  = lstm_data["scaler"]
            seq_len      = lstm_data["seq_len"]
            history_vals = lstm_data["history"]   # numpy array (날짜별 총 매출)
            history_dates= lstm_data["dates"]     # list of date strings

            FORECAST_DAYS = st.slider("예측 기간 (일)", 7, 60, 30)

            # 예측
            model_lstm.eval()
            last_seq = torch.tensor(scaler_lstm.transform(
                history_vals[-seq_len:].reshape(-1,1)), dtype=torch.float32).unsqueeze(0)
            preds = []
            with torch.no_grad():
                seq = last_seq.clone()
                for _ in range(FORECAST_DAYS):
                    out = model_lstm(seq)
                    preds.append(out.item())
                    next_val = out.unsqueeze(0)
                    seq = torch.cat([seq[:,1:,:], next_val.unsqueeze(2)], dim=1)
            pred_vals = scaler_lstm.inverse_transform(
                [[p] for p in preds])[:,0]

            last_date = pd.to_datetime(history_dates[-1])
            future_dates = [str((last_date + pd.Timedelta(days=i+1)).date())
                            for i in range(FORECAST_DAYS)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_dates[-90:], y=history_vals[-90:],
                                     name="실제 매출", line=dict(color="#58a6ff")))
            fig.add_trace(go.Scatter(x=future_dates, y=pred_vals,
                                     name="예측 매출", line=dict(color="#3fb950", dash="dash")))
            fig.update_layout(title="일별 매출 추이 및 예측",
                              xaxis_title="날짜", yaxis_title="매출액 (원)",
                              hovermode="x unified", height=420)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("예측 기간 평균 매출", f"{pred_vals.mean():,.0f}원/일")
            c2.metric("예측 최고 매출일",    f"{pred_vals.max():,.0f}원")
            c3.metric("예측 최저 매출일",    f"{pred_vals.min():,.0f}원")
        except Exception as e:
            st.error(f"LSTM 모델 로드 오류: {e}")

# =====================================================
# 👥 고객 군집 분석 (Autoencoder + KMeans)
# =====================================================
with tab_cluster:
    st.subheader("우리 동네 고객은 어떤 유형일까요?")
    st.caption("Autoencoder로 소비 패턴을 압축하고 군집화해 고객 유형을 분류합니다.")

    if not os.path.exists(CLUSTER_MODEL_PATH):
        st.info("📦 군집 모델이 아직 없습니다. Kaggle에서 `kaggle_cluster.ipynb`를 학습한 뒤 `model/cluster_model.pkl`을 업로드해주세요.")
    else:
        try:
            import plotly.express as px
            cluster_data   = joblib.load(CLUSTER_MODEL_PATH)
            cluster_labels = cluster_data["labels"]       # numpy array
            cluster_centers= cluster_data["centers"]      # 2D (n_clusters, 2) for viz
            cluster_stats  = cluster_data["stats"]        # DataFrame: 군집별 특성
            cluster_names  = cluster_data.get("names", [f"유형 {i+1}" for i in range(len(cluster_stats))])

            st.subheader("고객 유형별 특성")
            st.dataframe(cluster_stats, width="stretch")

            st.subheader("고객 분포 시각화")
            viz_df = pd.DataFrame(cluster_centers, columns=["x", "y"])
            viz_df["유형"] = cluster_names
            viz_df["비율(%)"] = cluster_data.get("ratios", [100/len(cluster_names)]*len(cluster_names))
            fig_c = px.scatter(viz_df, x="x", y="y", size="비율(%)", color="유형",
                               text="유형", size_max=60, height=400,
                               title="고객 유형 분포 (Autoencoder 잠재 공간)")
            fig_c.update_traces(textposition="top center")
            st.plotly_chart(fig_c, use_container_width=True)
        except Exception as e:
            st.error(f"군집 모델 로드 오류: {e}")

# =====================================================
# 📝 AI 리포트 (Claude API)
# =====================================================
with tab_ai:
    st.subheader("AI가 분석한 우리 동네 소비 리포트")
    st.caption("데이터를 요약해 Claude AI가 사장님을 위한 인사이트 리포트를 작성합니다.")

    # API 키: secrets 우선, 없으면 입력창
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = st.text_input("Anthropic API Key", type="password",
                                placeholder="sk-ant-...")

    # 리포트 생성 옵션
    report_district = st.selectbox("분석할 지역", admin_district_list if admin_ok else ["전체"])
    report_biz      = st.selectbox("분석할 업종 대분류", ["전체"] + sorted(df["card_tpbuz_nm_1"].dropna().unique().tolist()))

    if st.button("📝 AI 리포트 생성", type="primary"):
        if not api_key:
            st.warning("Anthropic API Key를 입력해주세요.")
        else:
            with st.spinner("Claude AI가 리포트를 작성 중입니다..."):
                try:
                    import anthropic

                    # 데이터 요약 생성
                    filtered = df.copy()
                    if report_biz != "전체":
                        filtered = filtered[filtered["card_tpbuz_nm_1"] == report_biz]

                    top_biz2  = filtered.groupby("card_tpbuz_nm_2")["amt"].sum().nlargest(5)
                    top_hour  = filtered.groupby("hour")["amt"].sum().idxmax()
                    top_day   = filtered.groupby("day")["amt"].sum().idxmax()
                    top_age   = filtered.groupby("age")["amt"].sum().idxmax()
                    total_amt = filtered["amt"].sum()
                    avg_amt   = filtered["amt"].mean()

                    hour_label = HOUR_MAP.get(top_hour, str(top_hour))
                    day_label  = DAY_MAP.get(top_day, str(top_day))
                    age_label  = AGE_MAP.get(top_age, str(top_age))

                    summary = f"""
경기도 카드 소비 데이터 분석 요약:
- 분석 지역: {report_district}
- 분석 업종: {report_biz}
- 분석 기간: {_mi.get('data_start','알 수 없음')} ~ {_mi.get('data_end','알 수 없음')}
- 총 매출액: {total_amt:,.0f}원
- 건당 평균 매출: {avg_amt:,.0f}원
- 매출이 가장 높은 시간대: {hour_label}
- 매출이 가장 높은 요일: {day_label}
- 주요 소비 연령대: {age_label}
- 매출 상위 업종 중분류 TOP 5: {', '.join([f'{k}({v:,.0f}원)' for k, v in top_biz2.items()])}
"""

                    client = anthropic.Anthropic(api_key=api_key)
                    message = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": f"""당신은 소상공인을 위한 경영 컨설턴트입니다.
아래 경기도 카드 소비 데이터 분석 결과를 바탕으로 사장님이 바로 활용할 수 있는 인사이트 리포트를 작성해주세요.

{summary}

다음 형식으로 작성해주세요:
1. 핵심 요약 (3줄 이내)
2. 주목할 소비 패턴 (2~3가지)
3. 사장님께 드리는 제안 (2~3가지 실용적인 조언)

전문 용어보다는 사장님이 바로 이해할 수 있는 쉬운 언어로 작성해주세요."""
                        }]
                    )
                    report_text = message.content[0].text
                    st.markdown(report_text)
                    st.download_button("리포트 저장 (텍스트)", data=report_text,
                                       file_name="ai_report.txt", mime="text/plain")
                except Exception as e:
                    st.error(f"AI 리포트 생성 오류: {e}")
