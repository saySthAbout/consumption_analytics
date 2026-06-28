import os
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
    page_title="화성시 소비 트렌드 분석 및 매출 예측 AI",
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
    os.path.join(DATASET_DIR, "hwaseong_admin_code.csv"),
    os.path.join(BASE_DIR,    "hwaseong_admin_code.csv"),
]

MODEL_FEATURES = ["sex", "age", "day", "hour",
                  "admi_cty_no", "card_tpbuz_nm_1", "card_tpbuz_nm_2"]
LABEL_COLS  = ["age", "day", "hour"]
ONEHOT_COLS = ["sex", "admi_cty_no", "card_tpbuz_nm_1", "card_tpbuz_nm_2"]

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
        "hwaseong_admin_code.csv 파일을 찾을 수 없습니다.\n"
        "dataset/hwaseong_admin_code.csv 위치를 확인해주세요."
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
    raise ValueError(f"hwaseong_admin_code.csv 읽기 실패: {last_err}")


def build_admin_maps(admin_df):
    admin_df = admin_df.copy()
    admin_df["admi_cty_no"]   = admin_df["admi_cty_no"].astype(int)
    admin_df["admi_cty_name"] = admin_df["admi_cty_name"].astype(str).str.strip()
    name_options = admin_df["admi_cty_name"].tolist()
    name_to_code = dict(zip(admin_df["admi_cty_name"], admin_df["admi_cty_no"]))
    code_to_name = dict(zip(admin_df["admi_cty_no"],   admin_df["admi_cty_name"]))
    return name_options, name_to_code, code_to_name


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

    encoded_X       = pd.concat([X[LABEL_COLS].reset_index(drop=True),
                                 ohe_df.reset_index(drop=True)], axis=1)
    feature_columns = encoded_X.columns.tolist()

    joblib.dump(label_encoders,  LABEL_ENCODER_PATH)
    joblib.dump(ohe,             ONEHOT_ENCODER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    return encoded_X


def load_encoders():
    return (joblib.load(LABEL_ENCODER_PATH),
            joblib.load(ONEHOT_ENCODER_PATH),
            joblib.load(FEATURE_COLUMNS_PATH))


def transform_with_saved_encoders(X):
    X = X.copy()
    label_encoders, ohe, feature_columns = load_encoders()
    for col in LABEL_COLS:
        X[col] = X[col].astype(str)
        le = label_encoders[col]
        unknown = set(X[col].unique()) - set(le.classes_)
        if unknown:
            raise ValueError(f"'{col}' 컬럼에 학습되지 않은 값: {unknown}")
        X[col] = le.transform(X[col])
    ohe_arr   = ohe.transform(X[ONEHOT_COLS].astype(str))
    ohe_names = ohe.get_feature_names_out(ONEHOT_COLS).tolist()
    ohe_df    = pd.DataFrame(ohe_arr, columns=ohe_names, index=X.index)
    encoded_X = pd.concat([X[LABEL_COLS].reset_index(drop=True),
                           ohe_df.reset_index(drop=True)], axis=1)
    return encoded_X.reindex(columns=feature_columns, fill_value=0)


# =========================================================
# 3. 모델 학습 / 저장 / 예측
# =========================================================
def train_and_save_model(df, sample_size=100000, use_log_target=True, model_name="RandomForest"):
    model_df = df[MODEL_FEATURES + ["amt", "log_amt"]].dropna().copy()
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
        model = RandomForestRegressor(n_estimators=100, max_depth=18,
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
        "model_name":     model_name,
        "use_log_target": use_log_target,
        "sample_size":    sample_size,
        "features":       MODEL_FEATURES,
        "label_cols":     LABEL_COLS,
        "onehot_cols":    ONEHOT_COLS,
        "metrics":        metrics,
    }
    joblib.dump(model,      SALES_MODEL_PATH)
    joblib.dump(model_info, MODEL_INFO_PATH)
    return model, model_info, X_test, y_test_real, pred_real


def load_saved_model():
    return joblib.load(SALES_MODEL_PATH), joblib.load(MODEL_INFO_PATH)


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
st.title("화성시 소비 트렌드 분석 및 매출 예측 AI")
st.caption("성별, 연령, 요일, 시간대, 행정동, 업종을 입력하면 예상 매출금액을 예측합니다.")
st.caption(f"실행 경로: {BASE_DIR}")

# ── 사이드바 (탭과 무관하게 항상 표시) ─────────────────
st.sidebar.header("모델 설정")
if st.sidebar.button("Streamlit 캐시 초기화"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("캐시 초기화 완료. 새로고침해주세요.")

sample_size    = st.sidebar.slider("학습 샘플 수", 10000, 300000, 300000, step=10000)
use_log_target = st.sidebar.checkbox("매출금액 로그 변환 후 학습", value=True)
model_name     = st.sidebar.selectbox("모델 선택", ["RandomForest", "LightGBM", "LinearRegression"])

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
    admin_name_options, admin_name_to_code, _ = build_admin_maps(admin_df)
    admin_ok = True
except Exception:
    admin_ok           = False
    admin_df           = pd.DataFrame(columns=["admi_cty_no", "admi_cty_name"])
    admin_name_options = sorted(df["admi_cty_no"].dropna().astype(str).unique().tolist())
    admin_name_to_code = {v: v for v in admin_name_options}
    admin_enc          = "-"
    admin_path         = "-"

if st.sidebar.button("모델 재학습 및 저장"):
    with st.spinner("모델과 인코더를 새로 학습 중..."):
        train_and_save_model(df, sample_size, use_log_target, model_name)
    st.sidebar.success("model/, encoders/ 폴더에 저장 완료")

# ── 탭 (항상 생성) ───────────────────────────────────────
tab_ov, tab_eda, tab_hm, tab_model, tab_pred, tab_report = st.tabs([
    "1. 데이터 개요",
    "2. 탐색적 데이터 분석",
    "3. Heatmap",
    "4. 모델 학습",
    "5. 매출 예측",
    "6. 보고서 문구",
])

# =====================================================
# 1. 데이터 개요
# =====================================================
with tab_ov:
    st.subheader("1. 데이터 수집 및 임포팅")
    st.write(f"매출 데이터 파일: `{sales_path}`  /  인코딩: `{sales_enc}`")
    st.write(f"행정동 코드 파일: `{admin_path}`  /  인코딩: `{admin_enc}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 개수",        f"{df.shape[0]:,}")
    c2.metric("열 개수",        f"{df.shape[1]:,}")
    c3.metric("총 매출액",      f"{df['amt'].sum():,.0f}원")
    c4.metric("행정동 선택 수", f"{len(admin_name_options)}개")

    if not admin_ok:
        st.warning("⚠️ hwaseong_admin_code.csv 파일이 없어 데이터의 admi_cty_no 값을 그대로 사용합니다.")

    st.subheader("모델 입력 변수 (7개)")
    st.dataframe(pd.DataFrame({
        "변수명": MODEL_FEATURES,
        "인코딩 방식": ["One-Hot Encoding", "Label Encoding", "Label Encoding",
                       "Label Encoding", "One-Hot Encoding",
                       "One-Hot Encoding", "One-Hot Encoding"],
    }), width='stretch')

    st.subheader("행정동 코드 데이터")
    st.write(f"행정동 코드 CSV 로드 개수: {len(admin_df)}개  /  행정동 옵션 개수: {len(admin_name_options)}개")
    st.dataframe(admin_df, width='stretch')

    st.subheader("모델 / 인코더 저장 경로 및 존재 여부")
    paths_check = [SALES_MODEL_PATH, MODEL_INFO_PATH,
                   LABEL_ENCODER_PATH, ONEHOT_ENCODER_PATH, FEATURE_COLUMNS_PATH]
    st.dataframe(pd.DataFrame({
        "파일": paths_check,
        "존재 여부": [os.path.exists(p) for p in paths_check],
    }), width='stretch')

    st.subheader("매출 데이터 미리보기")
    st.dataframe(df.head(30), width='stretch')

# =====================================================
# 2. 탐색적 데이터 분석
# =====================================================
with tab_eda:
    st.subheader("2.1 결측치 및 중복값 통계")
    missing_df = pd.DataFrame({
        "컬럼명":       df.columns,
        "결측치 개수":  df.isnull().sum().values,
        "결측 비율(%)": (df.isnull().mean().values * 100).round(2),
    })
    c1, c2 = st.columns(2)
    c1.metric("전체 결측치 개수", f"{int(df.isnull().sum().sum()):,}")
    c2.metric("중복 행 개수",     f"{int(df.duplicated().sum()):,}")
    st.dataframe(missing_df, width='stretch')

    st.subheader("2.2 주요 변수 기술통계")
    disp_cols = [c for c in ["amt","cnt","log_amt","log_cnt"] if c in df.columns]
    st.dataframe(df[disp_cols].describe().T, width='stretch')

    st.subheader("2.3 주요 변수별 데이터 분포 (Histogram)")
    with st.spinner("히스토그램 생성 중..."):
        hist_fig = plot_histograms(df)
    st.pyplot(hist_fig)
    st.download_button("Histogram 이미지 다운로드", data=fig_to_bytes(hist_fig),
                       file_name="histogram_amt_log_amt.png", mime="image/png")

# =====================================================
# 3. Heatmap 분석
# =====================================================
with tab_hm:
    st.subheader("3.1 수치형 변수 상관관계 Heatmap")
    with st.spinner("상관관계 Heatmap 생성 중..."):
        corr_fig, corr_df = plot_corr_heatmap(df)
    st.pyplot(corr_fig)
    st.download_button("상관관계 Heatmap 다운로드", data=fig_to_bytes(corr_fig),
                       file_name="correlation_heatmap.png", mime="image/png")
    st.dataframe(corr_df.round(3), width='stretch')

    st.subheader("3.2 연령대 × 시간대 소비 Heatmap")
    with st.spinner("연령대 × 시간대 Heatmap 생성 중..."):
        ah_fig, ah_tbl = plot_age_hour_heatmap(df)
    st.pyplot(ah_fig)
    st.download_button("연령대_시간대 Heatmap 다운로드", data=fig_to_bytes(ah_fig),
                       file_name="age_hour_sales_heatmap.png", mime="image/png")
    st.dataframe(ah_tbl.round(2), width='stretch')

    st.subheader("3.3 요일 × 시간대 소비 Heatmap")
    with st.spinner("요일 × 시간대 Heatmap 생성 중..."):
        dh_fig, dh_tbl = plot_day_hour_heatmap(df)
    st.pyplot(dh_fig)
    st.download_button("요일_시간대 Heatmap 다운로드", data=fig_to_bytes(dh_fig),
                       file_name="day_hour_sales_heatmap.png", mime="image/png")
    st.dataframe(dh_tbl.round(2), width='stretch')

    st.subheader("3.4 업종 TOP 10 × 시간대 소비 Heatmap")
    with st.spinner("업종 × 시간대 Heatmap 생성 중..."):
        bh_fig, bh_tbl = plot_biz_hour_heatmap(df, top_n=10)
    st.pyplot(bh_fig)
    st.download_button("업종_시간대 Heatmap 다운로드", data=fig_to_bytes(bh_fig),
                       file_name="biz_hour_sales_heatmap.png", mime="image/png")
    st.dataframe(bh_tbl.round(2), width='stretch')

# =====================================================
# 4. 모델 학습
# =====================================================
with tab_model:
    st.subheader("4.1 인코딩 및 모델 학습")
    st.write("""
    본 프로젝트는 7개 입력 변수를 기준으로 매출금액을 예측한다.
    순서 의미가 있는 `age`, `day`, `hour`는 Label Encoding을 적용하고,
    순서 의미가 없는 `sex`, `admi_cty_no`, `card_tpbuz_nm_1`, `card_tpbuz_nm_2`는 One-Hot Encoding을 적용한다.
    """)

    if not model_files_exist():
        st.warning("저장된 모델 또는 인코더가 없습니다. 아래 버튼을 눌러 먼저 학습해주세요.")

    if st.button("모델 학습 및 저장"):
        with st.spinner("모델과 인코더를 학습 중입니다..."):
            model, model_info, X_test, y_test_real, pred_real = train_and_save_model(
                df, sample_size, use_log_target, model_name)
        st.success("학습 완료. model/ 폴더와 encoders/ 폴더에 저장되었습니다.")

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{model_info['metrics']['RMSE']:,.0f}")
        c2.metric("MAE",  f"{model_info['metrics']['MAE']:,.0f}")
        c3.metric("R2",   f"{model_info['metrics']['R2']:.4f}")

        st.subheader("예측값 vs 실제값")
        ap_fig = plot_actual_pred(y_test_real, pred_real)
        st.pyplot(ap_fig)
        st.download_button("예측값_실제값 그래프 다운로드", data=fig_to_bytes(ap_fig),
                           file_name="actual_vs_pred.png", mime="image/png")

    if model_files_exist():
        _, model_info = load_saved_model()
        st.subheader("저장된 모델 정보")
        st.json(model_info)
        metrics = model_info.get("metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{metrics.get('RMSE', 0):,.0f}")
        c2.metric("MAE",  f"{metrics.get('MAE',  0):,.0f}")
        c3.metric("R2",   f"{metrics.get('R2',   0):.4f}")

        st.subheader("저장 파일")
        st.code(f"""
{SALES_MODEL_PATH}
{MODEL_INFO_PATH}
{LABEL_ENCODER_PATH}
{ONEHOT_ENCODER_PATH}
{FEATURE_COLUMNS_PATH}
        """)

# =====================================================
# 5. 매출 예측
# =====================================================
with tab_pred:
    st.subheader("5. 매출 예측 화면")

    if not model_files_exist():
        st.info("저장된 모델/인코더가 아직 없습니다. 예측 버튼을 누르면 자동으로 학습 후 저장합니다.")

    p1, p2 = st.columns(2)

    with p1:
        sel_sex_label  = st.selectbox("성별(sex)",    ["남성", "여성"])
        sel_sex        = SEX_REVERSE_MAP[sel_sex_label]

        sel_age_label  = st.selectbox("연령(age)",    list(AGE_MAP.values()), index=3)
        sel_age        = {v: k for k, v in AGE_MAP.items()}[sel_age_label]

        sel_day_label  = st.selectbox("요일(day)",    list(DAY_MAP.values()), index=4)
        sel_day        = {v: k for k, v in DAY_MAP.items()}[sel_day_label]

        sel_hour_label = st.selectbox("시간대(hour)", list(HOUR_MAP.values()), index=4)
        sel_hour       = {v: k for k, v in HOUR_MAP.items()}[sel_hour_label]

    with p2:
        st.caption(f"행정동 선택 가능 개수: {len(admin_name_options)}개"
                   + ("" if admin_ok else "  ⚠️ hwaseong_admin_code.csv 없음 — 코드값으로 표시"))
        sel_admi_name = st.selectbox("행정동(admi_cty_no)", options=admin_name_options)
        sel_admi      = admin_name_to_code[sel_admi_name]

        if admin_ok:
            with st.expander("행정동 전체 목록 확인"):
                st.dataframe(admin_df, width='stretch')

        sel_biz1 = st.selectbox("업종 대분류(card_tpbuz_nm_1)",
                                sorted(df["card_tpbuz_nm_1"].dropna().unique()))
        biz2_opts = sorted(df[df["card_tpbuz_nm_1"] == sel_biz1]["card_tpbuz_nm_2"]
                           .dropna().unique())
        if not biz2_opts:
            st.warning("선택한 대분류에 해당하는 중분류가 없습니다.")
            sel_biz2 = None
        else:
            sel_biz2 = st.selectbox("업종 중분류(card_tpbuz_nm_2)", biz2_opts)

        if sel_biz2 is not None:
            input_df = pd.DataFrame([{
                "sex": sel_sex, "age": sel_age, "day": sel_day, "hour": sel_hour,
                "admi_cty_no": sel_admi,
                "card_tpbuz_nm_1": sel_biz1, "card_tpbuz_nm_2": sel_biz2,
            }])

            st.subheader("입력값 확인")
            st.dataframe(pd.DataFrame([{
                "성별": sel_sex_label, "연령": sel_age_label, "요일": sel_day_label,
                "시간대": sel_hour_label, "행정동": sel_admi_name,
                "업종 대분류": sel_biz1, "업종 중분류": sel_biz2,
            }]), width='stretch')
        else:
            input_df = None

    if st.button("예상 매출액 예측하기") and input_df is not None:
        if not model_files_exist():
            with st.spinner("저장된 모델이 없어 자동으로 학습 및 저장 중입니다..."):
                train_and_save_model(df, sample_size, use_log_target, model_name)
            st.info("모델과 인코더가 model/, encoders/ 폴더에 자동 저장되었습니다.")

        try:
            model, model_info = load_saved_model()
            encoded = transform_with_saved_encoders(input_df)

            with st.expander("디버그: 인코딩된 입력값 확인"):
                st.write("원본 입력값:", input_df.to_dict(orient="records")[0])
                raw_pred = model.predict(encoded)[0]
                st.write(f"모델 raw 출력 (log 공간): {raw_pred:.6f}")
                st.write(f"use_log_target: {model_info.get('use_log_target')}")
                st.dataframe(encoded, width='stretch')

            pred = np.expm1(raw_pred) if model_info.get("use_log_target", True) else raw_pred
            pred = max(pred, 0)
            st.success(f"예상 매출액: {pred:,.0f}원")
        except Exception as e:
            st.error(f"예측 오류: {e}")

        st.subheader("저장 파일 확인")
        paths = [SALES_MODEL_PATH, MODEL_INFO_PATH,
                 LABEL_ENCODER_PATH, ONEHOT_ENCODER_PATH, FEATURE_COLUMNS_PATH]
        st.dataframe(pd.DataFrame({
            "파일": paths,
            "존재 여부": [os.path.exists(p) for p in paths],
        }), width='stretch')

# =====================================================
# 6. 보고서 문구
# =====================================================
with tab_report:
    st.subheader("보고서에 붙여넣을 수 있는 문구")
    st.markdown("""
### 인코딩 및 모델 학습

○ 본 프로젝트는 성별, 연령, 요일, 시간대, 행정동, 업종 정보를 입력받아 매출금액을 예측하는 회귀 모델을 구축하였다.

○ 순서에 의미가 있는 연령(`age`), 요일(`day`), 시간대(`hour`) 변수는 Label Encoding을 적용하였다.

○ 순서에 의미가 없는 성별(`sex`), 행정동(`admi_cty_no`), 업종 대분류(`card_tpbuz_nm_1`), 업종 중분류(`card_tpbuz_nm_2`) 변수는 One-Hot Encoding을 적용하였다.

○ 학습된 모델은 `model` 폴더에 저장하고, 학습에 사용된 인코더는 `encoders` 폴더에 저장하여 예측 시 동일한 전처리 기준을 재사용할 수 있도록 구현하였다.

### 프로토타이핑

○ Streamlit 기반 웹 화면을 구성하여 사용자가 성별, 연령, 요일, 시간대, 행정동, 업종을 선택하면 예상 매출금액을 확인할 수 있도록 구현하였다.

○ 저장된 모델이 없는 경우 예측 버튼 클릭 시 자동으로 모델을 학습하고 저장한 뒤 예측 결과를 출력하도록 구현하였다.
    """)
