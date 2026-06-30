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

FLOWPOP_ZIP_PATH = os.path.join(
    os.path.expanduser("~"), "Downloads",
    "유동인구_행정동 단위 집계_202601-202603.zip"
)

SEMAS_DIR      = DATASET_DIR
SEMAS_ZIP_NAME = "semas_store_info_202603.zip"
SEMAS_ZIP_PATH = os.path.join(DATASET_DIR, SEMAS_ZIP_NAME)
SEMAS_GDRIVE_FILE_ID = "1Gp573SzYdObGWVi4r6hSJFX0oFumq8qr"

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

def ensure_semas_data():
    """SEMAS zip이 없으면 Google Drive에서 다운로드 후 압축 해제."""
    # CSV가 이미 하나라도 있으면 스킵
    if glob.glob(os.path.join(SEMAS_DIR, "semas_store_info_*.csv")):
        return
    # zip이 있으면 바로 해제
    if os.path.exists(SEMAS_ZIP_PATH):
        _extract_semas_zip()
        return
    # zip도 없고 Drive ID도 없으면 패스 (탭에서 경고 표시)
    if not SEMAS_GDRIVE_FILE_ID:
        return
    import gdown
    os.makedirs(DATASET_DIR, exist_ok=True)
    url = f"https://drive.google.com/uc?id={SEMAS_GDRIVE_FILE_ID}"
    with st.spinner("상가 데이터를 Google Drive에서 다운로드 중입니다... (최초 1회, 약 240MB)"):
        gdown.download(url, SEMAS_ZIP_PATH, quiet=False)
    _extract_semas_zip()


def _extract_semas_zip():
    import zipfile
    with st.spinner("상가 데이터 압축 해제 중..."):
        with zipfile.ZipFile(SEMAS_ZIP_PATH, "r") as zf:
            zf.extractall(SEMAS_DIR)


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
_SALES_DTYPES = {
    "cty_rgn_no": "int32", "admi_cty_no": "int32",
    "hour": "int8", "sex": "category", "age": "int8",
    "day": "int8", "amt": "int32", "cnt": "int16",
    "card_tpbuz_cd": "category",
    "card_tpbuz_nm_1": "category", "card_tpbuz_nm_2": "category",
}

def read_csv_auto(path_list):
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    for path in path_list:
        if os.path.exists(path):
            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc, dtype=_SALES_DTYPES)
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

@st.cache_data
def load_flowpop_data(zip_path: str):
    """유동인구 zip 파일에서 전체 CSV를 읽어 하나의 DataFrame으로 반환."""
    import zipfile, io
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                try:
                    chunk = pd.read_csv(io.BytesIO(f.read()), encoding="utf-8")
                    frames.append(chunk)
                except Exception:
                    pass
    if not frames:
        return pd.DataFrame()
    fp = pd.concat(frames, ignore_index=True)
    fp["ETL_YMD"] = pd.to_datetime(fp["ETL_YMD"].astype(str), format="%Y%m%d", errors="coerce")
    # 성별×연령대 컬럼을 long 형식으로도 미리 준비
    age_cols_m = [c for c in fp.columns if c.startswith("M_") and c.endswith("_CNT")]
    age_cols_f = [c for c in fp.columns if c.startswith("F_") and c.endswith("_CNT")]
    # 합계 컬럼
    fp["TOTAL_CNT"] = fp[age_cols_m + age_cols_f].sum(axis=1)
    fp["MALE_CNT"]  = fp[age_cols_m].sum(axis=1)
    fp["FEMALE_CNT"] = fp[age_cols_f].sum(axis=1)
    return fp


def _flowpop_long(fp: pd.DataFrame) -> pd.DataFrame:
    """유동인구를 성별×연령대 long 형식으로 변환."""
    age_map = {
        "10": "10대 미만", "15": "10대", "20": "20대", "25": "20대후반",
        "30": "30대", "35": "30대후반", "40": "40대", "45": "40대후반",
        "50": "50대", "55": "50대후반", "60": "60대", "65": "60대후반", "70": "70대 이상"
    }
    records = []
    for _, row in fp.iterrows():
        for prefix, sex_label in [("M", "남성"), ("F", "여성")]:
            for key, age_label in age_map.items():
                col = f"{prefix}_{key}_CNT"
                if col in fp.columns:
                    records.append({
                        "ADMI_CD": row["ADMI_CD"],
                        "CTY_NM": row.get("CTY_NM"),
                        "ADMI_NM": row.get("ADMI_NM"),
                        "ETL_YMD": row["ETL_YMD"],
                        "TIME_CD": row["TIME_CD"],
                        "FORN_GB": row["FORN_GB"],
                        "SEX": sex_label,
                        "AGE_GRP": age_label,
                        "CNT": row[col],
                    })
    return pd.DataFrame(records)


@st.cache_data
def load_semas_data(semas_dir: str, zip_path: str) -> pd.DataFrame:
    """SEMAS 상가 데이터 로드 — CSV 파일 또는 zip에서 읽음."""
    use_cols = [
        "상가업소번호", "상호명", "상권업종대분류명",
        "상권업종중분류명", "상권업종소분류명",
        "시도명", "시군구명", "행정동명", "행정동코드",
        "도로명주소", "경도", "위도",
    ]

    def _read_csv_bytes(raw_bytes):
        import io
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", usecols=use_cols)

    frames = []

    # 1) CSV 파일이 이미 풀려 있으면 그대로 사용
    csv_files = sorted(glob.glob(os.path.join(semas_dir, "semas_store_info_*.csv")))
    if csv_files:
        for fp in csv_files:
            try:
                frames.append(pd.read_csv(fp, encoding="utf-8", usecols=use_cols))
            except Exception:
                pass

    # 2) CSV 없으면 zip에서 직접 읽기
    elif os.path.exists(zip_path):
        import zipfile, io
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    try:
                        frames.append(_read_csv_bytes(zf.read(name)))
                    except Exception:
                        pass

    if not frames:
        return pd.DataFrame()

    semas = pd.concat(frames, ignore_index=True)
    semas = semas.dropna(subset=["경도", "위도"])
    semas["경도"] = pd.to_numeric(semas["경도"], errors="coerce")
    semas["위도"] = pd.to_numeric(semas["위도"], errors="coerce")
    return semas


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
    df["log_amt"]    = np.log1p(df["amt"].astype("float32")).astype("float32")
    df["log_cnt"]    = np.log1p(df["cnt"].astype("float32")).astype("float32")
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


@st.cache_resource
def load_saved_model():
    return joblib.load(SALES_MODEL_PATH), joblib.load(MODEL_INFO_PATH)

@st.cache_resource
def load_lstm_model():
    return joblib.load(LSTM_MODEL_PATH)

@st.cache_resource
def load_cluster_model():
    return joblib.load(CLUSTER_MODEL_PATH)

@st.cache_data
def load_store_counts():
    path = os.path.join(DATASET_DIR, "semas_store_count_mid.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"행정동코드": str})

# 카드 중분류 → SEMAS 중분류 매핑 (매핑 불가 항목은 포함하지 않음)
CARD_TO_SEMAS_MID = {
    # 음식
    "한식":             "한식",
    "중식":             "중식",
    "양식":             "서양식",
    "일식/수산물":      "일식",
    "커피/음료":        "비알코올 ",
    "부페":             "구내식당·뷔페",
    "간이주점":         "주점",
    "유흥주점":         "주점",
    # 소매/유통
    "음/식료품소매":    "식료품 소매",
    "의복/의류":        "섬유·의복·신발 소매",
    "종합소매점":       "종합 소매",
    "가전제품":         "가전·통신 소매",
    "화장품소매":       "의약·화장품 소매",
    "인테리어/가정용품":"기타 생활용품 소매",
    "차량관리/부품":    "자동차 부품 소매",
    # 생활서비스
    "미용서비스":       "이용·미용",
    "부동산":           "부동산 서비스",
    "세탁/가사서비스":  "세탁",
    "광고/인쇄/인화":   "광고",
    "차량관리/서비스":  "자동차 수리·세차",
    "연료판매":         "연료 소매",
    "사우나/휴게시설":  "욕탕·신체관리",
    "여행/유학대행":    "여행사·보조",
    "수리서비스":       "기타 가정용품 수리",
    # 의료/건강
    "일반병원":         "의원",
    "종합병원":         "병원",
    "특화병원":         "의원",
    "기타의료":         "기타 보건",
    "수의업":           "수의",
    "의약/의료품":      "의약·화장품 소매",
    # 학문/교육
    "기타교육":         "기타 교육",
    "유아교육":         "일반 교육",
    "예체능계학원":     "일반 교육",
    "외국어학원":       "일반 교육",
    "입시학원":         "일반 교육",
    "기술/직업교육학원":"일반 교육",
    "독서실/고시원":    "기타 교육",
    # 여가/오락
    "일반스포츠":       "스포츠 서비스",
    "취미/오락":        "유원지·오락",
    "요가/단전/마사지": "욕탕·신체관리",
    "숙박":             "일반 숙박",
}


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
def fmt(v):
    if v >= 1e8: return f"{v/1e8:,.1f}억원"
    if v >= 1e4: return f"{v/1e4:,.0f}만원"
    return f"{v:,.0f}원"


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
    top_biz = (df.groupby("card_tpbuz_nm_2", observed=True)["amt"]
               .sum().sort_values(ascending=False).head(top_n).index)
    filtered = df[df["card_tpbuz_nm_2"].isin(top_biz)].copy()
    filtered["card_tpbuz_nm_2"] = filtered["card_tpbuz_nm_2"].astype(str)
    filtered["hour_label"] = filtered["hour_label"].astype(str)
    pivot = pd.pivot_table(filtered,
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
tab_pred, tab_hm, tab_eda, tab_lstm, tab_cluster, tab_ai, tab_fp, tab_semas, tab_ov = st.tabs([
    "💰 매출 예측",
    "⏰ 시간대·요일 분석",
    "📊 소비 트렌드",
    "📈 시계열 예측",
    "👥 고객 군집 분석",
    "📝 AI 리포트",
    "🚶 유동인구 분석",
    "🏪 상권 분석",
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
            f"시간대당 예상 거래 건수  ※ {sel_biz2} 시간대 평균: {avg_cnt}건",
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

            # 실제 데이터에 존재하는 시간대·성별·연령 조합 + 각 조합의 출현 날 수 계산
            total_ref_days = ref["ta_ymd"].nunique() if "ta_ymd" in ref.columns and not ref.empty else 1
            combo_days = (
                ref.groupby(["hour", "sex", "age"])["ta_ymd"]
                .nunique()
                .reset_index()
                .rename(columns={"ta_ymd": "days"})
            ) if "ta_ymd" in ref.columns else (
                ref[["hour", "sex", "age"]].drop_duplicates().assign(days=1)
            )
            combo_days = combo_days.dropna()
            active_hours = int(ref["hour"].nunique()) if not ref.empty else 10

            # (시간대 × 성별 × 연령) 조합별 예측 — 캐시해서 세그먼트 표에 재사용
            pred_cache = {}
            for _, row in combo_days.iterrows():
                h, s, a = int(row["hour"]), row["sex"], int(row["age"])
                days = int(row["days"])
                input_row = pd.DataFrame([{
                    "sex": s, "age": a,
                    "day": sel_day, "hour": h,
                    "month": sel_month, "admi_cty_no": sel_admi,
                    "card_tpbuz_nm_1": sel_biz1, "card_tpbuz_nm_2": sel_biz2,
                    "cnt": sel_cnt,
                }])
                try:
                    encoded  = transform_with_saved_encoders(input_row)
                    raw_pred = model.predict(encoded)[0]
                    p = max(np.expm1(raw_pred) if model_info.get("use_log_target", True) else raw_pred, 0)
                except Exception:
                    p = 0.0
                # 출현 빈도(days/total_ref_days)를 가중치로 적용
                weight = days / max(total_ref_days, 1)
                pred_cache[(h, s, a)] = (p, weight)

            # 하루 예상 매출 = Σ(예측값 × 출현 빈도)
            # → 해당 조합이 자주 나타날수록 더 많이 반영
            weighted_preds = [p * w for p, w in pred_cache.values()]
            preds          = [p for p, _ in pred_cache.values()]
            daily_pred     = sum(weighted_preds)
            avg_per_grp    = float(np.mean(preds)) if preds else 0.0

            # ── 업체당 매출 계산 (SEMAS 중분류 매핑 가능 업종만) ──
            store_cnt_df = load_store_counts()
            semas_mid = CARD_TO_SEMAS_MID.get(sel_biz2)
            per_store_pred = None
            n_stores = 0
            if store_cnt_df is not None and semas_mid and sel_admi:
                sc = store_cnt_df[
                    (store_cnt_df["행정동코드"] == str(sel_admi)) &
                    (store_cnt_df["상권업종중분류명"] == semas_mid)
                ]
                n_stores = int(sc["store_count"].sum())
                if n_stores > 0:
                    per_store_pred = daily_pred / n_stores

            # 결과 표시
            st.success(f"### {sel_admi_name} {sel_biz2} 동네 전체 예상 일매출: **{fmt(daily_pred)}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("조합당 평균 예측 매출", fmt(avg_per_grp))
            c2.metric("분석에 사용된 조합 수", f"{len(preds)}개 (시간대×성별×연령)")
            if per_store_pred is not None:
                c3.metric("업체당 예상 일매출",
                          fmt(per_store_pred),
                          help=f"SEMAS 기준 {sel_admi_name} '{semas_mid}' 업체 수: {n_stores:,}개")
            elif semas_mid is None:
                c3.metric("업체당 예상 일매출", "데이터 없음",
                          help=f"'{sel_biz2}'은 소상공인 상가정보와 업종 분류가 달라 업체 수 추정 불가")
            else:
                c3.metric("업체당 예상 일매출", "-", help="동네를 선택하면 업체당 매출이 표시됩니다")

            with st.expander("계산 과정 보기"):
                st.text(f"- 예측에 사용된 조합 수: {len(preds)}개 (시간대 × 성별 × 연령)")
                st.text(f"- 시간대당 거래건수(입력값): {sel_cnt}건")
                st.text(f"- 하루 활성 시간대: {active_hours}개 (실제 데이터 기준)")
                st.text(f"- 기준 날짜 수: {total_ref_days}일")
                st.text(f"- 하루 예상 매출 = Σ(조합 예측값 × 출현 빈도) = {fmt(daily_pred)}")
                st.text(f"  (자주 나타나는 조합일수록 더 많이 반영, 희귀 조합은 적게 반영)")

            st.caption(f"※ {sel_district} {sel_admi_name} · {sel_biz2} · {sel_month}월 {sel_day_label} 기준 예측")

            # ── 고객 세그먼트별 상세 분석 표 ──
            st.divider()
            st.subheader("고객 유형별 매출 분석")

            seg_rows = []
            total_ref_cnt = len(ref)
            for _, combo in combo_days.iterrows():
                h, s, a = int(combo["hour"]), combo["sex"], int(combo["age"])
                days = int(combo["days"])
                weight = days / max(total_ref_days, 1)
                seg_mask  = (ref["hour"] == h) & (ref["sex"] == s) & (ref["age"] == a)
                seg_cnt   = seg_mask.sum()
                seg_ratio = seg_cnt / total_ref_cnt * 100 if total_ref_cnt > 0 else 0
                p, _ = pred_cache.get((h, s, a), (0.0, 0.0))

                seg_rows.append({
                    "시간대":            HOUR_MAP.get(h, str(h)),
                    "성별":              "여성" if s == "F" else "남성",
                    "연령대":            AGE_MAP.get(a, f"{a}"),
                    "조합 예측 매출(원)": int(p),
                    "출현 빈도":         f"{weight*100:.0f}% ({days}/{total_ref_days}일)",
                    "가중 예측 매출(원)": int(p * weight),
                    "데이터 비중(%)":    round(seg_ratio, 1),
                })

            if seg_rows:
                seg_df = (
                    pd.DataFrame(seg_rows)
                    .sort_values("데이터 비중(%)", ascending=False)
                    .reset_index(drop=True)
                )

                # 요약 피벗: 연령대 × 성별 가중 예측 매출 합계
                st.markdown("**연령대 × 성별 가중 예측 매출 합계 (원)**")
                pivot = (
                    seg_df.groupby(["연령대", "성별"])["가중 예측 매출(원)"]
                    .sum()
                    .round(0)
                    .astype(int)
                    .unstack("성별")
                    .reindex([v for v in AGE_MAP.values() if v in seg_df["연령대"].values])
                )
                st.dataframe(
                    pivot.style.format("{:,}").background_gradient(cmap="Blues", axis=None),
                    width="stretch"
                )

                # 전체 세그먼트 상세 표
                st.markdown("**시간대 × 성별 × 연령대 상세**")
                st.dataframe(
                    seg_df.style.format({
                        "조합 예측 매출(원)": "{:,}",
                        "가중 예측 매출(원)": "{:,}",
                        "데이터 비중(%)": "{:.1f}%",
                    }).background_gradient(subset=["가중 예측 매출(원)"], cmap="Greens"),
                    width="stretch",
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
        lt_codes = {admin_name_to_code[d] for d in admin_district_to_dongs.get(lt_district, []) if d in admin_name_to_code}
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
        all_daily = lt_df.groupby("_date")["amt"].sum().reset_index().sort_values("_date")
        all_daily = all_daily.rename(columns={"_date": "date"})

        # ── 시작일자 설정 ──
        min_date = all_daily["date"].min().date()
        max_date = all_daily["date"].max().date()
        sc1, sc2 = st.columns(2)
        with sc1:
            start_date = st.date_input(
                "시작일자",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="lt_start_date"
            )
        with sc2:
            FORECAST_DAYS = st.slider("미래 예측 기간 (일)", 7, 60, 30)

        daily = all_daily[all_daily["date"] >= pd.Timestamp(start_date)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"].astype(str), y=daily["amt"],
            name="실제 매출", line=dict(color="#58a6ff")
        ))

        # LSTM 미래 예측 (모델 있을 때만)
        if os.path.exists(LSTM_MODEL_PATH):
            try:
                import torch
                lstm_data  = load_lstm_model()
                model_lstm = lstm_data["model"]
                seq_len    = lstm_data["seq_len"]

                vals = daily["amt"].values.astype("float32").reshape(-1, 1)
                if len(vals) < seq_len:
                    st.caption(f"ℹ️ 데이터가 {len(vals)}일치로 예측에 필요한 {seq_len}일보다 부족해 LSTM 예측을 생략합니다.")
                else:
                    # 필터된 데이터 스케일로 자체 정규화 (전체 스케일러 대신)
                    from sklearn.preprocessing import MinMaxScaler
                    local_scaler = MinMaxScaler()
                    scaled   = local_scaler.fit_transform(vals)
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
                    pred_vals    = local_scaler.inverse_transform([[p] for p in preds])[:,0]
                    pred_vals    = np.clip(pred_vals, 0, None)
                    last_date    = daily["date"].iloc[-1]
                    future_dates = [str((last_date + pd.Timedelta(days=i+1)).date()) for i in range(FORECAST_DAYS)]

                    fig.add_trace(go.Scatter(
                        x=future_dates, y=pred_vals,
                        name="LSTM 예측", line=dict(color="#3fb950", dash="dash")
                    ))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("예측 평균 일매출", fmt(pred_vals.mean()))
                    c2.metric("예측 최고 매출일", fmt(pred_vals.max()))
                    c3.metric("예측 최저 매출일", fmt(pred_vals.min()))
            except Exception as e:
                st.warning(f"LSTM 예측 오류: {e}")

        # 단위 자동 선택
        max_amt = daily["amt"].max()
        if max_amt >= 1e8:
            unit_div, unit_label = 1e8, "억원"
        elif max_amt >= 1e4:
            unit_div, unit_label = 1e4, "만원"
        else:
            unit_div, unit_label = 1, "원"

        # 차트 y값 단위 변환
        for trace in fig.data:
            trace.y = [v / unit_div for v in trace.y]
            trace.hovertemplate = f"%{{y:,.1f}}{unit_label}<extra>%{{fullData.name}}</extra>"

        fig.update_layout(
            title=f"일별 매출 추이 ({lt_district} {lt_admi_name} · {lt_biz2})",
            xaxis_title="날짜",
            yaxis_title=f"매출액 ({unit_label})",
            yaxis=dict(tickformat=",.1f"),
            hovermode="x unified",
            height=440
        )
        st.plotly_chart(fig, width="stretch")

        avg_amt = daily["amt"].mean()
        tot_amt = daily["amt"].sum()

        d1, d2 = st.columns(2)
        d1.metric("기간 평균 일매출", fmt(avg_amt))
        d2.metric("기간 총 매출",     fmt(tot_amt))

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
        cl_codes = {admin_name_to_code[d] for d in admin_district_to_dongs.get(cl_district, []) if d in admin_name_to_code}
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
            st.plotly_chart(fig_age, width="stretch")

        # 성별 매출 비중
        with ch2:
            sex_grp = cl_df.groupby("sex")["amt"].sum().reset_index()
            sex_grp["성별"] = sex_grp["sex"].map(SEX_MAP)
            fig_sex = px.pie(sex_grp, names="성별", values="amt", title="성별 매출 비중",
                             color_discrete_map={"여성": "#f472b6", "남성": "#60a5fa"})
            fig_sex.update_layout(height=320)
            st.plotly_chart(fig_sex, width="stretch")

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
            st.plotly_chart(fig_hour, width="stretch")

        # 요일별 매출
        with ch4:
            day_grp = cl_df.groupby("day")["amt"].sum().reset_index()
            day_grp["요일"] = day_grp["day"].map(DAY_MAP)
            fig_day = px.bar(day_grp, x="요일", y="amt", title="요일별 매출",
                             labels={"amt": "매출액 (원)"}, color="amt",
                             color_continuous_scale="Oranges")
            fig_day.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_day, width="stretch")

        # 업종 TOP10
        st.subheader("업종별 매출 TOP 10")
        top_biz = (cl_df.groupby("card_tpbuz_nm_2")["amt"].sum()
                   .nlargest(10).reset_index()
                   .rename(columns={"card_tpbuz_nm_2": "업종", "amt": "매출액 (원)"}))
        fig_biz = px.bar(top_biz, x="매출액 (원)", y="업종", orientation="h",
                         color="매출액 (원)", color_continuous_scale="Purples")
        fig_biz.update_layout(coloraxis_showscale=False, height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_biz, width="stretch")

        # Autoencoder 군집 모델 요약 (있을 때만)
        if os.path.exists(CLUSTER_MODEL_PATH):
            st.divider()
            st.subheader("전체 고객 유형 분류 (Autoencoder 모델)")
            try:
                cluster_data  = load_cluster_model()
                cluster_stats = cluster_data["stats"]
                cluster_names = cluster_data.get("names", [])
                cluster_centers = cluster_data["centers"]
                ratios = cluster_data.get("ratios", [])
                st.dataframe(cluster_stats, width="stretch", hide_index=True)
                viz_df = pd.DataFrame(cluster_centers, columns=["x", "y"])
                viz_df["유형"] = cluster_names
                viz_df["비율(%)"] = ratios
                fig_c = px.scatter(viz_df, x="x", y="y", size="비율(%)", color="유형",
                                   text="유형", size_max=60, height=380,
                                   title="고객 유형 분포 (잠재 공간)")
                fig_c.update_traces(textposition="top center")
                st.plotly_chart(fig_c, width="stretch")
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

    # 버튼 활성화 조건 검사
    _ai_missing = []
    if not api_key:
        _ai_missing.append("OpenAI API Key")
    if admin_ok and ai_district == "전체":
        _ai_missing.append("지역 (시/구)")
    if admin_ok and ai_admi_name == "전체":
        _ai_missing.append("동네 선택")
    if ai_biz1 == "전체":
        _ai_missing.append("업종 대분류")
    if ai_biz2 == "전체":
        _ai_missing.append("업종 중분류")

    if _ai_missing:
        st.info(f"다음 항목을 모두 입력해야 리포트를 생성할 수 있습니다: **{', '.join(_ai_missing)}**")

    if st.button("📝 AI 리포트 생성", type="primary", disabled=bool(_ai_missing)):
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
                        ai_codes = {admin_name_to_code[d] for d in admin_district_to_dongs.get(ai_district, []) if d in admin_name_to_code}
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

# =====================================================
# 🚶 유동인구 분석
# =====================================================
with tab_fp:
    st.subheader("유동인구 분석")
    st.caption("행정동 단위 시간대별 유동인구 데이터를 기반으로 상권 방문 패턴을 분석합니다.")

    if not os.path.exists(FLOWPOP_ZIP_PATH):
        st.warning(
            f"유동인구 데이터 파일을 찾을 수 없습니다.\n\n"
            f"**{FLOWPOP_ZIP_PATH}** 경로에 zip 파일을 배치해 주세요."
        )
    else:
        fp_all = load_flowpop_data(FLOWPOP_ZIP_PATH)

        if fp_all.empty:
            st.error("유동인구 데이터를 불러오지 못했습니다.")
        else:
            # ── 공통 필터 ──────────────────────────────────────
            fp_areas = sorted(fp_all["CTY_NM"].dropna().unique())
            fp_dongs = sorted(fp_all["ADMI_NM"].dropna().unique())

            fc1, fc2 = st.columns(2)
            with fc1:
                fp_sel_city = st.selectbox("시/구 선택", ["전체"] + fp_areas, key="fp_city")
            with fc2:
                if fp_sel_city != "전체":
                    dong_pool = sorted(fp_all[fp_all["CTY_NM"] == fp_sel_city]["ADMI_NM"].dropna().unique())
                else:
                    dong_pool = fp_dongs
                fp_sel_dong = st.selectbox("행정동 선택", ["전체"] + dong_pool, key="fp_dong")

            fp_forn = st.radio(
                "내/외국인 구분", ["전체", "내국인만", "외국인만"],
                horizontal=True, key="fp_forn"
            )

            # 필터 적용
            fp = fp_all.copy()
            if fp_sel_city != "전체":
                fp = fp[fp["CTY_NM"] == fp_sel_city]
            if fp_sel_dong != "전체":
                fp = fp[fp["ADMI_NM"] == fp_sel_dong]
            if fp_forn == "내국인만":
                fp = fp[fp["FORN_GB"] == "N"]
            elif fp_forn == "외국인만":
                fp = fp[fp["FORN_GB"] == "F"]

            st.divider()

            # ══════════════════════════════════════════════════
            # [기능 2] 시간대별 유동인구 히트맵
            # ══════════════════════════════════════════════════
            st.markdown("#### ⏰ 시간대 × 요일 유동인구 히트맵")
            st.caption("어느 요일·시간대에 유동인구가 집중되는지 보여줍니다. 매출 피크와 비교해 보세요.")

            fp_dow = fp.copy()
            fp_dow["DOW"] = fp_dow["ETL_YMD"].dt.dayofweek  # 0=월
            dow_label = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
            fp_dow["DOW_LABEL"] = fp_dow["DOW"].map(dow_label)

            hm_data = (
                fp_dow.groupby(["DOW_LABEL", "TIME_CD"])["TOTAL_CNT"]
                .mean()
                .reset_index()
            )
            dow_order = ["월","화","수","목","금","토","일"]
            hm_pivot = hm_data.pivot(index="DOW_LABEL", columns="TIME_CD", values="TOTAL_CNT").reindex(dow_order)

            fig_hm = go.Figure(go.Heatmap(
                z=hm_pivot.values,
                x=[f"{h}시" for h in hm_pivot.columns],
                y=hm_pivot.index,
                colorscale="Blues",
                hovertemplate="요일: %{y}<br>시간: %{x}<br>평균 유동인구: %{z:,.0f}명<extra></extra>",
            ))
            fig_hm.update_layout(
                xaxis_title="시간대",
                yaxis_title="요일",
                height=340,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            st.divider()

            # ══════════════════════════════════════════════════
            # [기능 3] 타겟 고객층 분석 (성별×연령대)
            # ══════════════════════════════════════════════════
            st.markdown("#### 👥 성별 × 연령대 유동인구 분포")
            st.caption("이 지역을 오가는 주요 고객층의 성별·연령대 비율을 확인합니다.")

            age_cols_m = [c for c in fp.columns if c.startswith("M_") and c.endswith("_CNT")]
            age_cols_f = [c for c in fp.columns if c.startswith("F_") and c.endswith("_CNT")]

            def _age_label(col):
                age_num = int(col.split("_")[1])
                return f"{age_num}대" if age_num >= 20 else ("10대" if age_num == 15 else "10대 미만")

            male_sums   = {_age_label(c): fp[c].sum() for c in age_cols_m}
            female_sums = {_age_label(c): fp[c].sum() for c in age_cols_f}

            # 연령대 통합 (같은 라벨 합산)
            from collections import defaultdict
            male_agg, female_agg = defaultdict(float), defaultdict(float)
            for c in age_cols_m:
                male_agg[_age_label(c)] += fp[c].sum()
            for c in age_cols_f:
                female_agg[_age_label(c)] += fp[c].sum()

            age_order = ["10대 미만","10대","20대","30대","40대","50대","60대","70대 이상"]
            male_vals   = [male_agg.get(a, 0)   for a in age_order]
            female_vals = [female_agg.get(a, 0) for a in age_order]

            fig_age = go.Figure()
            fig_age.add_bar(name="남성", x=age_order, y=male_vals,
                            marker_color="#58a6ff",
                            hovertemplate="%{x}: %{y:,.0f}명<extra>남성</extra>")
            fig_age.add_bar(name="여성", x=age_order, y=female_vals,
                            marker_color="#f78166",
                            hovertemplate="%{x}: %{y:,.0f}명<extra>여성</extra>")
            fig_age.update_layout(
                barmode="group",
                xaxis_title="연령대",
                yaxis_title="유동인구 합계 (명)",
                height=340,
                margin=dict(t=20, b=40),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_age, use_container_width=True)

            total_m = sum(male_agg.values())
            total_f = sum(female_agg.values())
            total_all = total_m + total_f
            if total_all > 0:
                top_age_m = max(male_agg, key=male_agg.get)
                top_age_f = max(female_agg, key=female_agg.get)
                a1, a2, a3 = st.columns(3)
                a1.metric("남성 비율", f"{total_m/total_all*100:.1f}%")
                a2.metric("여성 비율", f"{total_f/total_all*100:.1f}%")
                a3.metric("최다 연령대", f"남 {top_age_m} / 여 {top_age_f}")

            st.divider()

            # ══════════════════════════════════════════════════
            # [기능 4] 내국인 vs 외국인 비율
            # ══════════════════════════════════════════════════
            st.markdown("#### 🌏 내국인 vs 외국인 유동인구 비율")
            st.caption("외국인 유동인구 비율이 높은 지역은 관광·외국인 특화 상권일 가능성이 높습니다.")

            forn_agg = (
                fp_all[
                    (fp_all["CTY_NM"] == fp_sel_city if fp_sel_city != "전체" else pd.Series(True, index=fp_all.index)) &
                    (fp_all["ADMI_NM"] == fp_sel_dong if fp_sel_dong != "전체" else pd.Series(True, index=fp_all.index))
                ]
                .groupby("FORN_GB")["TOTAL_CNT"].sum()
                .rename({"N": "내국인", "F": "외국인"})
            )

            if not forn_agg.empty and forn_agg.sum() > 0:
                fig_forn = go.Figure(go.Pie(
                    labels=forn_agg.index,
                    values=forn_agg.values,
                    marker_colors=["#58a6ff","#f78166"],
                    hole=0.45,
                    hovertemplate="%{label}: %{value:,.0f}명 (%{percent})<extra></extra>",
                ))
                fig_forn.update_layout(height=320, margin=dict(t=20, b=20))

                forn_col1, forn_col2 = st.columns([1, 1])
                with forn_col1:
                    st.plotly_chart(fig_forn, use_container_width=True)
                with forn_col2:
                    forn_total = forn_agg.sum()
                    for label, val in forn_agg.items():
                        st.metric(f"{label} 유동인구", f"{val:,.0f}명", f"{val/forn_total*100:.1f}%")

                # 동네별 외국인 비율 랭킹 (전체 선택 시)
                if fp_sel_dong == "전체":
                    st.markdown("##### 외국인 비율 상위 행정동")
                    scope = fp_all.copy()
                    if fp_sel_city != "전체":
                        scope = scope[scope["CTY_NM"] == fp_sel_city]
                    forn_by_dong = scope.groupby(["ADMI_NM","FORN_GB"])["TOTAL_CNT"].sum().unstack(fill_value=0)
                    if "F" in forn_by_dong.columns and "N" in forn_by_dong.columns:
                        forn_by_dong["외국인비율(%)"] = (
                            forn_by_dong["F"] / (forn_by_dong["F"] + forn_by_dong["N"]) * 100
                        ).round(2)
                        top_forn = forn_by_dong.sort_values("외국인비율(%)", ascending=False).head(10).reset_index()
                        top_forn.columns = ["행정동", "외국인", "내국인", "외국인비율(%)"]
                        st.dataframe(top_forn[["행정동","내국인","외국인","외국인비율(%)"]], use_container_width=True)

            st.divider()

            # ══════════════════════════════════════════════════
            # [기능 1] 유동인구 × 매출 상관 분석
            # ══════════════════════════════════════════════════
            st.markdown("#### 📈 유동인구 × 매출 상관 분석")
            st.caption("같은 날짜·행정동 기준으로 유동인구와 매출을 연결해 상관관계를 분석합니다.")

            # 매출 일별 집계 (admi_cty_no 기준)
            if "ta_ymd" in df.columns:
                sales_daily = (
                    df.groupby(["ta_ymd", "admi_cty_no"])["amt"]
                    .sum()
                    .reset_index()
                    .rename(columns={"ta_ymd":"date","admi_cty_no":"ADMI_CD","amt":"SALES"})
                )
                sales_daily["date"] = pd.to_datetime(sales_daily["date"])
                # 유동인구 일별 집계 (ADMI_CD 기준, 내+외국인 합산)
                fp_daily = (
                    fp_all.groupby(["ETL_YMD","ADMI_CD"])["TOTAL_CNT"]
                    .sum()
                    .reset_index()
                    .rename(columns={"ETL_YMD":"date","TOTAL_CNT":"FLOWPOP"})
                )
                fp_daily["ADMI_CD"] = fp_daily["ADMI_CD"].astype(int)

                merged = pd.merge(
                    sales_daily, fp_daily,
                    left_on=["date","ADMI_CD"], right_on=["date","ADMI_CD"],
                    how="inner"
                )

                if merged.empty:
                    st.info("매출 데이터와 날짜·행정동이 겹치는 데이터가 없습니다. (데이터 기간 또는 지역이 다를 수 있습니다)")
                else:
                    corr_val = merged["FLOWPOP"].corr(merged["SALES"])

                    # 업종 필터 추가
                    corr_biz1 = st.selectbox(
                        "업종 대분류 (상관분석용)", ["전체"] + sorted(df["card_tpbuz_nm_1"].dropna().unique()),
                        key="corr_biz1"
                    )
                    if corr_biz1 != "전체":
                        sales_biz = (
                            df[df["card_tpbuz_nm_1"] == corr_biz1]
                            .groupby(["ta_ymd","admi_cty_no"])["amt"]
                            .sum().reset_index()
                            .rename(columns={"ta_ymd":"date","admi_cty_no":"ADMI_CD","amt":"SALES"})
                        )
                        sales_biz["date"] = pd.to_datetime(sales_biz["date"])
                        merged = pd.merge(sales_biz, fp_daily, on=["date","ADMI_CD"], how="inner")
                        if not merged.empty:
                            corr_val = merged["FLOWPOP"].corr(merged["SALES"])

                    if not merged.empty:
                        fig_corr = px.scatter(
                            merged, x="FLOWPOP", y="SALES",
                            trendline="ols",
                            labels={"FLOWPOP":"유동인구 (명)","SALES":"일별 매출 (원)"},
                            opacity=0.6,
                            color_discrete_sequence=["#58a6ff"],
                        )
                        fig_corr.update_layout(height=380, margin=dict(t=20,b=40))
                        st.plotly_chart(fig_corr, use_container_width=True)

                        strength = (
                            "강한 양의 상관" if corr_val > 0.7 else
                            "중간 양의 상관" if corr_val > 0.4 else
                            "약한 양의 상관" if corr_val > 0.1 else
                            "거의 무상관" if corr_val > -0.1 else
                            "음의 상관"
                        )
                        cr1, cr2, cr3 = st.columns(3)
                        cr1.metric("피어슨 상관계수", f"{corr_val:.3f}")
                        cr2.metric("상관 강도", strength)
                        cr3.metric("분석 데이터 포인트", f"{len(merged):,}건")
            else:
                st.info("매출 데이터에 날짜 컬럼(ta_ymd)이 없어 상관 분석을 수행할 수 없습니다.")

# =====================================================
# 🏪 상권 분석 (SEMAS 소상공인 상가 데이터)
# =====================================================
with tab_semas:
    st.subheader("전국 상권 분석")
    st.caption("소상공인시장진흥공단 상가(상권)정보를 기반으로 업종 분포·경쟁 강도·입지 추천·주변 상권을 분석합니다.")

    ensure_semas_data()
    semas_df = load_semas_data(SEMAS_DIR, SEMAS_ZIP_PATH)

    if semas_df.empty:
        st.warning(
            "`dataset/semas_store_info_202603.zip` 또는 CSV 파일을 찾을 수 없습니다. "
            "Google Drive 파일 ID(`SEMAS_GDRIVE_FILE_ID`)를 코드에 입력하거나 "
            "zip 파일을 dataset 폴더에 직접 배치해 주세요."
        )
    else:
        # ── 공통 필터 ─────────────────────────────────────────
        sido_list = sorted(semas_df["시도명"].dropna().unique())
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sel_sido = st.selectbox("시/도 선택", ["전체"] + sido_list, key="sm_sido")
        with sc2:
            if sel_sido != "전체":
                gu_pool = sorted(semas_df[semas_df["시도명"] == sel_sido]["시군구명"].dropna().unique())
            else:
                gu_pool = sorted(semas_df["시군구명"].dropna().unique())
            sel_gu = st.selectbox("시/군/구 선택", ["전체"] + gu_pool, key="sm_gu")
        with sc3:
            if sel_gu != "전체":
                dong_pool = sorted(semas_df[semas_df["시군구명"] == sel_gu]["행정동명"].dropna().unique())
            elif sel_sido != "전체":
                dong_pool = sorted(semas_df[semas_df["시도명"] == sel_sido]["행정동명"].dropna().unique())
            else:
                dong_pool = []
            sel_dong = st.selectbox("행정동 선택", ["전체"] + dong_pool, key="sm_dong")

        # 필터 적용
        sm = semas_df.copy()
        if sel_sido != "전체":
            sm = sm[sm["시도명"] == sel_sido]
        if sel_gu != "전체":
            sm = sm[sm["시군구명"] == sel_gu]
        if sel_dong != "전체":
            sm = sm[sm["행정동명"] == sel_dong]

        sm_biz1_list = sorted(sm["상권업종대분류명"].dropna().unique())
        sel_biz1 = st.selectbox(
            "업종 대분류 (전체 기능에 적용)",
            ["전체"] + sm_biz1_list, key="sm_biz1"
        )
        if sel_biz1 != "전체":
            sm = sm[sm["상권업종대분류명"] == sel_biz1]

        st.caption(f"현재 조건 점포 수: **{len(sm):,}개**")
        st.divider()

        # ══════════════════════════════════════════════════════
        # [기능 1] 상권 밀집도 지도
        # ══════════════════════════════════════════════════════
        st.markdown("#### 🗺️ 상권 밀집도 지도")
        st.caption("점포 위치를 지도에 표시합니다. 색상은 업종 대분류를 나타냅니다.")

        map_sample = sm.dropna(subset=["경도","위도"])
        if len(map_sample) > 5000:
            map_sample = map_sample.sample(5000, random_state=42)

        if map_sample.empty:
            st.info("지도에 표시할 좌표 데이터가 없습니다.")
        else:
            center_lat = map_sample["위도"].mean()
            center_lon = map_sample["경도"].mean()

            # 업종별 색상
            biz_cats = map_sample["상권업종대분류명"].dropna().unique().tolist()
            palette = px.colors.qualitative.Safe
            color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(biz_cats)}

            fig_map = go.Figure()
            for cat in biz_cats:
                sub = map_sample[map_sample["상권업종대분류명"] == cat]
                fig_map.add_trace(go.Scattermapbox(
                    lat=sub["위도"], lon=sub["경도"],
                    mode="markers",
                    marker=dict(size=5, color=color_map[cat], opacity=0.7),
                    name=cat,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]} · %{customdata[2]}<extra></extra>"
                    ),
                    customdata=sub[["상호명","상권업종중분류명","행정동명"]].values,
                ))
            fig_map.update_layout(
                mapbox=dict(
                    style="carto-positron",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=11 if sel_dong != "전체" else (12 if sel_gu != "전체" else 9),
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=460,
                legend=dict(orientation="v", x=0, y=1, bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_map, use_container_width=True)
            if len(sm) > 5000:
                st.caption(f"⚠️ 지도에는 성능을 위해 무작위 5,000개만 표시됩니다. (전체 {len(sm):,}개)")

        st.divider()

        # ══════════════════════════════════════════════════════
        # [기능 2] 업종별 경쟁 강도 분석
        # ══════════════════════════════════════════════════════
        st.markdown("#### ⚔️ 업종별 경쟁 강도 분석")
        st.caption("행정동 단위로 같은 업종(중분류)이 몇 개나 밀집해 있는지 경쟁 강도를 보여줍니다.")

        comp_biz2_list = sorted(sm["상권업종중분류명"].dropna().unique())
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            comp_biz2 = st.selectbox("분석할 업종 (중분류)", comp_biz2_list, key="sm_comp_biz2")
        with comp_col2:
            comp_top_n = st.slider("상위 행정동 수", 5, 30, 15, key="sm_comp_n")

        comp_df = sm[sm["상권업종중분류명"] == comp_biz2].copy()
        if comp_df.empty:
            st.info("선택한 업종의 데이터가 없습니다.")
        else:
            comp_by_dong = (
                comp_df.groupby(["시도명","시군구명","행정동명"])
                .size().reset_index(name="점포수")
                .sort_values("점포수", ascending=False)
                .head(comp_top_n)
            )
            comp_by_dong["지역"] = comp_by_dong["시군구명"] + " " + comp_by_dong["행정동명"]

            fig_comp = go.Figure(go.Bar(
                x=comp_by_dong["점포수"],
                y=comp_by_dong["지역"],
                orientation="h",
                marker_color="#f78166",
                text=comp_by_dong["점포수"],
                textposition="outside",
                hovertemplate="%{y}: %{x}개<extra></extra>",
            ))
            fig_comp.update_layout(
                xaxis_title="점포 수",
                yaxis=dict(autorange="reversed"),
                height=max(300, comp_top_n * 26),
                margin=dict(t=20, b=40, r=80),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # 전국 동일 업종 총계 요약
            total_comp = sm[sm["상권업종중분류명"] == comp_biz2]
            avg_per_dong = total_comp.groupby("행정동명").size().mean()
            cx1, cx2, cx3 = st.columns(3)
            cx1.metric(f"'{comp_biz2}' 총 점포 수", f"{len(total_comp):,}개")
            cx2.metric("행정동 평균 점포 수", f"{avg_per_dong:.1f}개")
            cx3.metric("1위 동네 점포 수", f"{comp_by_dong['점포수'].iloc[0]:,}개")

        st.divider()

        # ══════════════════════════════════════════════════════
        # [기능 3] 신규 창업 입지 추천
        # ══════════════════════════════════════════════════════
        st.markdown("#### 🎯 신규 창업 입지 추천")
        st.caption(
            "행정동별 **점포 수 대비 유동인구** 비율이 높은 곳 = 수요 대비 공급이 부족한 유망 입지입니다. "
            "(유동인구 데이터가 로드된 경우에만 작동합니다)"
        )

        rec_biz2_list = sorted(semas_df["상권업종중분류명"].dropna().unique())
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            rec_biz2 = st.selectbox("창업 희망 업종 (중분류)", rec_biz2_list, key="sm_rec_biz2")
        with rec_col2:
            rec_top_n = st.slider("추천 지역 수", 5, 20, 10, key="sm_rec_n")

        # 업종별 행정동 점포 수
        rec_store_cnt = (
            semas_df[semas_df["상권업종중분류명"] == rec_biz2]
            .groupby("행정동명").size().reset_index(name="점포수")
        )

        try:
            fp_rec = load_flowpop_data(FLOWPOP_ZIP_PATH)
            fp_available = not fp_rec.empty
        except Exception:
            fp_available = False

        if fp_available:
            fp_dong_pop = (
                fp_rec[fp_rec["FORN_GB"] == "N"]
                .groupby("ADMI_NM")["TOTAL_CNT"].mean()
                .reset_index()
                .rename(columns={"ADMI_NM": "행정동명", "TOTAL_CNT": "평균유동인구"})
            )
            rec_merged = pd.merge(rec_store_cnt, fp_dong_pop, on="행정동명", how="inner")
            if rec_merged.empty:
                st.info("유동인구와 상가 데이터가 겹치는 행정동이 없습니다. (지역이 다를 수 있습니다)")
            else:
                rec_merged["유동인구/점포"] = (
                    rec_merged["평균유동인구"] / rec_merged["점포수"].clip(lower=1)
                ).round(1)
                rec_result = rec_merged.sort_values("유동인구/점포", ascending=False).head(rec_top_n)

                fig_rec = go.Figure(go.Bar(
                    x=rec_result["유동인구/점포"],
                    y=rec_result["행정동명"],
                    orientation="h",
                    marker=dict(
                        color=rec_result["유동인구/점포"],
                        colorscale="Greens", showscale=True,
                        colorbar=dict(title="유동인구/점포"),
                    ),
                    hovertemplate=(
                        "%{y}<br>유동인구/점포: %{x:,.1f}<br>"
                        "<extra></extra>"
                    ),
                    customdata=rec_result[["점포수","평균유동인구"]].values,
                ))
                fig_rec.update_layout(
                    xaxis_title="유동인구 / 점포 수 (높을수록 유망)",
                    yaxis=dict(autorange="reversed"),
                    height=max(300, rec_top_n * 30),
                    margin=dict(t=20, b=40, r=80),
                )
                st.plotly_chart(fig_rec, use_container_width=True)

                rec_result_show = rec_result.copy()
                rec_result_show.columns = ["행정동명","점포수","평균유동인구","유동인구/점포비율"]
                rec_result_show["평균유동인구"] = rec_result_show["평균유동인구"].round(0).astype(int)
                st.dataframe(rec_result_show.reset_index(drop=True), use_container_width=True)
        else:
            # 유동인구 없으면 점포 수 역순(경쟁 적은 곳)만 보여줌
            st.info("유동인구 데이터가 없어 점포 수가 적은 행정동(경쟁 약한 지역)으로 대체 추천합니다.")
            rec_scope = semas_df.copy()
            if sel_sido != "전체":
                rec_scope = rec_scope[rec_scope["시도명"] == sel_sido]
            if sel_gu != "전체":
                rec_scope = rec_scope[rec_scope["시군구명"] == sel_gu]
            rec_low = (
                rec_scope[rec_scope["상권업종중분류명"] == rec_biz2]
                .groupby(["시군구명","행정동명"]).size().reset_index(name="점포수")
                .sort_values("점포수").head(rec_top_n)
            )
            rec_low["지역"] = rec_low["시군구명"] + " " + rec_low["행정동명"]
            fig_rec2 = go.Figure(go.Bar(
                x=rec_low["점포수"], y=rec_low["지역"],
                orientation="h", marker_color="#3fb950",
                hovertemplate="%{y}: %{x}개<extra></extra>",
            ))
            fig_rec2.update_layout(
                xaxis_title="점포 수 (적을수록 경쟁 약함)",
                yaxis=dict(autorange="reversed"),
                height=max(300, rec_top_n * 26),
                margin=dict(t=20, b=40, r=80),
            )
            st.plotly_chart(fig_rec2, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════
        # [기능 4] 주변 상권 검색 (행정동 상권 생태계)
        # ══════════════════════════════════════════════════════
        st.markdown("#### 🔍 주변 상권 검색")
        st.caption("행정동을 선택하면 해당 동네의 업종 구성과 점포 목록을 한눈에 확인할 수 있습니다.")

        srch_sido = st.selectbox("시/도", sido_list, key="srch_sido")
        srch_gu_pool = sorted(semas_df[semas_df["시도명"] == srch_sido]["시군구명"].dropna().unique())
        srch_gu = st.selectbox("시/군/구", srch_gu_pool, key="srch_gu")
        srch_dong_pool = sorted(
            semas_df[(semas_df["시도명"] == srch_sido) & (semas_df["시군구명"] == srch_gu)]["행정동명"].dropna().unique()
        )
        srch_dong = st.selectbox("행정동", srch_dong_pool, key="srch_dong")

        srch_df = semas_df[
            (semas_df["시도명"] == srch_sido) &
            (semas_df["시군구명"] == srch_gu) &
            (semas_df["행정동명"] == srch_dong)
        ]

        if srch_df.empty:
            st.info("해당 행정동의 상권 데이터가 없습니다.")
        else:
            sr1, sr2 = st.columns([1, 1])

            with sr1:
                st.markdown("##### 업종 대분류 구성")
                pie_data = srch_df["상권업종대분류명"].value_counts().reset_index()
                pie_data.columns = ["업종", "점포수"]
                fig_pie = go.Figure(go.Pie(
                    labels=pie_data["업종"],
                    values=pie_data["점포수"],
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Safe,
                    hovertemplate="%{label}: %{value}개 (%{percent})<extra></extra>",
                ))
                fig_pie.update_layout(height=340, margin=dict(t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with sr2:
                st.markdown("##### 업종 중분류 TOP 15")
                mid_cnt = (
                    srch_df["상권업종중분류명"].value_counts()
                    .head(15).reset_index()
                )
                mid_cnt.columns = ["업종(중분류)", "점포수"]
                fig_mid = go.Figure(go.Bar(
                    x=mid_cnt["점포수"],
                    y=mid_cnt["업종(중분류)"],
                    orientation="h",
                    marker_color="#58a6ff",
                    hovertemplate="%{y}: %{x}개<extra></extra>",
                ))
                fig_mid.update_layout(
                    xaxis_title="점포 수",
                    yaxis=dict(autorange="reversed"),
                    height=340,
                    margin=dict(t=10, b=10, r=60),
                )
                st.plotly_chart(fig_mid, use_container_width=True)

            # 요약 메트릭
            m1, m2, m3 = st.columns(3)
            m1.metric("총 점포 수", f"{len(srch_df):,}개")
            m2.metric("업종 대분류 종류", f"{srch_df['상권업종대분류명'].nunique()}개")
            m3.metric("업종 중분류 종류", f"{srch_df['상권업종중분류명'].nunique()}개")

            st.markdown("##### 점포 목록")
            show_cols = ["상호명","상권업종대분류명","상권업종중분류명","상권업종소분류명","도로명주소"] \
                if "도로명주소" in srch_df.columns else \
                ["상호명","상권업종대분류명","상권업종중분류명","상권업종소분류명"]
            # 도로명주소가 semas_df에 없으므로 안전하게 처리
            available = [c for c in show_cols if c in srch_df.columns]
            st.dataframe(
                srch_df[available].reset_index(drop=True),
                use_container_width=True,
                height=300,
            )
