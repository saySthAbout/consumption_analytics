import io
import os
import glob
import zipfile
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

FLOWPOP_COMBINED_ZIP_PATH = os.path.join(DATASET_DIR, "flowpop_admi_202601-202603.zip")
FLOWPOP_COMBINED_YYYYMM   = {"202601", "202602", "202603"}
FLOWPOP_COMBINED_ID       = "1CI89pcksxhFkfkVdxnoKqvSpSKarEJ7R"

# 월별 개별 ZIP (2025-04 ~ 2025-12)
FLOWPOP_MONTHLY_IDS = {
    "202504": "18fU8qLk_kHmS4K4ZHbQlATVEA2eo5TYM",
    "202505": "1AaPrEuiSgTPDJp4Jbi7a_0cwIzsUi8TG",
    "202506": "1CR4kylS19Y76yMw6B0Qwusp92-Ukz8cy",
    "202507": "1Cz4CngfRrp7W3DaOc4B0SqEepV0iJbp6",
    "202508": "1Ic_31gRQXWyg2C2lG4AIsn0tiI9RN_zM",
    "202509": "1ONnu8Lkgc45yyYpyXxO0XBacrIgCFj6S",
    "202510": "1_-BmxjNhLP-csmxhVArBposVdFvsT3XR",
    "202511": "1hyMjrEjwwFZRfcRrXzVSLsJKI-tDdVWV",
    "202512": "1vWTMZgON8X-IHmbnZAw1MWTzK5O6OkR0",
}

def get_flowpop_zip_path(yyyymm: str) -> str:
    if yyyymm in FLOWPOP_COMBINED_YYYYMM:
        return FLOWPOP_COMBINED_ZIP_PATH
    return os.path.join(DATASET_DIR, f"flowpop_admi_{yyyymm}.zip")

def _hf_download(fname: str, dest_path: str, label: str) -> bool:
    """HuggingFace Dataset에서 파일 다운로드 (진행률 표시)."""
    import requests
    url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{fname}"
    try:
        resp = requests.get(url, stream=True, timeout=300,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 404:
            st.error(f"{fname} 파일을 HuggingFace에서 찾을 수 없습니다.")
            return False
        resp.raise_for_status()
        total_size = int(resp.headers.get("Content-Length", 0))
        chunk_size = 1 * 1024 * 1024  # 1MB
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        status_text = st.empty()
        progress_bar = st.progress(0)
        downloaded = 0

        with open(dest_path, "wb") as fp:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fp.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = min(downloaded / total_size, 1.0)
                        mb_done = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        progress_bar.progress(pct)
                        status_text.caption(
                            f"📥 {label} 다운로드 중... "
                            f"{mb_done:.1f} MB / {mb_total:.1f} MB  ({pct*100:.0f}%)"
                        )
                    else:
                        mb_done = downloaded / 1024 / 1024
                        status_text.caption(f"📥 {label} 다운로드 중... {mb_done:.1f} MB")

        progress_bar.progress(1.0)
        status_text.caption(f"✅ {label} 다운로드 완료 ({downloaded/1024/1024:.1f} MB)")
        return True
    except Exception as e:
        st.error(f"{label} 다운로드 오류: {e}")
        return False


def ensure_flowpop_zip(yyyymm: str) -> bool:
    """해당 월 flowpop ZIP이 없거나 깨져 있으면 HuggingFace에서 다운로드."""
    path = get_flowpop_zip_path(yyyymm)
    if os.path.exists(path) and _is_valid_zip(path):
        return True
    if os.path.exists(path):
        os.remove(path)
    if yyyymm in FLOWPOP_COMBINED_YYYYMM:
        fname = "flowpop_admi_202601-202603.zip"
        label = f"유동인구 {YYYYMM_LABEL.get(yyyymm, yyyymm)}"
        ok = _hf_download(fname, path, label)
        return ok and _is_valid_zip(path)
    elif yyyymm in FLOWPOP_MONTHLY_IDS:
        label = f"유동인구 {YYYYMM_LABEL.get(yyyymm, yyyymm)}"
        fname = f"flowpop_admi_{yyyymm}.zip"
        # HuggingFace 우선 시도
        ok = _hf_download(fname, path, label)
        if ok and _is_valid_zip(path):
            return True
        # HF 실패 시 Google Drive fallback
        if os.path.exists(path):
            os.remove(path)
        file_id = FLOWPOP_MONTHLY_IDS[yyyymm]
        try:
            _gdrive_download(file_id, path, label)
        except Exception as e:
            st.error(f"유동인구 다운로드 오류: {e}")
            return False
        return _is_valid_zip(path)
    return False

SEMAS_DIR      = DATASET_DIR
SEMAS_ZIP_NAME = "semas_store_info_202603.zip"
SEMAS_ZIP_PATH = os.path.join(DATASET_DIR, SEMAS_ZIP_NAME)
SEMAS_GDRIVE_FILE_ID = ""  # HuggingFace로 이전

CARD_CSV_DIR = os.path.join(DATASET_DIR, "card_csvs")

# 140개 개별 CSV 파일 Drive ID 목록
CARD_FILE_IDS = [
    "1-3pAZuCvLAkRRWs-sb1onUpkpxlXw2gS","1-VJpHQUFtRszuPAqP69v6Yom94tJk3dC",
    "1-axkNlIOnGTbSEe58ytQL5SUXOOriQml","102Qp1t4DfG7293HBqUuHfSRTqmJN1xKD",
    "109yoCUh5byvP8t_dF5wtoreoI5ZPzL29","10taGIxHTaYCkTimv_jJWD-Jd3VtKWOVW",
    "11WS7Bz_3jUCIfCVuE0tTzRL5uyWbjJ0w","11_Hm7R88HIDUkDUNaIxFwfjTKGSTkmh4",
    "11bzOh_jJqyuYt9BZXBdoUuodt1X585fn","11tsh9kFTGHKUCyf_8Clu94fB2r9yGcR2",
    "12SCHkMxAyv6pHcNL7js2IHuEMUdKh5dd","12kEF6ZCbH716UwPcknLnUnyRhbOvepgR",
    "12zCrZ63LshOFn_3k0cIv9XVcf2R9P86g","132Om4qXZ8S71NEkAc8jz1Aix-InIlhcO",
    "135rdfWjilzgNjXi3V26tlZPyma4wpgJR","13_Zw3vNOlIMQBt_TsVNl2-6EHkBiip-5",
    "15pJPylZa2HetElGWlbcCZq-RsjhRH1mn","168R1aFtvezaobUVkrRamx_pu6ZF1d5k9",
    "16PEc97-tp3k6p4Ss0cMLgEcOJPbhm2FI","17Ut_y7MjPV9zSuDOf99vOMP0UsKtiLIx",
    "18yKo4gL1dTgOPbol8PD1jr54_8OqFCA0","18zTiXQb0nTPXVlnEVNukbRtujjPNV6vF",
    "196Ur0-z_TazDJnIaJUg-MJepBE1An1aI","19nIIjArgJWANDsxdVrTCm0GALA8xUdWb",
    "1A6vyHsZNdt5tZIFZCcPdI-z19Qt1_9VG","1ALw_N0s4XstR4TmlyYYabPLGv31aYx2s",
    "1Adqk9ioPIukCIjIN3Wncz3SoB6MOi3hS","1AiEyf0wFL9RsoR9XPoXScC8ljreu7Sc6",
    "1B8nFYKSaNaSlVEW2fE2JnQ6t8LmiHZFX","1C5kv9BHtiVzM8F6f_7RxENWqjDidFx_t",
    "1CLIN6b2hAym1Z3PufuO8XNqfGj2SGgAs","1Cq1ugU2gihHCiv1EEp1H2lRrt6NXUhCS",
    "1Cxo4AIQhtWIC5wiI2QLeTcnorjZT6pTB","1DEAnn_u0bBQSkaUFFI_UEtjmoINU2Eni",
    "1DM7bShml5ejo6C73QFPguIFFEWA1JlRA","1E5oCNA4YIJUSO2lCLsra1_OfYG26hsYQ",
    "1EG87hirfNT896u2-P38VtRvWF591rAaR","1EQqdejs23kH8dj_2D4MjQb03szO-uBsf",
    "1EaX6Cbg4dx9krOmM_W4shp0nBw4WTBDK","1EqGcT-bbHUe9TLNzpDTxQViD00xg92Yp",
    "1ErWtQtJVfcD2BQxnJ5b3ErSrVb1o_2WH","1GA6Y8b6SHRoa-B21QMFaB9jc2S_y4ieO",
    "1GdKs73XGN4Lb_tmPCHv5ytO06L-rBqYp","1Gqwdu0IAdMe37XBe-zs-fIKITV4etTWV",
    "1GuYZWEAY0IrDcrLuwAKDlPOxEXE14yRJ","1H5Ck3EpJiQ5SsnsgekgGGVcW4KKG9rNx",
    "1HPp8VFfZgS5QfoPSx3EOl_vUEbJ9NbLK","1IZv1632xLJBeF6gDQZLoHMlAbB8NYFgw",
    "1ImPDU5H9-tMgdodINneqv1YP23EKWRuh","1IrREy9pEM2BwhscX6iVA0Lh8MQcfqBqh",
    "1IvL0EEhsh1fW2FQlL5E8oIGzGBZe1XoC","1Ix0onecekPb2sgF_dAuO8kshNnrbcwPd",
    "1Jj3XyeSQr8xodBZVzvpV6R9UKpK-DFHX","1JnqB9Yu0eo_VO5iOKkxZHMhTFAH17BPh",
    "1K0PBm6OgnAQwkvwNz_FB-YMmDEjDMCDr","1KV7WJV8YDFNPEeQmJ9VKSXLIKCdb1TUM",
    "1Kt9ZsDJYdrBt8VC7r5B85zJT94q1nMB_","1KzpEMZw2BxcMJxglpWfLVAoodSMR_tEF",
    "1LJvPl45Jz7ZLRqjCq0f0L8nCy6xe7lc_","1LwZ9LUDoSRagzvF6HNfdnR75CTqCERCf",
    "1MXL1Co_AFMJpNklLZ0J2_IVDBPYJfxPU","1MqxIMfuDzgMrWYFMGbt0-0s3c91fJNp0",
    "1NVkYghC6laKnGheZhbWlHny7QJkg0z2R","1OfwlqP-2hRoz6lLHjJulnvZQkBWagSBU",
    "1PtfX9zBmynYaQWJed_lTiAye2dqasGgh","1R11TDy_Q-3ExIB8KBpnpXKNuJJcE6WrV",
    "1R4VdKLoT23PaFTCZFjnxBztHK4k4J9Y6","1Rejq6wlpSTX4KmjM8164KOhCaMdMQEYh",
    "1Rl50q4zc6hk3JKDSsYF4eJdP2gYbrrsP","1RlNpPH349v81e4uwLvwtojUM1aPOVIlN",
    "1S9DGmu4tcCe9JpqQhcK7dL-AM9hnRvVV","1S_3P3sUaVxnP9chTW0xVSUAle2UTlSgo",
    "1T1lgmsCwDp4X3nuFHxrQ4GKcRw1bFfqx","1T5gcC8Ze7C8N5GIVtvXzcmTBJgP5x4Fg",
    "1U2v_9wujImPlE0BYXmYRrDTMQ7BSmzus","1VG6mW_pRCaMiVcoj1zr9SV5uE8IV00tK",
    "1VasYJZkOfGJgBCpGV92ZVCjypEvWA8Uy","1VxpBgFX6VdFvu6qqLaWzPQCyex-X1y0A",
    "1WVTnUy8MBKC3rnkFu6Acxzlpk0NIcH0l","1XImAoAg67Mluy1Gw_jcQY8QdumJHYruK",
    "1YGyxv2r54rnfOLJn1xK7ujoTb5ECYyQ0","1Z4x9uPEIaUxdygmdPXmqkrGejlinkPjv",
    "1_2R5XRyH25I8IbbpjT9zJpU1R3vudgQs","1_5rj-9TOYQ1MxoQjJEJQSRbCs8TOuO5A",
    "1_6PCCFGv2w7zMMtnrS5ryvevvDW19niz","1_UdVpJQOXX0nQs6CPG4rcxcvppfQxPOp",
    "1_V4wdu2C9qO75S0OA2c5qorM2db4Yl8N","1_pi2DdOWS-EkFYNrnUT1SHCTQTLmbVH-",
    "1a7leFmTm0r9cOVaIFosg-5xhyCVJh4Ff","1akKBYs8KJq8-nn2JSxoG9KeOCtP_d5a-",
    "1aqpgujqg0EZJ1WoL_ggHFmxfWZqtl6Oj","1bQkgD8S7v7Qrm8dOrSkGt7hJQ2yd2bP4",
    "1bkXOpDIQoxMP2QeqP_zOENaF4gXANmf4","1cNYcBLdld5MHiJ73I0o0nNrUy2aB-7HG",
    "1caH5KNOAVH6d45alrea47HAG7MBjNbmL","1dE7c7k7KkAdCNAafBN6iegyPF7vDepGl",
    "1dVhnTUV0Yg20OOjW2XKUVV-_cgbstJQ5","1fnxS9wGEGcYTCdg4iGOs9zpm4SUSSMZ5",
    "1gwbFLX-FDhcknzedtj4pkS4M072UlTtF","1hHIEU-NyH_m0FBASdlliUSc6Odv66bxe",
    "1he1WER3cIeZudJb57u4GxDMpKh8X4V2c","1iYD_pHXwURHv3QudBWfYC-_CUuQGmVOn",
    "1iey45-XXnXSQFL2mc-w8dcPUCjpXFeCx","1ir9j7IldHIYxAtI5DNeTKQHxrYuDCipx",
    "1itH68-fmA210AZOr-AOwZwHpxglRQFWA","1jGn9oiIL2a1GMonpI3KMcGOmuKY-rcdB",
    "1kLAG-WgWQyzBmTgTnePQMpz3FBzgJ_9O","1kLx45wxLgksrv0EWUjb8mGsVp4s-7QHw",
    "1kTX8CE6-V-D7wUUT-2XzwdMNefZqYIui","1l6mEZBNq2bScXtjbzunOVKX7PWFXggwB",
    "1lDxBW59oKK1-sTavn0JZc7jst2OfJrG7","1lVB1y61Ssic03Vn8TO5F6LsZpQSLmWZs",
    "1mYh_trO15DPouZccp4uuOY_9vfzzimLi","1mfMPKcGJZnk-HC83Jf5W9vCgEjBslJ2s",
    "1o8rbCU3GwuIbukcdzopTFqNNkxz5bKds","1ommdybTOealR5JVhko_5SNdtArmprq3-",
    "1q2OJrhrXU7NPn2PMDVcc3WcKhoK7dOad","1q6wTBqRU7FzKgjK39ulRBzkHiIcnWvz_",
    "1qpcXVz8SMZOjhcgst56rIfh_VuDUMO7-","1r3eyiqBc55EELf7r1TLQFiE48rS_8tz2",
    "1rPzFb24ACNDDOuL1x9TKFbhI3LwPpVLJ","1rlgJTUIqtpL0tZf14OWAJnmVRgbSA8di",
    "1sQs9Wp-fFv89etsbB137o0r6saLGOPfj","1sXoNoaIm4CFI0NAeed5rSCsdmdTOtEmN",
    "1sZxDNCC-FcVz9DGokIEfRbaqCGimPuom","1tSvNkmP4DVz9I-lR62RNiNuAVYxOTv0k",
    "1tgHO2cmfslsdx-Jo457ijw58hDQSojab","1tyo3SqP6-wSKEeV3MnPDFH6TcfAh0Ycw",
    "1u-PjQfZDicYtaqev_prwDoyd7dsUJrgd","1uDoUvNO9piz8oQ-nwx0-cXXNCZg5sWBb",
    "1vkC0XakqYDnnF1u4A4F_sI4h0k1Ee_wz","1vy3m9ziDFhs88NebBiC3vSVoz7ixv1-H",
    "1w9ZxRvf_LpBbvtUDfu-x-8LxLwdX5IHs","1x4CkqcGyc7gxzmgUbBkXncLB6f8dvBc4",
    "1x68VDK4ejfrz5xfucHsMGT-OOX_EEJqd","1xZhX6FQNpPAdntUUvaYTJ0M7kmv28kGK",
    "1xfX12az_715RmxV1BX9hLy5LfUTUaYN-","1xk5p0hhuuzGQgsrSwa4zxemwKrcw1Wqn",
    "1xse4zxfInecF_WsTwXrPg_C_lMLI_S6M","1yBN85LCHwQgq39YEJ9A9rpZWS8QwuO8m",
    "1yesrAQ7HQ4XPCTBZz9016yFicq8zFjXl","1yvm_a3I_fl7rnWc9-tv6qIwSSPyP54cv",
    "1z2tQkddkrHxdE26leXoaEzdHEIJEYOvd","1zIxFQQ3NaFP2yL82dtPnewEqMNvoT8O0",
    "1zUy8yl7qFEqjWImhW0bcZj_mnl8FKB6T","1zaz2BBY74-o9aw4y1hpo159FdtE8FuxM",
    "1zlDoeInZqNdadp3HW_B-weqrlf6NgT8K",
]

# 2025-04 ~ 2026-03 (12개월)
AVAILABLE_YYYYMM = [
    "202504","202505","202506","202507","202508","202509",
    "202510","202511","202512","202601","202602","202603",
]
YYYYMM_LABEL = {
    "202504":"2025년 4월","202505":"2025년 5월","202506":"2025년 6월",
    "202507":"2025년 7월","202508":"2025년 8월","202509":"2025년 9월",
    "202510":"2025년 10월","202511":"2025년 11월","202512":"2025년 12월",
    "202601":"2026년 1월","202602":"2026년 2월","202603":"2026년 3월",
}


_BIZ_CAT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biz_categories.json")

def _load_biz_categories():
    try:
        import json
        with open(_BIZ_CAT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"biz1": [], "biz2": {}}

_BIZ_CATS = _load_biz_categories()
BIZ1_OPTS = _BIZ_CATS.get("biz1", [])
BIZ2_MAP  = _BIZ_CATS.get("biz2", {})


def _is_valid_zip(path):
    """Python zipfile로 ZIP 유효성 확인."""
    import zipfile as _zf
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    try:
        return _zf.is_zipfile(path)
    except Exception:
        return False


def _gdrive_download(file_id: str, output_path: str, label: str = "파일",
                     expected_mb: float = 500):
    """wget으로 Google Drive 직접 다운로드 (가장 빠름)."""
    import subprocess, shutil
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dl_url = (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )

    wget_bin = shutil.which("wget") or shutil.which("curl")
    if not wget_bin:
        raise RuntimeError("wget/curl 없음")

    spinner_text = f"{label} 다운로드 중... (완료될 때까지 기다려 주세요)"
    with st.spinner(spinner_text):
        if "wget" in wget_bin:
            cmd = [wget_bin, "-q", "--show-progress",
                   "--no-check-certificate", "-O", output_path, dl_url]
        else:
            cmd = [wget_bin, "-L", "--silent", "--show-error",
                   "-o", output_path, dl_url]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if proc.returncode != 0:
        raise RuntimeError(f"다운로드 오류: {proc.stderr[:300]}")
    size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    if size < 1024 * 100:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"다운로드 크기 이상 ({size} bytes) — Drive 파일 확인 필요")



def _extract_cd_fname(headers, file_id: str) -> str:
    import re
    cd = headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)', cd, re.IGNORECASE)
    return m.group(1).strip() if m else f"card_{file_id}.csv"


def _download_gdrive_csv(file_id: str, dest_dir: str) -> str | None:
    """Drive 파일을 Content-Disposition 파일명으로 dest_dir에 저장. 저장된 경로 반환."""
    import requests
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    try:
        resp = requests.get(url, stream=True, timeout=120, allow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        # HTML 응답(쿼터 초과 등)이면 즉시 중단
        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            resp.close()
            st.error("Google Drive 다운로드 쿼터 초과. 잠시 후 다시 시도해주세요.")
            return None
        fname = _extract_cd_fname(resp.headers, file_id)
        out = os.path.join(dest_dir, fname)
        size = 0
        with open(out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
        if size < 1024:
            os.remove(out)
            return None
        # 다운로드 성공 → 파일명→ID 맵 업데이트 (메모리 + 디스크)
        fmap = st.session_state.get("_fileid_map", {})
        fmap[fname] = file_id
        st.session_state["_fileid_map"] = fmap
        _save_fileid_map(fmap)
        return out
    except Exception:
        return None


def ensure_month_csvs(yyyymm: str) -> bool:
    """해당 월 CSV가 없으면 미다운로드 파일 ID를 순차 다운로드."""
    os.makedirs(CARD_CSV_DIR, exist_ok=True)

    if glob.glob(os.path.join(CARD_CSV_DIR, f"*{yyyymm}*.csv")):
        return True

    # 이미 다운로드된 파일 ID를 파일명으로 추적
    existing_names = {os.path.basename(p) for p in glob.glob(os.path.join(CARD_CSV_DIR, "*.csv"))}
    pending = [fid for fid in CARD_FILE_IDS
               if not any(fid[:10] in n for n in existing_names)]

    if not pending:
        st.error(f"모든 파일 다운로드 완료됐으나 {yyyymm} 데이터 없음")
        return False

    progress = st.progress(0.0, text=f"카드 데이터 다운로드 중... 0/{len(pending)}")
    for i, fid in enumerate(pending, 1):
        _download_gdrive_csv(fid, CARD_CSV_DIR)
        progress.progress(i / len(pending), text=f"카드 데이터 다운로드 중... {i}/{len(pending)}")
        if glob.glob(os.path.join(CARD_CSV_DIR, f"*{yyyymm}*.csv")):
            progress.empty()
            return True

    progress.empty()
    return bool(glob.glob(os.path.join(CARD_CSV_DIR, f"*{yyyymm}*.csv")))


def load_month_csv(yyyymm: str):
    """로컬 CSV 디렉터리에서 해당 월 모든 도시 파일을 합쳐 반환."""
    candidates = sorted(glob.glob(os.path.join(CARD_CSV_DIR, f"*{yyyymm}*.csv")))
    if not candidates:
        raise FileNotFoundError(f"{CARD_CSV_DIR} 에 {yyyymm} CSV 없음")
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    frames, enc_used = [], "utf-8-sig"
    for path in candidates:
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, dtype=_SALES_DTYPES)
                enc_used = enc
                frames.append(df)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"{os.path.basename(path)} 읽기 실패")
    return pd.concat(frames, ignore_index=True), enc_used, candidates[0]

def ensure_semas_data():
    """SEMAS zip이 없으면 HuggingFace에서 다운로드 후 압축 해제."""
    if glob.glob(os.path.join(SEMAS_DIR, "semas_store_info_*.csv")):
        return
    if os.path.exists(SEMAS_ZIP_PATH):
        _extract_semas_zip()
        return
    ok = _hf_download(SEMAS_ZIP_NAME, SEMAS_ZIP_PATH, "상권 데이터 ZIP (약 240MB)")
    if ok:
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
    1: "00:00 - 06:59", 2: "07:00 - 08:59", 3: "09:00 - 10:59",
    4: "11:00 - 12:59", 5: "13:00 - 14:59", 6: "15:00 - 16:59",
    7: "17:00 - 18:59", 8: "19:00 - 20:59", 9: "21:00 - 22:59",
    10: "23:00 - 23:59"
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
def load_sales_data(yyyymm: str):
    raw, enc, path = load_month_csv(yyyymm)
    return raw, enc, path


# 한글 시/구명 → CSV 파일명 영문 접미어 매핑
CITY_KO_TO_EN: dict[str, str] = {
    "안산시":   "ansan",       "안양시":  "anyang",     "김포시":  "gimpo",
    "광명시":   "gwangmyeong", "하남시":  "hanam",      "화성시":  "hwaseong",
    "남양주시": "namyangju",   "파주시":  "paju",       "포천시":  "pocheon",
    "성남시":   "seongnam",    "시흥시":  "siheung",    "수원시":  "suwon",
    "의정부시": "uijeongbu",   "여주시":  "yeoju",      "용인시":  "yongin",
    "과천시":   "gwacheon",    "이천시":  "icheon",     "연천군":  "yeoncheon",
}

# 도시별 가용 월 (HuggingFace Dataset 기준)
_AVAIL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "available_data.json")
def _load_available_data() -> dict[str, list[str]]:
    try:
        import json
        with open(_AVAIL_DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
CITY_AVAILABLE_MONTHS: dict[str, list[str]] = _load_available_data()  # city_en -> [yyyymm]

def get_available_months_for_city(city_korean: str) -> list[int]:
    """도시 한글명 → 가용 월(int) 리스트. 없으면 AVAILABLE_YYYYMM 전체 반환."""
    city_en = CITY_KO_TO_EN.get(city_korean)
    if not city_en or city_en not in CITY_AVAILABLE_MONTHS:
        return sorted({int(m[4:]) for m in AVAILABLE_YYYYMM})
    return sorted({int(m[4:]) for m in CITY_AVAILABLE_MONTHS[city_en]})


_FILEID_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_csvs", "_fileid_map.json")


def _load_fileid_map() -> dict[str, str]:
    """디스크 + session_state에서 파일명→ID 맵 로드."""
    fmap = {}
    if os.path.exists(_FILEID_MAP_PATH):
        try:
            import json
            with open(_FILEID_MAP_PATH, encoding="utf-8") as f:
                fmap = json.load(f)
        except Exception:
            fmap = {}
    fmap.update(st.session_state.get("_fileid_map", {}))
    st.session_state["_fileid_map"] = fmap
    return fmap


def _save_fileid_map(fmap: dict) -> None:
    """파일명→ID 맵을 디스크에 저장 (앱 재시작 후 재사용)."""
    try:
        import json
        os.makedirs(os.path.dirname(_FILEID_MAP_PATH), exist_ok=True)
        with open(_FILEID_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(fmap, f)
    except Exception:
        pass


HF_DATASET_REPO = "JinAhKwak/gyeonggi-card-csvs"


def ensure_city_month_csv(city_korean: str, yyyymm: str) -> bool:
    """선택 도시·월 CSV 파일 1개를 HuggingFace Dataset에서 다운로드."""
    import requests
    os.makedirs(CARD_CSV_DIR, exist_ok=True)
    city_en = CITY_KO_TO_EN.get(city_korean)

    # 이미 파일 존재하면 즉시 반환
    if city_en and glob.glob(os.path.join(CARD_CSV_DIR, f"*{yyyymm}*{city_en}*.csv")):
        return True
    if not city_en and glob.glob(os.path.join(CARD_CSV_DIR, f"*{yyyymm}*.csv")):
        return True

    target_name = f"tbsh_gyeonggi_day_{yyyymm}_{city_en}.csv" if city_en else None
    if not target_name:
        return False

    label = f"{YYYYMM_LABEL.get(yyyymm, yyyymm)} {city_korean}"
    url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{target_name}"
    out = os.path.join(CARD_CSV_DIR, target_name)

    with st.spinner(f"{label} 데이터 다운로드 중..."):
        try:
            resp = requests.get(url, stream=True, timeout=120,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 404:
                st.warning(f"⚠️ {city_korean} {yyyymm[4:]}월 카드 데이터가 제공되지 않습니다. 다른 지역 또는 월을 선택해주세요.")
                return False
            resp.raise_for_status()
            size = 0
            with open(out, "wb") as fp:
                for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        fp.write(chunk)
                        size += len(chunk)
            if size < 1024:
                os.remove(out)
                return False
            return True
        except Exception as e:
            st.error(f"다운로드 실패: {e}")
            return False


def _city_month_in_df(df, city_district: str, month_int: int | None = None) -> bool:
    """df에 해당 지역(+월) 데이터가 실제로 있는지 확인."""
    if df is None or df.empty or "admi_cty_no" not in df.columns:
        return False
    city_prefixes = [k for k, v in DISTRICT_MAP.items() if v == city_district]
    if not city_prefixes:
        return True  # 알 수 없는 지역은 있다고 가정
    mask = df["admi_cty_no"].astype(str).str[:5].isin(city_prefixes)
    if month_int is not None and "month" in df.columns:
        mask = mask & (df["month"] == month_int)
    return bool(mask.any())


def ensure_month_in_df(month_int: int, city_korean: str | None = None) -> bool:
    """df에 해당 월·도시 데이터가 없으면 다운로드 후 merge."""
    df_cur = st.session_state.get("df")
    # AVAILABLE_YYYYMM에서 해당 month_int와 일치하는 yyyymm 찾기
    yyyymm = next((m for m in AVAILABLE_YYYYMM if int(m[4:]) == month_int), None)
    if yyyymm is None:
        st.warning(f"{month_int}월 데이터는 제공되지 않습니다.")
        return False

    # "안양시 만안구" → "안양시" 처럼 시 단위로 정규화
    city_base = None
    if city_korean:
        city_base = next((c for c in CITY_KO_TO_EN if city_korean.startswith(c)), None)

    # 이미 해당 월 데이터 있는지 확인
    if df_cur is not None and "month" in df_cur.columns and month_int in df_cur["month"].values:
        if city_base is None:
            return True
        city_en = CITY_KO_TO_EN.get(city_base)
        if city_en is None:
            return True
        # 파일 존재 여부가 아니라 실제 df에 해당 도시+월 데이터가 있는지 확인
        city_prefixes = [k for k, v in DISTRICT_MAP.items() if v.startswith(city_base)]
        if city_prefixes:
            mask = (
                df_cur["admi_cty_no"].astype(str).str[:5].isin(city_prefixes) &
                (df_cur["month"] == month_int)
            )
            if mask.any():
                return True
        elif glob.glob(os.path.join(CARD_CSV_DIR, f"*{yyyymm}*{city_en}*.csv")):
            return True

    # 도시 지정이 있으면 해당 도시 파일만 다운로드
    if city_base:
        csv_ok = ensure_city_month_csv(city_base, yyyymm)
    else:
        csv_ok = ensure_month_csvs(yyyymm)
    if not csv_ok:
        return False

    try:
        with st.spinner(f"{month_int}월 데이터 로드 중..."):
            raw_new, _, _ = load_sales_data(yyyymm)
            df_new = preprocess_data(raw_new)
        merged = pd.concat([df_cur, df_new], ignore_index=True).drop_duplicates() if df_cur is not None else df_new
        st.session_state["df"] = merged
        st.rerun()
    except Exception as e:
        st.error(f"{month_int}월 데이터 로드 실패: {e}")
        return False
    return True


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
    "41150": "안양시 만안구",
    "41171": "안양시 만안구", "41173": "안양시 동안구",
    "41210": "부천시",
    "41270": "광명시",
    "41271": "안산시 상록구", "41273": "안산시 단원구", "41275": "안산시 상록구", "41290": "안산시",
    "41360": "남양주시",
    "41390": "시흥시",
    "41450": "하남시",
    "41461": "용인시 처인구", "41463": "용인시 기흥구", "41465": "용인시 수지구",
    "41480": "과천시",
    "41500": "이천시",
    "41570": "의정부시",
    "41590": "화성시", "41591": "화성시", "41593": "화성시", "41595": "화성시", "41597": "화성시",
    "41630": "포천시",
    "41650": "파주시",
    "41670": "김포시",
    "41800": "여주시",
    "41820": "연천군",
}

AGE_COLS_M = [f"M_{a}_CNT" for a in [10,15,20,25,30,35,40,45,50,55,60,65,70]]
AGE_COLS_F = [f"F_{a}_CNT" for a in [10,15,20,25,30,35,40,45,50,55,60,65,70]]

def _age_label(col: str) -> str:
    n = int(col.split("_")[1])
    if n < 15:  return "10대 미만"
    if n < 20:  return "10대"
    if n < 30:  return "20대"
    if n < 40:  return "30대"
    if n < 50:  return "40대"
    if n < 60:  return "50대"
    if n < 70:  return "60대"
    return "70대 이상"

@st.cache_data
def load_flowpop_data(zip_path: str) -> dict:
    """파일별로 읽으면서 즉시 집계 → 작은 집계 DataFrame 딕셔너리 반환."""
    import zipfile, io

    hm_frames, age_frames, daily_frames = [], [], []

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            try:
                with z.open(name) as f:
                    chunk = pd.read_csv(io.BytesIO(f.read()), encoding="utf-8")

                chunk["ETL_YMD"] = pd.to_datetime(
                    chunk["ETL_YMD"].astype(str), format="%Y%m%d", errors="coerce")
                chunk["DOW"] = chunk["ETL_YMD"].dt.dayofweek

                am = [c for c in AGE_COLS_M if c in chunk.columns]
                af = [c for c in AGE_COLS_F if c in chunk.columns]
                chunk["TOTAL_CNT"]  = chunk[am + af].sum(axis=1)
                chunk["MALE_CNT"]   = chunk[am].sum(axis=1)
                chunk["FEMALE_CNT"] = chunk[af].sum(axis=1)

                key_cols = ["CTY_NM", "ADMI_NM", "ADMI_CD", "FORN_GB"]

                # ① 히트맵용: 시간대×요일×지역 집계
                hm = chunk.groupby(key_cols + ["TIME_CD", "DOW"])["TOTAL_CNT"].sum().reset_index()
                hm_frames.append(hm)

                # ② 연령대·성별용: 지역×성별 집계 (연령대 컬럼 유지)
                ag_cols = key_cols + am + af + ["TOTAL_CNT", "MALE_CNT", "FEMALE_CNT"]
                ag = chunk[ag_cols].groupby(key_cols).sum().reset_index()
                age_frames.append(ag)

                # ③ 매출 상관용: 일별×행정동 집계
                dl = chunk.groupby(["ETL_YMD", "ADMI_CD"])["TOTAL_CNT"].sum().reset_index()
                daily_frames.append(dl)

            except Exception:
                pass

    if not hm_frames:
        return {}

    grp = ["CTY_NM", "ADMI_NM", "ADMI_CD", "FORN_GB"]
    heatmap_df = (pd.concat(hm_frames, ignore_index=True)
                    .groupby(grp + ["TIME_CD", "DOW"])["TOTAL_CNT"].sum().reset_index())
    age_df     = (pd.concat(age_frames, ignore_index=True)
                    .groupby(grp).sum().reset_index())
    daily_df   = (pd.concat(daily_frames, ignore_index=True)
                    .groupby(["ETL_YMD", "ADMI_CD"])["TOTAL_CNT"].sum().reset_index())

    return {"heatmap": heatmap_df, "age": age_df, "daily": daily_df}


def load_semas_data(semas_dir: str, zip_path: str) -> dict:
    """SEMAS 데이터를 파일별로 읽으면서 즉시 집계 — 작은 집계 DataFrame 딕셔너리 반환."""
    import zipfile, io

    use_cols = [
        "상호명", "상권업종대분류명", "상권업종중분류명", "상권업종소분류명",
        "시도명", "시군구명", "행정동명", "경도", "위도",
    ]
    key_cols  = ["시도명", "시군구명", "행정동명"]
    MAP_SAMPLE_PER_FILE = 400   # 파일당 지도 샘플 수 (16파일 × 400 = 6,400개)

    cnt_frames, map_frames, dong_frames = [], [], []

    def _process(chunk):
        chunk = chunk.dropna(subset=["상권업종대분류명"])
        chunk["경도"] = pd.to_numeric(chunk.get("경도", pd.Series(dtype=float)), errors="coerce")
        chunk["위도"] = pd.to_numeric(chunk.get("위도", pd.Series(dtype=float)), errors="coerce")

        # ① 경쟁 강도·입지 추천용: 동×중분류 점포 수
        cnt = (chunk.groupby(key_cols + ["상권업종대분류명", "상권업종중분류명"])
               .size().reset_index(name="점포수"))
        cnt_frames.append(cnt)

        # ② 지도용: 좌표 있는 행만 샘플링
        has_coord = chunk.dropna(subset=["경도", "위도"])
        sample_n  = min(MAP_SAMPLE_PER_FILE, len(has_coord))
        if sample_n > 0:
            map_frames.append(
                has_coord[["상호명","상권업종대분류명","상권업종중분류명","행정동명","경도","위도"]]
                .sample(sample_n, random_state=42)
            )

        # ③ 주변 상권 검색용: 동×대분류×중분류 점포 수
        dong_cnt = (chunk.groupby(key_cols + ["상권업종대분류명","상권업종중분류명","상권업종소분류명"])
                    .size().reset_index(name="점포수"))
        dong_frames.append(dong_cnt)

    csv_files = sorted(glob.glob(os.path.join(semas_dir, "semas_store_info_*.csv")))
    if csv_files:
        for fp in csv_files:
            try:
                _process(pd.read_csv(fp, encoding="utf-8", usecols=use_cols))
            except Exception:
                pass
    elif os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".csv"):
                        try:
                            _process(pd.read_csv(io.BytesIO(zf.read(name)),
                                                 encoding="utf-8", usecols=use_cols))
                        except Exception:
                            pass
        except zipfile.BadZipFile:
            # 깨진 ZIP → 삭제하고 빈 결과 반환 (탭 코드에서 재다운로드 유도)
            try:
                os.remove(zip_path)
            except Exception:
                pass
            return {}

    if not cnt_frames:
        return {}

    grp = key_cols + ["상권업종대분류명", "상권업종중분류명"]
    counts_df = (pd.concat(cnt_frames, ignore_index=True)
                 .groupby(grp)["점포수"].sum().reset_index())
    map_df    = pd.concat(map_frames, ignore_index=True) if map_frames else pd.DataFrame()
    dong_df   = (pd.concat(dong_frames, ignore_index=True)
                 .groupby(key_cols + ["상권업종대분류명","상권업종중분류명","상권업종소분류명"])
                 ["점포수"].sum().reset_index())

    return {"counts": counts_df, "map": map_df, "dong": dong_df}


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
        _hf_download("semas_store_count_mid.csv", path, "상권 집계 데이터")
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

# ── 매출 데이터 로드 (메인 화면 버튼으로 온디맨드 로드) ──
loaded_yyyymm = st.session_state.get("loaded_yyyymm")

if "df" not in st.session_state:
    # 빈 df로 시작 — 각 탭에서 조건 선택 시 온디맨드 다운로드
    st.session_state["df"]            = pd.DataFrame()
    st.session_state["loaded_yyyymm"] = AVAILABLE_YYYYMM[-1]
    st.session_state["sales_enc"]     = "-"
    st.session_state["sales_path"]    = "-"

df            = st.session_state["df"]
sales_enc     = st.session_state.get("sales_enc", "-")
sales_path    = st.session_state.get("sales_path", "-")
loaded_yyyymm = st.session_state["loaded_yyyymm"]


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
tab_pred, tab_hm, tab_lstm, tab_cluster, tab_ai, tab_fp, tab_semas, tab_ov = st.tabs([
    "💰 1인당 소비 예측",
    "⏰ 시간대·요일 분석",
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
    st.subheader("1인당 소비금액 예측")
    st.caption("조건을 모두 선택한 뒤 버튼을 누르면 해당 고객 유형의 1인당 예상 소비금액을 알려드립니다.")

    p1, p2 = st.columns(2)

    with p1:
        if admin_ok:
            _pred_dist_list = [d for d in admin_district_list if any(d.startswith(c) for c in CITY_KO_TO_EN)]
            sel_district  = st.selectbox("지역 (시/구)", ["전체"] + _pred_dist_list, key="pred_dist")
            dong_opts     = admin_district_to_dongs.get(sel_district, []) if sel_district != "전체" else []
            sel_admi_name = st.selectbox("동네 선택", ["전체"] + dong_opts, key="pred_dong")
            sel_admi      = admin_name_to_code.get(sel_admi_name, 0)
        else:
            sel_district = sel_admi_name = "전체"
            sel_admi = 0

        _avail_months = get_available_months_for_city(sel_district) if sel_district != "전체" else sorted({int(m[4:]) for m in AVAILABLE_YYYYMM})
        sel_month     = st.selectbox("월", ["전체"] + _avail_months,
                                     format_func=lambda x: f"{x}월" if x != "전체" else "전체",
                                     index=0,
                                     key="pred_month")
        sel_day_label = st.selectbox("요일", ["전체"] + list(DAY_MAP.values()), key="pred_day")
        sel_day       = {v: k for k, v in DAY_MAP.items()}.get(sel_day_label)

    with p2:
        sel_biz1  = st.selectbox("업종 대분류", ["전체"] + BIZ1_OPTS, key="pred_biz1")
        biz2_opts = BIZ2_MAP.get(sel_biz1, []) if sel_biz1 != "전체" else []
        sel_biz2  = st.selectbox("업종 중분류", ["전체"] + biz2_opts, key="pred_biz2")

        sel_sex_label = st.selectbox("성별", ["전체", "남성", "여성"], key="pred_sex")
        sel_sex       = {"남성": "M", "여성": "F"}.get(sel_sex_label)

        age_labels    = list(AGE_MAP.values())
        sel_age_label = st.selectbox("나이대", ["전체"] + age_labels, key="pred_age")
        sel_age       = {v: k for k, v in AGE_MAP.items()}.get(sel_age_label)

        hour_labels    = list(HOUR_MAP.values())
        sel_hour_label = st.selectbox("시간대", ["전체"] + hour_labels, key="pred_hour")
        sel_hour       = {v: k for k, v in HOUR_MAP.items()}.get(sel_hour_label)

    # 필수 조건 체크
    pred_required = (
        sel_district  != "전체" and
        sel_admi_name != "전체" and
        sel_month     != "전체" and
        sel_day_label != "전체" and
        sel_biz1      != "전체" and
        sel_biz2      != "전체" and
        sel_biz2      in (["전체"] + biz2_opts) and
        sel_sex_label != "전체" and
        sel_age_label != "전체" and
        sel_hour_label!= "전체"
    )

    # ── 데이터 사전 다운로드 (예측 버튼보다 먼저) ──────────────────────
    pred_data_ready = True
    if sel_district != "전체" and sel_month != "전체":
        if not _city_month_in_df(df, sel_district, sel_month):
            pred_data_ready = False
            st.info(
                f"📥 **{sel_district} {sel_month}월** 데이터가 필요합니다. "
                f"아래 버튼을 눌러 먼저 다운로드하세요. (파일 크기에 따라 1~3분 소요될 수 있습니다)"
            )
            if st.button("📥 데이터 다운로드", key="pred_download_btn"):
                with st.spinner(f"{sel_district} {sel_month}월 데이터 다운로드 중... 잠시만 기다려 주세요."):
                    ok = ensure_month_in_df(sel_month, city_korean=sel_district)
                if not ok:
                    st.error("데이터 다운로드에 실패했습니다. 다른 지역이나 월을 선택해주세요.")

    if st.button("1인당 소비금액 예측", type="primary", key="pred_btn",
                 disabled=not pred_data_ready):
        missing = []
        if sel_district  == "전체": missing.append("지역 (시/구)")
        if sel_admi_name == "전체": missing.append("동네")
        if sel_month     == "전체": missing.append("월")
        if sel_day_label == "전체": missing.append("요일")
        if sel_biz1      == "전체": missing.append("업종 대분류")
        if sel_biz2      == "전체": missing.append("업종 중분류")
        if sel_sex_label == "전체": missing.append("성별")
        if sel_age_label == "전체": missing.append("나이대")
        if sel_hour_label== "전체": missing.append("시간대")
        if missing:
            st.warning(f"⚠️ 다음 조건을 선택해주세요: **{', '.join(missing)}**")
        else:
            st.session_state["pred_run"] = True

    if pred_required and pred_data_ready and st.session_state.get("pred_run"):
        st.session_state["pred_run"] = False
        try:
            model, model_info = load_saved_model()

            # 선택 조건에 맞는 참조 데이터 필터링
            ref_exact = df[
                (df["card_tpbuz_nm_2"] == sel_biz2) &
                (df["day"]   == sel_day) &
                (df["month"] == sel_month) &
                (df["sex"]   == sel_sex) &
                (df["age"]   == sel_age) &
                (df["hour"]  == sel_hour)
            ]
            ref_fallback = len(ref_exact) < 5
            ref = ref_exact if not ref_fallback else df[df["card_tpbuz_nm_2"] == sel_biz2]

            avg_cnt = int(round(ref["cnt"].mean())) if "cnt" in ref.columns and len(ref) > 0 else 1
            sel_cnt = max(avg_cnt, 1)

            # 모델 입력 구성
            # amt = 그룹(날짜×시간대×성별×나이대×지역×업종) 총 매출합계
            # cnt = 그룹 내 거래 건수
            # 모델은 cnt를 입력받아 그룹 총 amt를 예측 → 1인당 = pred_amt / cnt
            input_row = pd.DataFrame([{
                "sex":             sel_sex,
                "age":             sel_age,
                "day":             sel_day,
                "hour":            sel_hour,
                "month":           sel_month,
                "admi_cty_no":     sel_admi,
                "card_tpbuz_nm_1": sel_biz1,
                "card_tpbuz_nm_2": sel_biz2,
                "cnt":             sel_cnt,
            }])
            encoded   = transform_with_saved_encoders(input_row)
            raw_pred  = model.predict(encoded)[0]
            pred_grp  = max(np.expm1(raw_pred) if model_info.get("use_log_target", True) else raw_pred, 0)
            pred_amt  = pred_grp / sel_cnt   # 1인(1건)당 소비금액

            # 실제 데이터 1인당 소비금액 = 각 행의 amt/cnt 평균·중앙값
            ref_per_person = ref["amt"] / ref["cnt"].clip(lower=1)
            actual_mean = ref_per_person.mean() if len(ref) > 0 else 0
            actual_med  = ref_per_person.median() if len(ref) > 0 else 0

            # 결과 표시
            st.info("ℹ️ **구매 고객 기준**: 카드 결제 데이터 특성상 실제 결제가 발생한 고객만 집계됩니다. 방문했으나 결제하지 않은 고객은 포함되지 않습니다.")
            st.success(
                f"### {sel_admi_name} · {sel_biz2} · {sel_month}월 {sel_day_label} {sel_hour_label} "
                f"· {sel_age_label} {sel_sex_label}  예측 1인당 소비금액: **{fmt(pred_amt)}**"
            )

            st.metric("예측 1인당 소비금액", fmt(pred_amt),
                      help=f"모델 예측 그룹 총매출({fmt(pred_grp)}) ÷ 평균 거래건수({sel_cnt}건)")


        except Exception as e:
            st.error(f"예측 오류: {e}")

# =====================================================
# ℹ️ 데이터 정보 (구 탭1)
# =====================================================
with tab_ov:
    st.subheader("데이터 출처")
    st.markdown("""
| 데이터 | 출처 | 링크 |
|--------|------|------|
| 경기도 카드 소비 데이터 (tbsh_gyeonggi_day) | 공공데이터포털 | [data.go.kr](https://www.data.go.kr) |
| 경기도 유동인구 데이터 (flowpop_admi) | 공공데이터포털 | [data.go.kr](https://www.data.go.kr) |
| 경기도 상권 정보 (SEMAS) | 소상공인시장진흥공단 상권정보시스템 | [sg.sbiz.or.kr](https://sg.sbiz.or.kr) |
| 행정동 코드 (city_admin_code) | 행정안전부 행정표준코드관리시스템 | [code.go.kr](https://www.code.go.kr) |
""")
    st.caption("※ 본 서비스는 공공데이터 활용 목적으로 제작되었으며, 데이터 원본의 저작권은 각 제공 기관에 있습니다.")
    st.divider()

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
    c3.metric("총 매출액",      f"{df['amt'].sum():,.0f}원" if "amt" in df.columns else "-")
    c4.metric("행정동 수",      f"{len(admin_name_to_code)}개")

    if not admin_ok:
        st.warning("⚠️ city_admin_code.csv 파일이 없어 데이터의 admi_cty_no 값을 그대로 사용합니다.")

    st.subheader("AI 모델 정보")
    enc_map = {c: "Label Encoding" for c in NUM_COLS}
    enc_map.update({c: "Label Encoding (LightGBM 카테고리)" for c in CAT_COLS})
    st.dataframe(pd.DataFrame({
        "변수명": MODEL_FEATURES,
        "인코딩 방식": [enc_map.get(c, "Label Encoding") for c in MODEL_FEATURES],
    }), use_container_width=True)

    st.subheader("행정동 목록")
    st.write(f"행정동 코드 CSV 로드 개수: {len(admin_df)}개  /  행정동 옵션 개수: {len(admin_name_to_code)}개")
    st.dataframe(admin_df, use_container_width=True)


    st.subheader("원본 데이터 미리보기")
    if df is None or df.empty:
        st.info("사이드바에서 지역과 월을 선택하면 카드 소비 데이터가 표시됩니다.")
    else:
        st.dataframe(df.head(30), use_container_width=True)

# =====================================================
# 📊 소비 트렌드 (구 탭2)
# =====================================================
# =====================================================
# ⏰ 시간대·요일 분석
# =====================================================
with tab_hm:
    st.subheader("언제 매출이 높을까요?")
    st.caption("조건을 모두 선택하면 해당 지역·업종의 시간대·요일 분석 차트를 보여줍니다.")

    # ── 필터 ──
    hm1, hm2 = st.columns(2)
    with hm1:
        if admin_ok:
            hm_district  = st.selectbox("지역 (시/구)", ["전체"] + admin_district_list, key="hm_dist")
            hm_dong_opts = ["전체"] + admin_district_to_dongs.get(hm_district, []) if hm_district != "전체" else ["전체"]
            hm_admi_name = st.selectbox("동네 선택", hm_dong_opts, key="hm_dong")
        else:
            hm_district = hm_admi_name = "전체"
        _hm_avail = get_available_months_for_city(hm_district) if hm_district != "전체" else sorted({int(m[4:]) for m in AVAILABLE_YYYYMM})
        _hm_month_opts = ["전체"] + [f"{m}월" for m in _hm_avail]
        hm_month = st.selectbox("월", _hm_month_opts, key="hm_month", index=0)
    with hm2:
        hm_biz1      = st.selectbox("업종 대분류", ["전체"] + BIZ1_OPTS, key="hm_biz1")
        hm_biz2_opts = ["전체"] + BIZ2_MAP.get(hm_biz1, []) if hm_biz1 != "전체" else ["전체"]
        hm_biz2      = st.selectbox("업종 중분류", hm_biz2_opts, key="hm_biz2")

    # ── 필수 조건 체크 ──
    hm_required = (
        hm_district  != "전체" and
        hm_admi_name != "전체" and
        hm_month     != "전체" and
        hm_biz1      != "전체" and
        hm_biz2      != "전체" and
        hm_biz2      in hm_biz2_opts
    )

    if not hm_required:
        st.info("📌 지역(시/구), 동네, 월, 업종 대분류, 업종 중분류를 모두 선택하면 차트가 표시됩니다.")
    else:
        # ── 먼저 데이터 다운로드 확인 (필터보다 앞에) ──
        hm_m = int(hm_month.replace("월", ""))
        if not _city_month_in_df(df, hm_district, hm_m):
            ok = ensure_month_in_df(hm_m, city_korean=hm_district if hm_district != "전체" else None)
            if not ok:
                st.error("데이터 다운로드에 실패했습니다. 다른 조건을 선택해주세요.")
            st.stop()

        # ── 필터 적용 ──
        hm_df = df.copy()
        if admin_ok:
            hm_code = admin_name_to_code.get(hm_admi_name)
            if hm_code:
                hm_df = hm_df[hm_df["admi_cty_no"].astype(int) == int(hm_code)]
        hm_df = hm_df[hm_df["card_tpbuz_nm_1"] == hm_biz1]
        hm_df = hm_df[hm_df["card_tpbuz_nm_2"] == hm_biz2]
        hm_df = hm_df[hm_df["month"] == hm_m]

        if hm_df.empty:
            # 어느 필터에서 빈 결과가 됐는지 진단
            _diag = df.copy()
            _msgs = []
            if admin_ok and admin_name_to_code.get(hm_admi_name):
                _diag = _diag[_diag["admi_cty_no"].astype(int) == int(admin_name_to_code[hm_admi_name])]
                _msgs.append(f"{hm_admi_name}: {len(_diag):,}건")
            if "card_tpbuz_nm_1" in _diag.columns:
                _diag = _diag[_diag["card_tpbuz_nm_1"] == hm_biz1]
                _msgs.append(f"{hm_biz1}: {len(_diag):,}건")
            if "card_tpbuz_nm_2" in _diag.columns:
                _diag = _diag[_diag["card_tpbuz_nm_2"] == hm_biz2]
                _msgs.append(f"{hm_biz2}: {len(_diag):,}건")
            if "month" in _diag.columns:
                _diag = _diag[_diag["month"] == hm_m]
                _msgs.append(f"{hm_m}월: {len(_diag):,}건")
            st.warning("선택 조건에 해당하는 데이터가 없습니다.\n\n" + " → ".join(_msgs))
        else:
            st.caption(f"분석 데이터: {len(hm_df):,}건")

            # ── 차트 1: 시간대별 매출 ──
            st.markdown("#### ⏰ 시간대별 매출")
            hour_grp = hm_df.groupby("hour")["amt"].sum().reset_index()
            hour_grp["시간대"] = hour_grp["hour"].map(HOUR_MAP)
            fig_hour = go.Figure(go.Bar(
                x=hour_grp["시간대"], y=hour_grp["amt"],
                marker=dict(color=hour_grp["amt"], colorscale="Blues", showscale=False),
                hovertemplate="%{x}: %{y:,.0f}원<extra></extra>",
            ))
            fig_hour.update_layout(xaxis_title="시간대", yaxis_title="매출액 (원)",
                                   height=320, margin=dict(t=10, b=40),
                                   xaxis_tickangle=-30)
            st.plotly_chart(fig_hour, use_container_width=True)

            # ── 차트 2: 요일별 매출 ──
            st.markdown("#### 📅 요일별 매출")
            dow_order = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
            day_grp = hm_df.groupby("day")["amt"].sum().reset_index()
            day_grp["요일"] = day_grp["day"].map(DAY_MAP)
            day_grp["요일"] = pd.Categorical(day_grp["요일"], categories=dow_order, ordered=True)
            day_grp = day_grp.sort_values("요일")
            fig_day = go.Figure(go.Bar(
                x=day_grp["요일"], y=day_grp["amt"],
                marker=dict(color=day_grp["amt"], colorscale="Oranges", showscale=False),
                hovertemplate="%{x}: %{y:,.0f}원<extra></extra>",
            ))
            fig_day.update_layout(xaxis_title="요일", yaxis_title="매출액 (원)",
                                  height=320, margin=dict(t=10, b=40))
            st.plotly_chart(fig_day, use_container_width=True)

            # ── 차트 3: 요일 × 시간대 히트맵 ──
            st.markdown("#### 🔥 요일 × 시간대 매출 히트맵")
            dh_pivot = (hm_df.groupby(["day","hour"])["amt"].sum()
                        .unstack(fill_value=0))
            dh_pivot.index = [DAY_MAP.get(d, d) for d in dh_pivot.index]
            dh_pivot.columns = [HOUR_MAP.get(h, str(h)) for h in dh_pivot.columns]
            _dow_order_full = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
            dh_pivot = dh_pivot.reindex([d for d in _dow_order_full if d in dh_pivot.index])
            fig_dh = go.Figure(go.Heatmap(
                z=dh_pivot.values,
                x=dh_pivot.columns.tolist(),
                y=dh_pivot.index.tolist(),
                colorscale="Blues",
                hovertemplate="요일: %{y}<br>시간: %{x}<br>매출: %{z:,.0f}원<extra></extra>",
            ))
            fig_dh.update_layout(xaxis_title="시간대", yaxis_title="요일",
                                 height=340, margin=dict(t=10, b=40),
                                 xaxis_tickangle=-30)
            st.plotly_chart(fig_dh, use_container_width=True)

            # ── 차트 4: 연령대별 매출 ──
            st.markdown("#### 👤 연령대별 매출")
            age_grp = hm_df.groupby("age")["amt"].sum().reset_index()
            age_grp["연령대"] = age_grp["age"].map(AGE_MAP)
            fig_age = go.Figure(go.Bar(
                x=age_grp["연령대"], y=age_grp["amt"],
                marker=dict(color=age_grp["amt"], colorscale="Purples", showscale=False),
                hovertemplate="%{x}: %{y:,.0f}원<extra></extra>",
            ))
            fig_age.update_layout(xaxis_title="연령대", yaxis_title="매출액 (원)",
                                  height=320, margin=dict(t=10, b=40))
            st.plotly_chart(fig_age, use_container_width=True)

            # ── 차트 5: 성별 매출 비중 ──
            st.markdown("#### 🚻 성별 매출 비중")
            sex_grp = hm_df.groupby("sex")["amt"].sum().reset_index()
            sex_grp["성별"] = sex_grp["sex"].map(SEX_MAP)
            fig_sex = go.Figure(go.Pie(
                labels=sex_grp["성별"], values=sex_grp["amt"],
                marker_colors=["#60a5fa","#f472b6"],
                hole=0.4,
                hovertemplate="%{label}: %{value:,.0f}원 (%{percent})<extra></extra>",
            ))
            fig_sex.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig_sex, use_container_width=True)

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
        lt_biz1      = st.selectbox("업종 대분류", ["전체"] + BIZ1_OPTS, key="lt_biz1")
        lt_biz2_opts = ["전체"] + BIZ2_MAP.get(lt_biz1, []) if lt_biz1 != "전체" else ["전체"]
        lt_biz2      = st.selectbox("업종 중분류", lt_biz2_opts, key="lt_biz2")

    # ── 필수 조건 체크 ──
    lt_required = (
        lt_district  != "전체" and
        lt_admi_name != "전체" and
        lt_biz1      != "전체" and
        lt_biz2      != "전체" and
        lt_biz2      in lt_biz2_opts
    )

    # ── 날짜 범위 필터 (항상 표시) ──────────────────────────────
    _lt_all_yyyymm = sorted(
        CITY_AVAILABLE_MONTHS.get(CITY_KO_TO_EN.get(
            next((c for c in CITY_KO_TO_EN if lt_district.startswith(c)), ""), ""), [])
        or AVAILABLE_YYYYMM
    )
    _lt_date_min = pd.Timestamp(f"{_lt_all_yyyymm[0][:4]}-{_lt_all_yyyymm[0][4:]}-01").date()
    import calendar
    _last_ym = _lt_all_yyyymm[-1]
    _lt_date_max = pd.Timestamp(
        f"{_last_ym[:4]}-{_last_ym[4:]}-{calendar.monthrange(int(_last_ym[:4]), int(_last_ym[4:]))[1]}"
    ).date()

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        lt_start_date = st.date_input("시작일자", value=_lt_date_min,
                                      min_value=_lt_date_min, max_value=_lt_date_max,
                                      key="lt_start_date")
    with dc2:
        import datetime as _dt
        _lt_end_max = min(
            lt_start_date + _dt.timedelta(days=61),  # 시작일 기준 최대 2개월(62일)
            _lt_date_max
        )
        _lt_end_default = min(
            lt_start_date + _dt.timedelta(days=30),
            _lt_end_max
        )
        lt_end_date = st.date_input(
            "종료일자 (시작일 기준 최대 2개월)",
            value=_lt_end_default,
            min_value=lt_start_date,
            max_value=_lt_end_max,
            key="lt_end_date"
        )
    with dc3:
        FORECAST_DAYS = st.slider("미래 예측 기간 (일)", 7, 60, 30, key="lt_forecast_days")

    st.caption(
        f"📌 **조회 기간**: {lt_start_date} ~ {lt_end_date}  "
        f"({(lt_end_date - lt_start_date).days + 1}일)  |  "
        f"아래 차트는 **{lt_admi_name if lt_admi_name != '전체' else lt_district} · {lt_biz2 if lt_biz2 != '전체' else (lt_biz1 if lt_biz1 != '전체' else '전체 업종')}** "
        f"중분류에 해당하는 **모든 업장의 카드 매출 합산** 금액입니다."
    )

    if not lt_required:
        st.info("📌 지역(시/구), 동네, 업종 대분류, 업종 중분류를 모두 선택하면 차트가 표시됩니다.")
    else:
        # df 비어있으면 해당 도시의 최신 월 데이터 다운로드
        if not _city_month_in_df(df, lt_district):
            _city_months = get_available_months_for_city(lt_district) if lt_district != "전체" else sorted({int(m[4:]) for m in AVAILABLE_YYYYMM})
            _latest_m = _city_months[-1] if _city_months else int(sorted(AVAILABLE_YYYYMM)[-1][4:])
            ok = ensure_month_in_df(_latest_m, city_korean=lt_district if lt_district != "전체" else None)
            if not ok:
                st.error("데이터 다운로드에 실패했습니다. 다른 조건을 선택해주세요.")
            st.stop()

        # ── 필터 적용 ──
        lt_df = df.copy()
        if admin_ok:
            lt_admi_code = admin_name_to_code.get(lt_admi_name)
            if lt_admi_code:
                lt_df = lt_df[lt_df["admi_cty_no"].astype(int) == int(lt_admi_code)]
        lt_df = lt_df[lt_df["card_tpbuz_nm_1"] == lt_biz1]
        lt_df = lt_df[lt_df["card_tpbuz_nm_2"] == lt_biz2]

        if lt_df.empty:
            st.warning("선택 조건에 해당하는 데이터가 없습니다.")
        else:
            # 날짜 컬럼 찾기
            date_col = next((c for c in ["ta_ymd", "date"] if c in lt_df.columns), None)
            if date_col is None:
                st.warning("날짜 컬럼이 없어 시계열 차트를 표시할 수 없습니다.")
                st.stop()
            lt_df["_date"] = pd.to_datetime(lt_df[date_col], errors="coerce")
            lt_df = lt_df.dropna(subset=["_date"])
            if lt_df.empty:
                st.warning("날짜 파싱 후 유효한 데이터가 없습니다.")
                st.stop()
            all_daily = lt_df.groupby("_date")["amt"].sum().reset_index().sort_values("_date")
            all_daily = all_daily.rename(columns={"_date": "date"})

            if all_daily.empty:
                st.warning("선택 조건에 해당하는 날짜별 데이터가 없습니다.")
                st.stop()

            # 위에서 선택한 날짜 범위 적용
            daily = all_daily[
                (all_daily["date"] >= pd.Timestamp(lt_start_date)) &
                (all_daily["date"] <= pd.Timestamp(lt_end_date))
            ]
            if daily.empty:
                st.warning("선택한 날짜 범위에 해당하는 데이터가 없습니다. 날짜 범위를 넓혀보세요.")
                st.stop()

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
                        c1.metric("예측 평균 일매출 (전체 합산)", fmt(pred_vals.mean()))
                        c2.metric("예측 최고 매출일 (전체 합산)", fmt(pred_vals.max()))
                        c3.metric("예측 최저 매출일 (전체 합산)", fmt(pred_vals.min()))
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
            st.plotly_chart(fig, use_container_width=True)

            d1, d2 = st.columns(2)
            d1.metric("기간 평균 일매출 (전체 합산)", fmt(daily["amt"].mean()))
            d2.metric("기간 총 매출 (전체 합산)",     fmt(daily["amt"].sum()))

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
        _cl_avail = get_available_months_for_city(cl_district) if cl_district != "전체" else sorted({int(m[4:]) for m in AVAILABLE_YYYYMM})
        _cl_month_opts = ["전체"] + [f"{m}월" for m in _cl_avail]
        cl_month = st.selectbox("월", _cl_month_opts, key="cl_month", index=0)
        all_days  = list(DAY_MAP.values())
        cl_days   = st.multiselect("요일 (전체 선택 = 모든 요일)", all_days, default=all_days, key="cl_days")
    with cl2:
        cl_biz1      = st.selectbox("업종 대분류", ["전체"] + BIZ1_OPTS, key="cl_biz1")
        cl_biz2_opts = ["전체"] + BIZ2_MAP.get(cl_biz1, []) if cl_biz1 != "전체" else ["전체"]
        cl_biz2      = st.selectbox("업종 중분류", cl_biz2_opts, key="cl_biz2")

    # 지역 선택 시 필요한 월 데이터 확인 (필터보다 앞에)
    if cl_district != "전체":
        if cl_month != "전체":
            _cl_m = int(cl_month.replace("월", ""))
            if not _city_month_in_df(df, cl_district, _cl_m):
                ensure_month_in_df(_cl_m, city_korean=cl_district)
                st.stop()
        elif not _city_month_in_df(df, cl_district):
            _latest_m = int(sorted(AVAILABLE_YYYYMM)[-1][4:])
            ensure_month_in_df(_latest_m, city_korean=cl_district)
            st.stop()

    # ── 필터 적용 ──
    cl_df = df.copy()
    if cl_district != "전체" and cl_admi_name != "전체" and admin_ok:
        cl_code = admin_name_to_code.get(cl_admi_name)
        if cl_code:
            cl_df = cl_df[cl_df["admi_cty_no"].astype(int) == int(cl_code)]
    elif cl_district != "전체" and admin_ok:
        cl_codes = {admin_name_to_code[d] for d in admin_district_to_dongs.get(cl_district, []) if d in admin_name_to_code}
        cl_df = cl_df[cl_df["admi_cty_no"].isin(cl_codes)]
    if cl_biz1 != "전체":
        cl_df = cl_df[cl_df["card_tpbuz_nm_1"] == cl_biz1]
    if cl_biz2 != "전체":
        cl_df = cl_df[cl_df["card_tpbuz_nm_2"] == cl_biz2]
    if cl_month != "전체":
        cl_m = int(cl_month.replace("월", ""))
        if not _city_month_in_df(df, cl_district, cl_m) if cl_district != "전체" else ("month" not in df.columns or cl_m not in df["month"].values):
            ensure_month_in_df(cl_m, city_korean=cl_district if cl_district != "전체" else None)
            st.stop()
        cl_df = cl_df[cl_df["month"] == cl_m]
    if cl_days and len(cl_days) < len(all_days):
        day_rev = {v: k for k, v in DAY_MAP.items()}
        cl_day_nums = [day_rev[d] for d in cl_days]
        cl_df = cl_df[cl_df["day"].isin(cl_day_nums)]

    # 필수 조건: 지역(시/구), 동네, 업종 대분류, 업종 중분류 모두 선택해야 차트 표시
    required_selected = (
        cl_district != "전체" and
        cl_admi_name != "전체" and
        cl_biz1 != "전체" and
        cl_biz2 != "전체" and
        cl_biz2 in cl_biz2_opts
    )

    if not required_selected:
        st.info("📌 지역(시/구), 동네, 업종 대분류, 업종 중분류를 모두 선택하면 분석 결과가 표시됩니다.")
    elif cl_df.empty:
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



# =====================================================
# 📝 AI 리포트 (OpenAI API)
# =====================================================
with tab_ai:
    st.subheader("AI가 분석한 우리 동네 소비 리포트")
    st.caption("데이터를 요약해 GPT-4o가 사장님을 위한 인사이트 리포트를 작성합니다.")

    # API 키: secrets 우선, 없으면 입력창
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        api_key = ""
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
        ai_biz1      = st.selectbox("업종 대분류", ["전체"] + BIZ1_OPTS, key="ai_biz1")
        ai_biz2_opts = ["전체"] + BIZ2_MAP.get(ai_biz1, []) if ai_biz1 != "전체" else ["전체"]
        ai_biz2      = st.selectbox("업종 중분류", ai_biz2_opts, key="ai_biz2")

    # 지역 선택 시 해당 도시 데이터 자동 로드
    if ai_district != "전체":
        _ai_city_prefixes = [k for k, v in DISTRICT_MAP.items() if v == ai_district]
        _ai_city_in_df = (
            not df.empty and "admi_cty_no" in df.columns and
            df["admi_cty_no"].astype(str).str[:5].isin(_ai_city_prefixes).any()
        )
        if not _ai_city_in_df:
            _ai_city_months = get_available_months_for_city(ai_district)
            _ai_latest_m = _ai_city_months[-1] if _ai_city_months else int(sorted(AVAILABLE_YYYYMM)[-1][4:])
            with st.spinner(f"{ai_district} 데이터 로드 중..."):
                ensure_month_in_df(_ai_latest_m, city_korean=ai_district)
            st.stop()

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
                            filtered = filtered[filtered["admi_cty_no"].astype(int) == int(ai_code)]
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
                        _hour_s = filtered.groupby("hour")["amt"].sum()
                        _day_s  = filtered.groupby("day")["amt"].sum()
                        _age_s  = filtered.groupby("age")["amt"].sum()
                        top_hour  = _hour_s.idxmax() if not _hour_s.empty else None
                        top_day   = _day_s.idxmax()  if not _day_s.empty  else None
                        top_age   = _age_s.idxmax()  if not _age_s.empty  else None
                        total_amt = filtered["amt"].sum()
                        avg_amt   = filtered["amt"].mean()

                        hour_label = HOUR_MAP.get(top_hour, str(top_hour)) if top_hour is not None else "알 수 없음"
                        day_label  = DAY_MAP.get(top_day, str(top_day))   if top_day  is not None else "알 수 없음"
                        age_label  = AGE_MAP.get(top_age, str(top_age))   if top_age  is not None else "알 수 없음"

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

    # ── 공통 필터를 먼저 렌더링 (다운로드 버튼 전에 지역 선택 가능하게)
    _fp_areas_preview = ["전체"] + sorted(CITY_KO_TO_EN.keys())
    fc1, fc2 = st.columns(2)
    with fc1:
        fp_sel_city_pre = st.selectbox("시/구 선택", _fp_areas_preview, key="fp_city")
    fp_sel_dong = "전체"  # default; overwritten below when data is loaded
    fp_forn = st.radio("내/외국인 구분", ["전체", "내국인만", "외국인만"],
                       horizontal=True, key="fp_forn")

    # ── 데이터 로드 (지역 선택 시 전체 기간, 전체면 사이드바 월)
    ALL_FP_YYYYMM = sorted(list(FLOWPOP_MONTHLY_IDS.keys()) + list(FLOWPOP_COMBINED_YYYYMM))

    if fp_sel_city_pre != "전체":
        fp_cache_key = f"fp_data_all_{fp_sel_city_pre}"
    else:
        fp_yyyymm = st.session_state.get("loaded_yyyymm", AVAILABLE_YYYYMM[-1])
        fp_cache_key = f"fp_data_{fp_yyyymm}"

    if fp_cache_key not in st.session_state:
        if fp_sel_city_pre != "전체":
            # 전체 기간 다운로드 필요한 월 목록
            missing = [m for m in ALL_FP_YYYYMM if not os.path.exists(get_flowpop_zip_path(m))]
            if missing:
                labels = ", ".join(YYYYMM_LABEL.get(m, m) for m in missing)
                st.info(f"📥 {fp_sel_city_pre} 전체 기간 분석을 위해 유동인구 데이터를 다운로드합니다.\n\n미다운로드: **{labels}**")
                if st.button("📥 전체 기간 유동인구 데이터 다운로드", key="fp_download_all"):
                    prog = st.progress(0, text="다운로드 준비 중...")
                    for i, m in enumerate(missing):
                        prog.progress((i) / len(missing), text=f"{YYYYMM_LABEL.get(m,m)} 다운로드 중...")
                        ensure_flowpop_zip(m)
                    prog.progress(1.0, text="다운로드 완료!")
                    st.rerun()
            else:
                with st.spinner("전체 기간 유동인구 데이터 집계 중..."):
                    frames_hm, frames_age, frames_daily = [], [], []
                    for m in ALL_FP_YYYYMM:
                        zp = get_flowpop_zip_path(m)
                        if os.path.exists(zp):
                            d = load_flowpop_data(zp)
                            if d:
                                frames_hm.append(d["heatmap"])
                                frames_age.append(d["age"])
                                frames_daily.append(d["daily"])
                    if frames_hm:
                        grp = ["CTY_NM", "ADMI_NM", "ADMI_CD", "FORN_GB"]
                        st.session_state[fp_cache_key] = {
                            "heatmap": pd.concat(frames_hm).groupby(grp + ["TIME_CD", "DOW"])["TOTAL_CNT"].mean().reset_index(),
                            "age":     pd.concat(frames_age).groupby(grp).sum().reset_index(),
                            "daily":   pd.concat(frames_daily).groupby(["ETL_YMD", "ADMI_CD"])["TOTAL_CNT"].sum().reset_index(),
                        }
        else:
            fp_zip_path = get_flowpop_zip_path(fp_yyyymm)
            fp_label = YYYYMM_LABEL.get(fp_yyyymm, fp_yyyymm)
            if not os.path.exists(fp_zip_path):
                if st.button(f"📥 {fp_label} 유동인구 데이터 다운로드", key="fp_download"):
                    if ensure_flowpop_zip(fp_yyyymm):
                        st.rerun()
            if os.path.exists(fp_zip_path):
                with st.spinner(f"{fp_label} 유동인구 데이터 집계 중..."):
                    st.session_state[fp_cache_key] = load_flowpop_data(fp_zip_path)

    fp_data = st.session_state.get(fp_cache_key)

    if not fp_data:
        st.info("📥 위 버튼을 눌러 유동인구 데이터를 다운로드하세요.")
    else:
        fp_hm    = fp_data["heatmap"]
        fp_age   = fp_data["age"]
        fp_daily = fp_data["daily"]

        # 행정동 드롭다운 업데이트
        if fp_sel_city_pre != "전체":
            dong_pool = sorted(fp_hm[fp_hm["CTY_NM"] == fp_sel_city_pre]["ADMI_NM"].dropna().unique())
        else:
            dong_pool = sorted(fp_hm["ADMI_NM"].dropna().unique())

        # 행정동 드롭다운을 데이터 로드 후 갱신
        with fc2:
            fp_sel_dong = st.selectbox("행정동 선택", ["전체"] + dong_pool, key="fp_dong")

        fp_sel_city = fp_sel_city_pre

        def _fp_filter(tbl):
            t = tbl
            if fp_sel_city != "전체":
                t = t[t["CTY_NM"] == fp_sel_city]
            if fp_sel_dong != "전체":
                t = t[t["ADMI_NM"] == fp_sel_dong]
            if fp_forn == "내국인만":
                t = t[t["FORN_GB"] == "N"]
            elif fp_forn == "외국인만":
                t = t[t["FORN_GB"] == "F"]
            return t

        st.divider()

        if fp_sel_city == "전체" or fp_sel_dong == "전체":
            st.info("📌 시/구와 행정동을 모두 선택하면 분석 차트가 표시됩니다.")
            st.stop()

        # ══════════════════════════════════════════════════
        # [기능 2] 시간대 × 요일 히트맵
        # ══════════════════════════════════════════════════
        st.markdown("#### ⏰ 시간대 × 요일 유동인구 히트맵")
        st.caption("어느 요일·시간대에 유동인구가 집중되는지 보여줍니다.")

        dow_label = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
        dow_order = ["월","화","수","목","금","토","일"]
        hm_filtered = _fp_filter(fp_hm)
        if hm_filtered.empty:
            st.info("선택 조건에 해당하는 유동인구 데이터가 없습니다.")
        else:
            hm_filtered = hm_filtered.copy()
            hm_filtered["DOW_LABEL"] = hm_filtered["DOW"].map(dow_label)
            hm_data = (hm_filtered.groupby(["DOW_LABEL","TIME_CD"])["TOTAL_CNT"]
                       .mean().reset_index())
            hm_pivot = (hm_data.pivot(index="DOW_LABEL", columns="TIME_CD", values="TOTAL_CNT")
                        .reindex(dow_order))

            fig_hm = go.Figure(go.Heatmap(
                z=hm_pivot.values,
                x=[f"{h}시" for h in hm_pivot.columns],
                y=hm_pivot.index,
                colorscale="Blues",
                hovertemplate="요일: %{y}<br>시간: %{x}<br>평균 유동인구: %{z:,.0f}명<extra></extra>",
            ))
            fig_hm.update_layout(xaxis_title="시간대", yaxis_title="요일",
                                 height=340, margin=dict(t=20, b=40))
            st.plotly_chart(fig_hm, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════
        # [기능 3] 성별 × 연령대 분포
        # ══════════════════════════════════════════════════
        st.markdown("#### 👥 성별 × 연령대 유동인구 분포")
        st.caption("이 지역을 오가는 주요 고객층의 성별·연령대 비율을 확인합니다.")

        from collections import defaultdict
        age_filtered = _fp_filter(fp_age)
        am = [c for c in AGE_COLS_M if c in age_filtered.columns]
        af = [c for c in AGE_COLS_F if c in age_filtered.columns]
        male_agg, female_agg = defaultdict(float), defaultdict(float)
        if not age_filtered.empty:
            for c in am:
                male_agg[_age_label(c)] += age_filtered[c].sum()
            for c in af:
                female_agg[_age_label(c)] += age_filtered[c].sum()

        age_order = ["10대 미만","10대","20대","30대","40대","50대","60대","70대 이상"]
        fig_age = go.Figure()
        fig_age.add_bar(name="남성", x=age_order,
                        y=[male_agg.get(a,0) for a in age_order],
                        marker_color="#58a6ff",
                        hovertemplate="%{x}: %{y:,.0f}명<extra>남성</extra>")
        fig_age.add_bar(name="여성", x=age_order,
                        y=[female_agg.get(a,0) for a in age_order],
                        marker_color="#f78166",
                        hovertemplate="%{x}: %{y:,.0f}명<extra>여성</extra>")
        fig_age.update_layout(barmode="group", xaxis_title="연령대",
                              yaxis_title="유동인구 합계 (명)", height=340,
                              margin=dict(t=20, b=40),
                              legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_age, use_container_width=True)

        total_m = sum(male_agg.values())
        total_f = sum(female_agg.values())
        total_all = total_m + total_f
        if total_all > 0:
            a1, a2, a3 = st.columns(3)
            a1.metric("남성 비율", f"{total_m/total_all*100:.1f}%")
            a2.metric("여성 비율", f"{total_f/total_all*100:.1f}%")
            a3.metric("최다 연령대",
                      f"남 {max(male_agg,key=male_agg.get)} / 여 {max(female_agg,key=female_agg.get)}")

        st.divider()

        # ══════════════════════════════════════════════════
        # [기능 4] 내국인 vs 외국인 비율
        # ══════════════════════════════════════════════════
        st.markdown("#### 🌏 내국인 vs 외국인 유동인구 비율")
        st.caption("외국인 유동인구 비율이 높은 지역은 관광·외국인 특화 상권일 가능성이 높습니다.")

        forn_base = fp_age.copy()
        if fp_sel_city != "전체":
            forn_base = forn_base[forn_base["CTY_NM"] == fp_sel_city]
        if fp_sel_dong != "전체":
            forn_base = forn_base[forn_base["ADMI_NM"] == fp_sel_dong]

        forn_agg = (forn_base.groupby("FORN_GB")["TOTAL_CNT"].sum()
                    .rename({"N":"내국인","F":"외국인"}))

        if not forn_agg.empty and forn_agg.sum() > 0:
            fig_forn = go.Figure(go.Pie(
                labels=forn_agg.index, values=forn_agg.values,
                marker_colors=["#58a6ff","#f78166"], hole=0.45,
                hovertemplate="%{label}: %{value:,.0f}명 (%{percent})<extra></extra>",
            ))
            fig_forn.update_layout(height=320, margin=dict(t=20,b=20))
            forn_col1, forn_col2 = st.columns(2)
            with forn_col1:
                st.plotly_chart(fig_forn, use_container_width=True)
            with forn_col2:
                forn_total = forn_agg.sum()
                for label, val in forn_agg.items():
                    st.metric(f"{label} 유동인구", f"{val:,.0f}명",
                              f"{val/forn_total*100:.1f}%")

            if fp_sel_dong == "전체":
                st.markdown("##### 외국인 비율 상위 행정동")
                scope = fp_age.copy()
                if fp_sel_city != "전체":
                    scope = scope[scope["CTY_NM"] == fp_sel_city]
                forn_dong = (scope.groupby(["ADMI_NM","FORN_GB"])["TOTAL_CNT"]
                             .sum().unstack(fill_value=0))
                if "F" in forn_dong.columns and "N" in forn_dong.columns:
                    forn_dong["외국인비율(%)"] = (
                        forn_dong["F"]/(forn_dong["F"]+forn_dong["N"])*100).round(2)
                    top_forn = (forn_dong.sort_values("외국인비율(%)", ascending=False)
                                .head(10).reset_index())
                    top_forn.columns = ["행정동","외국인","내국인","외국인비율(%)"]
                    st.dataframe(top_forn[["행정동","내국인","외국인","외국인비율(%)"]],
                                 use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════

# =====================================================
# 🏪 상권 분석 (SEMAS 소상공인 상가 데이터)
# =====================================================
with tab_semas:
    st.subheader("전국 상권 분석")
    st.caption("소상공인시장진흥공단 상가(상권)정보를 기반으로 업종 분포·경쟁 강도·입지 추천·주변 상권을 분석합니다.")

    semas_df = pd.DataFrame()
    # 깨진 ZIP 자동 제거 (st.cache_data 캐시도 함께 클리어)
    if os.path.exists(SEMAS_ZIP_PATH) and not _is_valid_zip(SEMAS_ZIP_PATH):
        os.remove(SEMAS_ZIP_PATH)
        st.session_state.pop("semas_data", None)
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.warning("상권 데이터 파일이 손상되어 삭제했습니다. 아래 버튼으로 다시 다운로드해주세요.")

    zip_exists = os.path.exists(SEMAS_ZIP_PATH)
    csv_exists = bool(glob.glob(os.path.join(SEMAS_DIR, "semas_store_info_*.csv")))
    if not zip_exists and not csv_exists:
        if st.button("📥 상권 데이터 다운로드 (최초 1회)", key="semas_download"):
            if _hf_download(SEMAS_ZIP_NAME, SEMAS_ZIP_PATH, "상권 데이터 ZIP (약 240MB)"):
                st.rerun()
            else:
                st.error("상권 데이터 다운로드에 실패했습니다. 잠시 후 다시 시도해주세요.")

    if zip_exists or csv_exists:
        if "semas_data" not in st.session_state:
            with st.spinner("상권 데이터 집계 중... (최초 1회, 잠시 기다려 주세요)"):
                try:
                    st.session_state["semas_data"] = load_semas_data(SEMAS_DIR, SEMAS_ZIP_PATH)
                except Exception as e:
                    # ZIP이 깨진 경우 삭제 후 재다운로드 유도
                    if os.path.exists(SEMAS_ZIP_PATH):
                        os.remove(SEMAS_ZIP_PATH)
                    st.error(f"상권 데이터 로드 실패 (파일 손상): {e}\n\n페이지를 새로고침하면 재다운로드 버튼이 표시됩니다.")
                    st.stop()
        semas_data = st.session_state.get("semas_data", {})
    else:
        semas_data = {}

    if semas_data:
        sm_counts = semas_data["counts"]   # 시도명, 시군구명, 행정동명, 상권업종대분류명, 상권업종중분류명, 점포수
        sm_map    = semas_data["map"]       # 상호명, 상권업종대분류명, 상권업종중분류명, 행정동명, 경도, 위도
        sm_dong   = semas_data["dong"]      # + 상권업종소분류명

        # ── 공통 필터 ─────────────────────────────────────────
        sido_list = sorted(sm_counts["시도명"].dropna().unique())
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sel_sido = st.selectbox("시/도 선택", ["전체"] + sido_list, key="sm_sido")
        with sc2:
            if sel_sido != "전체":
                gu_pool = sorted(sm_counts[sm_counts["시도명"] == sel_sido]["시군구명"].dropna().unique())
            else:
                gu_pool = sorted(sm_counts["시군구명"].dropna().unique())
            sel_gu = st.selectbox("시/군/구 선택", ["전체"] + gu_pool, key="sm_gu")
        with sc3:
            if sel_gu != "전체":
                dong_pool = sorted(sm_counts[sm_counts["시군구명"] == sel_gu]["행정동명"].dropna().unique())
            elif sel_sido != "전체":
                dong_pool = sorted(sm_counts[sm_counts["시도명"] == sel_sido]["행정동명"].dropna().unique())
            else:
                dong_pool = []
            sel_dong = st.selectbox("행정동 선택", ["전체"] + dong_pool, key="sm_dong")

        def _sm_filter(tbl):
            t = tbl
            if sel_sido != "전체" and "시도명" in t.columns:
                t = t[t["시도명"] == sel_sido]
            if sel_gu != "전체" and "시군구명" in t.columns:
                t = t[t["시군구명"] == sel_gu]
            if sel_dong != "전체" and "행정동명" in t.columns:
                t = t[t["행정동명"] == sel_dong]
            return t

        sm_f = _sm_filter(sm_counts)
        biz1_list = sorted(sm_f["상권업종대분류명"].dropna().unique())
        sel_biz1 = st.selectbox("업종 대분류 (전체 기능에 적용)",
                                ["전체"] + biz1_list, key="sm_biz1")
        if sel_biz1 != "전체":
            sm_f = sm_f[sm_f["상권업종대분류명"] == sel_biz1]

        st.caption(f"현재 조건 점포 수: **{int(sm_f['점포수'].sum()):,}개**")
        st.divider()

        # ══════════════════════════════════════════════════════
        # [기능 1] 상권 밀집도 지도
        # ══════════════════════════════════════════════════════
        st.markdown("#### 🗺️ 상권 밀집도 지도")
        st.caption("점포 샘플을 지도에 표시합니다. 색상은 업종 대분류를 나타냅니다.")

        map_f = sm_map.copy()
        if sel_dong != "전체":
            map_f = map_f[map_f["행정동명"] == sel_dong]
        if sel_biz1 != "전체":
            map_f = map_f[map_f["상권업종대분류명"] == sel_biz1]

        if map_f.empty:
            st.info("지도에 표시할 좌표 데이터가 없습니다.")
        else:
            center_lat = map_f["위도"].mean()
            center_lon = map_f["경도"].mean()
            biz_cats = map_f["상권업종대분류명"].dropna().unique().tolist()
            palette = px.colors.qualitative.Safe
            color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(biz_cats)}

            fig_map = go.Figure()
            for cat in biz_cats:
                sub = map_f[map_f["상권업종대분류명"] == cat]
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
            st.caption(f"⚠️ 지도는 전체의 일부 샘플만 표시됩니다. (표시 수: {len(map_f):,}개)")

        st.divider()

        # ══════════════════════════════════════════════════════
        # [기능 2] 업종별 경쟁 강도 분석
        # ══════════════════════════════════════════════════════
        st.markdown("#### ⚔️ 업종별 경쟁 강도 분석")
        st.caption("행정동 단위로 같은 업종(중분류)이 몇 개나 밀집해 있는지 경쟁 강도를 보여줍니다.")

        comp_biz2_list = sorted(sm_f["상권업종중분류명"].dropna().unique())
        if not comp_biz2_list:
            st.info("선택한 필터 조건에 해당하는 업종 데이터가 없습니다.")
        else:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                comp_biz2 = st.selectbox("분석할 업종 (중분류)", comp_biz2_list, key="sm_comp_biz2")
            with comp_col2:
                comp_top_n = st.slider("상위 행정동 수", 5, 30, 15, key="sm_comp_n")

            comp_df = sm_f[sm_f["상권업종중분류명"] == comp_biz2]
            if comp_df.empty:
                st.info("선택한 업종의 데이터가 없습니다.")
            else:
                comp_by_dong = (comp_df.groupby(["시군구명","행정동명"])["점포수"]
                                .sum().reset_index()
                                .sort_values("점포수", ascending=False).head(comp_top_n))
                comp_by_dong["지역"] = comp_by_dong["시군구명"] + " " + comp_by_dong["행정동명"]

                fig_comp = go.Figure(go.Bar(
                    x=comp_by_dong["점포수"], y=comp_by_dong["지역"],
                    orientation="h", marker_color="#f78166",
                    text=comp_by_dong["점포수"], textposition="outside",
                    hovertemplate="%{y}: %{x}개<extra></extra>",
                ))
                fig_comp.update_layout(
                    xaxis_title="점포 수", yaxis=dict(autorange="reversed"),
                    height=max(300, comp_top_n * 26), margin=dict(t=20, b=40, r=80),
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                dong_cnt = comp_df.groupby("행정동명")["점포수"].sum()
                cx1, cx2, cx3 = st.columns(3)
                cx1.metric(f"'{comp_biz2}' 총 점포 수", f"{int(comp_df['점포수'].sum()):,}개")
                cx2.metric("행정동 평균 점포 수", f"{dong_cnt.mean():.1f}개")
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

        rec_biz2_list = sorted(sm_counts["상권업종중분류명"].dropna().unique())
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            rec_biz2 = st.selectbox("창업 희망 업종 (중분류)", rec_biz2_list, key="sm_rec_biz2")
        with rec_col2:
            rec_top_n = st.slider("추천 지역 수", 5, 20, 10, key="sm_rec_n")

        rec_scope = _sm_filter(sm_counts)
        rec_store_cnt = (rec_scope[rec_scope["상권업종중분류명"] == rec_biz2]
                         .groupby("행정동명")["점포수"].sum().reset_index())

        fp_data_rec = next(
            (v for k, v in st.session_state.items()
             if k.startswith("fp_data") and isinstance(v, dict) and "age" in v),
            {}
        )
        fp_available = bool(fp_data_rec)

        if fp_available:
            fp_dong_pop = (
                fp_data_rec["age"][fp_data_rec["age"]["FORN_GB"] == "N"]
                .groupby("ADMI_NM")["TOTAL_CNT"].mean().reset_index()
                .rename(columns={"ADMI_NM": "행정동명", "TOTAL_CNT": "평균유동인구"})
            )
            rec_merged = pd.merge(rec_store_cnt, fp_dong_pop, on="행정동명", how="inner")
            if rec_merged.empty:
                st.info("유동인구와 상가 데이터가 겹치는 행정동이 없습니다.")
            else:
                rec_merged["유동인구/점포"] = (
                    rec_merged["평균유동인구"] / rec_merged["점포수"].clip(lower=1)).round(1)
                rec_result = rec_merged.sort_values("유동인구/점포", ascending=False).head(rec_top_n)
                fig_rec = go.Figure(go.Bar(
                    x=rec_result["유동인구/점포"], y=rec_result["행정동명"],
                    orientation="h",
                    marker=dict(color=rec_result["유동인구/점포"],
                                colorscale="Greens", showscale=True,
                                colorbar=dict(title="유동인구/점포")),
                    hovertemplate="%{y}<br>유동인구/점포: %{x:,.1f}<extra></extra>",
                ))
                fig_rec.update_layout(
                    xaxis_title="유동인구 / 점포 수 (높을수록 유망)",
                    yaxis=dict(autorange="reversed"),
                    height=max(300, rec_top_n * 30), margin=dict(t=20, b=40, r=80),
                )
                st.plotly_chart(fig_rec, use_container_width=True)
                rec_show = rec_result.copy()
                rec_show.columns = ["행정동명","점포수","평균유동인구","유동인구/점포비율"]
                rec_show["평균유동인구"] = rec_show["평균유동인구"].round(0).astype(int)
                st.dataframe(rec_show.reset_index(drop=True), use_container_width=True)
        else:
            st.info("유동인구 데이터가 없어 점포 수가 적은 행정동(경쟁 약한 지역)으로 대체 추천합니다.")
            rec_low = rec_store_cnt.sort_values("점포수").head(rec_top_n)
            fig_rec2 = go.Figure(go.Bar(
                x=rec_low["점포수"], y=rec_low["행정동명"],
                orientation="h", marker_color="#3fb950",
                hovertemplate="%{y}: %{x}개<extra></extra>",
            ))
            fig_rec2.update_layout(
                xaxis_title="점포 수 (적을수록 경쟁 약함)",
                yaxis=dict(autorange="reversed"),
                height=max(300, rec_top_n * 26), margin=dict(t=20, b=40, r=80),
            )
            st.plotly_chart(fig_rec2, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════
        # [기능 4] 주변 상권 검색 (행정동 상권 생태계)
        # ══════════════════════════════════════════════════════
        st.markdown("#### 🔍 주변 상권 검색")
        st.caption("행정동을 선택하면 해당 동네의 업종 구성을 한눈에 확인할 수 있습니다.")

        srch_sido = st.selectbox("시/도", sido_list, key="srch_sido")
        srch_gu_pool = sorted(sm_dong[sm_dong["시도명"] == srch_sido]["시군구명"].dropna().unique())
        srch_gu = st.selectbox("시/군/구", srch_gu_pool, key="srch_gu") if srch_gu_pool else None
        if srch_gu:
            srch_dong_pool = sorted(
                sm_dong[(sm_dong["시도명"] == srch_sido) & (sm_dong["시군구명"] == srch_gu)]["행정동명"].dropna().unique()
            )
            srch_dong_sel = st.selectbox("행정동", srch_dong_pool, key="srch_dong") if srch_dong_pool else None
        else:
            srch_dong_sel = None

        if srch_dong_sel:
            srch_df = sm_dong[
                (sm_dong["시도명"] == srch_sido) &
                (sm_dong["시군구명"] == srch_gu) &
                (sm_dong["행정동명"] == srch_dong_sel)
            ]
        else:
            srch_df = pd.DataFrame()

        if srch_df.empty:
            st.info("해당 행정동의 상권 데이터가 없습니다.")
        else:
            sr1, sr2 = st.columns(2)
            with sr1:
                st.markdown("##### 업종 대분류 구성")
                pie_data = (srch_df.groupby("상권업종대분류명")["점포수"].sum()
                            .reset_index().rename(columns={"상권업종대분류명":"업종"}))
                fig_pie = go.Figure(go.Pie(
                    labels=pie_data["업종"], values=pie_data["점포수"],
                    hole=0.4, marker_colors=px.colors.qualitative.Safe,
                    hovertemplate="%{label}: %{value}개 (%{percent})<extra></extra>",
                ))
                fig_pie.update_layout(height=340, margin=dict(t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with sr2:
                st.markdown("##### 업종 중분류 TOP 15")
                mid_cnt = (srch_df.groupby("상권업종중분류명")["점포수"].sum()
                           .sort_values(ascending=False).head(15).reset_index())
                mid_cnt.columns = ["업종(중분류)", "점포수"]
                fig_mid = go.Figure(go.Bar(
                    x=mid_cnt["점포수"], y=mid_cnt["업종(중분류)"],
                    orientation="h", marker_color="#58a6ff",
                    hovertemplate="%{y}: %{x}개<extra></extra>",
                ))
                fig_mid.update_layout(
                    xaxis_title="점포 수", yaxis=dict(autorange="reversed"),
                    height=340, margin=dict(t=10, b=10, r=60),
                )
                st.plotly_chart(fig_mid, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("총 점포 수", f"{int(srch_df['점포수'].sum()):,}개")
            m2.metric("업종 대분류 종류", f"{srch_df['상권업종대분류명'].nunique()}개")
            m3.metric("업종 중분류 종류", f"{srch_df['상권업종중분류명'].nunique()}개")
