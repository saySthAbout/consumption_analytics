import glob, os, sys, time
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
ENCODER_DIR = os.path.join(BASE_DIR, "encoders")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

MODEL_FEATURES = ["sex", "age", "day", "hour", "month",
                  "admi_cty_no", "card_tpbuz_nm_1", "card_tpbuz_nm_2"]
LABEL_COLS  = ["age", "day", "hour", "month"]
ONEHOT_COLS = ["sex", "admi_cty_no", "card_tpbuz_nm_1", "card_tpbuz_nm_2"]

SAMPLE_SIZE    = 300000
REMOVE_OUTLIER = True

# 1) 전체 CSV 로드
files = sorted(glob.glob(os.path.join(DATASET_DIR, "tbsh_gyeonggi_day_*.csv")))
print(f"파일 수: {len(files)}개")

# 파일당 샘플 수 (총 SAMPLE_SIZE를 파일 수로 나눔)
per_file = max(1000, SAMPLE_SIZE // len(files))

dfs = []
for i, f in enumerate(files, 1):
    city = os.path.basename(f).split("_")[-1].replace(".csv", "")
    df = pd.read_csv(f, encoding="utf-8-sig",
                     usecols=["ta_ymd","admi_cty_no","card_tpbuz_nm_1","card_tpbuz_nm_2",
                               "hour","sex","age","day","amt"])
    df["month"] = df["ta_ymd"].astype(str).str[4:6].astype(int)
    df["admi_cty_no"] = df["admi_cty_no"].astype(str)
    df["log_amt"] = np.log1p(df["amt"])
    # 아웃라이어 제거
    if REMOVE_OUTLIER:
        upper = df["amt"].quantile(0.99)
        df = df[df["amt"] <= upper]
    # 필요 컬럼만 남기고 샘플링
    df = df[MODEL_FEATURES + ["log_amt"]].dropna()
    if len(df) > per_file:
        df = df.sample(per_file, random_state=42)
    dfs.append(df)
    print(f"  [{i}/{len(files)}] {city}: {len(df):,}행 샘플")

df_all = pd.concat(dfs, ignore_index=True)
print(f"\n전체 데이터: {len(df_all):,}행")

model_df = df_all
if len(model_df) > SAMPLE_SIZE:
    model_df = model_df.sample(SAMPLE_SIZE, random_state=42)
    print(f"최종 샘플링: {len(model_df):,}행")

X = model_df[MODEL_FEATURES].copy()
y = model_df["log_amt"]

# 3) 인코딩
for col in LABEL_COLS:
    X[col] = X[col].astype(str)

label_encoders = {}
for col in LABEL_COLS:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_arr   = ohe.fit_transform(X[ONEHOT_COLS].astype(str))
ohe_names = ohe.get_feature_names_out(ONEHOT_COLS).tolist()
ohe_df    = pd.DataFrame(ohe_arr, columns=ohe_names, index=X.index)
encoded_X = pd.concat([X[LABEL_COLS].reset_index(drop=True),
                       ohe_df.reset_index(drop=True)], axis=1)
feature_columns = encoded_X.columns.tolist()

print(f"피처 수: {len(feature_columns)}개")

# 4) 학습
X_train, X_test, y_train, y_test = train_test_split(
    encoded_X, y, test_size=0.2, random_state=42)

print(f"\nLightGBM 학습 중... (학습 데이터: {len(X_train):,}행)")
t0 = time.time()
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                           num_leaves=127, random_state=42, n_jobs=-1,
                           verbose=-1)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.log_evaluation(50)])
print(f"학습 완료: {time.time()-t0:.1f}초")

# 5) 평가
pred = model.predict(X_test)
y_real   = np.expm1(y_test)
pred_real = np.maximum(np.expm1(pred), 0)

r2   = r2_score(y_real, pred_real)
rmse = np.sqrt(mean_squared_error(y_real, pred_real))
mae  = mean_absolute_error(y_real, pred_real)
print(f"\nR²={r2:.4f}  RMSE={rmse:,.0f}  MAE={mae:,.0f}")

# 6) 저장
joblib.dump(model,          os.path.join(MODEL_DIR,   "sales_predict_model.pkl"))
joblib.dump(label_encoders, os.path.join(ENCODER_DIR, "label_encoders.pkl"))
joblib.dump(ohe,            os.path.join(ENCODER_DIR, "onehot_encoder.pkl"))
joblib.dump(feature_columns,os.path.join(ENCODER_DIR, "feature_columns.pkl"))
joblib.dump({
    "model_name": "LightGBM", "R2": r2, "RMSE": rmse, "MAE": mae,
    "sample_size": SAMPLE_SIZE, "features": MODEL_FEATURES,
    "use_log_target": True, "remove_outliers": REMOVE_OUTLIER,
}, os.path.join(MODEL_DIR, "model_info.pkl"))

print("\n모델 및 인코더 저장 완료!")
print(f"  {MODEL_DIR}/sales_predict_model.pkl")
print(f"  {ENCODER_DIR}/label_encoders.pkl, onehot_encoder.pkl, feature_columns.pkl")
