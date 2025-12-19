
import io
import numpy as np
import pandas as pd
import streamlit as st

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt


st.set_page_config(page_title="MLP 신용평가 (보조 실험)", layout="wide")
st.title("MLP(신경망) 보조 실험용 Streamlit")

# ----------------------------
# Helpers
# ----------------------------
def safe_read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding_errors="ignore")


def iqr_filter_mask(df_num: pd.DataFrame, k: float = 1.5) -> np.ndarray:
    """
    Returns a boolean mask (True = keep) based on IQR rule across numeric columns.
    If a row has an outlier in ANY numeric column, it will be filtered out.
    """
    if df_num.shape[1] == 0:
        return np.ones(len(df_num), dtype=bool)

    q1 = df_num.quantile(0.25)
    q3 = df_num.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    # Keep rows where all numeric values are within [lower, upper] (NaNs are kept)
    mask = np.ones(len(df_num), dtype=bool)
    for col in df_num.columns:
        x = df_num[col]
        lo = lower[col]
        hi = upper[col]
        # if iqr is nan (constant column), skip filtering that column
        if pd.isna(lo) or pd.isna(hi):
            continue
        mask &= (x.isna() | ((x >= lo) & (x <= hi))).to_numpy()
    return mask


def ttest_select_numeric(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 0.05):
    """
    Two-sample t-test on numeric columns between classes y=0 and y=1.
    Returns selected numeric columns and a p-value table.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        return [], pd.DataFrame(columns=["feature", "p_value"])

    y0 = X_train.loc[y_train == 0, num_cols]
    y1 = X_train.loc[y_train == 1, num_cols]

    rows = []
    for c in num_cols:
        a = y0[c].dropna().to_numpy()
        b = y1[c].dropna().to_numpy()
        # Require minimal samples
        if len(a) < 3 or len(b) < 3:
            p = np.nan
        else:
            # Welch's t-test (unequal variances)
            _, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append((c, p))

    ptab = pd.DataFrame(rows, columns=["feature", "p_value"]).sort_values("p_value", na_position="last")
    selected = ptab.loc[ptab["p_value"].notna() & (ptab["p_value"] <= alpha), "feature"].tolist()
    return selected, ptab


def plot_roc(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {auc:.4f})")
    st.pyplot(fig)


def plot_pr(y_true, proba):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig = plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AP = {ap:.4f})")
    st.pyplot(fig)


# ----------------------------
# UI
# ----------------------------
uploaded = st.file_uploader("CSV 업로드", type=["csv"])

if uploaded is None:
    st.info("분석할 CSV 파일을 업로드하세요. (예: loan_data(FICO).csv)")
    st.stop()

df = safe_read_csv(uploaded)
st.subheader("데이터 미리보기")
st.dataframe(df.head(20), use_container_width=True)

# Target selection
with st.sidebar:
    st.header("설정")
    target_col = st.selectbox("타겟(레이블) 컬럼", options=df.columns.tolist(), index=(df.columns.tolist().index("not.fully.paid") if "not.fully.paid" in df.columns else 0))
    test_size = st.slider("Test size", 0.1, 0.4, 0.3, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)

    st.divider()
    st.subheader("MLP 하이퍼파라미터")
    hidden = st.selectbox("hidden_layer_sizes", ["(64, 32)", "(128, 64)", "(64, 64)", "(100,)"], index=0)
    alpha = st.selectbox("alpha(L2)", [0.0001, 0.001, 0.01, 0.1], index=1)
    max_iter = st.selectbox("max_iter", [300, 500, 800], index=1)
    use_class_weight = st.checkbox("클래스 불균형 보정(샘플 가중치)", value=True)

    st.divider()
    st.caption("전처리: t-test p≤0.05 고정, IQR 이상치 제거 k=1.5 고정(강도 설정 없음).")


# Ensure binary target
y_raw = df[target_col]
if y_raw.dtype.kind in "OSU":
    # Try to map common strings to 0/1
    y = y_raw.astype(str).str.lower().map({"0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1})
else:
    y = y_raw

if y.isna().any():
    st.warning("타겟 컬럼에 결측/비정상 값이 있어 일부 행이 제외됩니다.")
mask_ok = y.notna()
df = df.loc[mask_ok].copy()
y = y.loc[mask_ok].astype(int)

X = df.drop(columns=[target_col])

st.subheader("타겟 분포")
dist = y.value_counts().sort_index()
st.write(dist)
st.write((dist / dist.sum()).rename("ratio"))

# ----------------------------
# Preprocess button
# ----------------------------
if "prepared" not in st.session_state:
    st.session_state.prepared = False

if st.button("데이터 전처리"):
    # 1) split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
    )

    # 2) fixed IQR outlier removal on TRAIN only (k=1.5)
    num_cols_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
    train_mask = iqr_filter_mask(X_train[num_cols_all], k=1.5) if len(num_cols_all) else np.ones(len(X_train), dtype=bool)
    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[X_train.index].copy()

    # 3) t-test selection on TRAIN numeric columns only (p<=0.05 fixed)
    selected_num, ptab = ttest_select_numeric(X_train, y_train, alpha=0.05)

    # 4) Build preprocessing pipeline:
    #    - numeric: impute median + standardize
    #    - categorical: impute most_frequent + onehot
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # If bool exists, treat as categorical (onehot). If you prefer as numeric, change here.

    # numeric columns after selection
    num_cols = [c for c in selected_num if c in X_train.columns]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    st.session_state.data_bundle = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "p_table": ptab,
        "preprocessor": preprocessor,
    }
    st.session_state.prepared = True

    st.success("전처리 완료: split → (train) IQR 제거(k=1.5) → t-test(p≤0.05) → 스케일링 준비")
    st.write(f"Train: {X_train.shape}, Test: {X_test.shape}")
    st.write(f"선택된 수치 변수 개수(t-test p≤0.05): {len(num_cols)}")
    with st.expander("t-test p-value 표 보기"):
        st.dataframe(ptab, use_container_width=True)

# ----------------------------
# Train MLP
# ----------------------------
if st.session_state.prepared:
    st.subheader("MLP 학습 및 평가")

    bundle = st.session_state.data_bundle
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    preprocessor = bundle["preprocessor"]

    # parse hidden layer string
    hidden_map = {
        "(64, 32)": (64, 32),
        "(128, 64)": (128, 64),
        "(64, 64)": (64, 64),
        "(100,)": (100,),
    }
    hidden_layers = hidden_map.get(hidden, (64, 32))

    # sample weights for imbalance (optional)
    sample_weight = None
    if use_class_weight:
        # inverse frequency weighting
        counts = y_train.value_counts()
        w0 = 1.0 / max(counts.get(0, 1), 1)
        w1 = 1.0 / max(counts.get(1, 1), 1)
        sample_weight = y_train.map({0: w0, 1: w1}).to_numpy()

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        max_iter=int(max_iter),
        random_state=int(random_state),
        early_stopping=True,
        n_iter_no_change=10,
    )

    clf = Pipeline(steps=[
        ("prep", preprocessor),
        ("mlp", mlp)
    ])

    if st.button("MLP 학습 시작"):
        clf.fit(X_train, y_train, mlp__sample_weight=sample_weight)

        # predict
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        cm = confusion_matrix(y_test, pred)

        st.session_state.trained = True
        st.session_state.model = clf
        st.session_state.eval = {"proba": proba, "pred": pred, "cm": cm}

        st.success("MLP 학습 완료")

    if st.session_state.get("trained", False):
        proba = st.session_state.eval["proba"]
        pred = st.session_state.eval["pred"]
        cm = st.session_state.eval["cm"]

        c1, c2 = st.columns(2)
        with c1:
            st.write("Confusion Matrix")
            st.write(cm)
            st.write("Classification Report")
            st.text(classification_report(y_test, pred, digits=4))
        with c2:
            st.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.4f}")
            st.metric("Average Precision(AP)", f"{average_precision_score(y_test, proba):.4f}")

        st.subheader("곡선")
        c3, c4 = st.columns(2)
        with c3:
            plot_roc(y_test, proba)
        with c4:
            plot_pr(y_test, proba)

        with st.expander("모델 설명(보고서용)"):
            st.write(
                "본 페이지는 MLP를 '보조 실험'으로 제시하기 위한 Streamlit 화면입니다. "
                "전처리는 (1) 범주형 변수의 수치화(원-핫), (2) X/y 분리, (3) 학습/테스트 분할, "
                "(4) 학습 데이터 기준 표준화(데이터 누수 방지) 순으로 수행합니다. "
                "또한 수치형 변수에 대해 학습 데이터에서만 IQR(k=1.5) 규칙으로 이상치를 제거하고, "
                "t-test(p≤0.05)로 유의한 수치 변수만 선택합니다."
            )
else:
    st.info("먼저 [데이터 전처리] 버튼을 눌러주세요.")
