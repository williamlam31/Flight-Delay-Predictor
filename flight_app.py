
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

st.title("Flight Delay Predictor ")

@st.cache_data
def load_kaggle_data():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "patrickzel/flight-delay-and-cancellation-dataset-2019-2023",
        "flights_sample_3m.csv"
    )

    df_processed = df.copy()
    features_to_fill = ['CRS_ELAPSED_TIME', 'DEP_TIME', 'DEP_DELAY', 'ARR_TIME',
                        'ARR_DELAY', 'ELAPSED_TIME', 'AIR_TIME', 'DELAY_DUE_CARRIER',
                        'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                        'DELAY_DUE_LATE_AIRCRAFT']
    df_processed[features_to_fill] = df_processed[features_to_fill].fillna(0)

    features_to_drop = ['TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON','TAXI_IN', 'CANCELLATION_CODE']
    df_processed.drop(features_to_drop, axis=1, inplace=True)
    return df_processed

@st.cache_data
def sample_data(n=10000, seed=42):
    df = load_kaggle_data()
    df = df.sample(n=n, random_state=seed)
    df = df.dropna(subset=['ARR_DELAY', 'CANCELLED'])
    df['MONTH'] = pd.to_datetime(df['FL_DATE']).dt.month
    df['FLIGHT_STATUS'] = np.where(df['ARR_DELAY'] > 15, 'Delayed', 'Not Delayed')
    df = df[['CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 'MONTH', 'AIRLINE', 'DEST', 'ARR_DELAY', 'CANCELLED', 'FLIGHT_STATUS']]
    return df


@st.cache_resource
def train_models():
    df = sample_data()
    X = df[['CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 'MONTH', 'AIRLINE', 'DEST']]
    X = pd.get_dummies(X, columns=['MONTH', 'AIRLINE', 'DEST'], drop_first=True)
    y = df['FLIGHT_STATUS']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    scaler = StandardScaler(with_mean=False) 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, solver='liblinear'),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12),
        'SVM (Linear)': SVC(random_state=42, kernel='linear', probability=True),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
    }

    trained = {}
    for name, model in models.items():
        X_fit = X_train_scaled.toarray() if (name == 'Naive Bayes' and hasattr(X_train_scaled, 'toarray')) else X_train_scaled
        model.fit(X_fit, y_train)
        trained[name] = model

    return trained, scaler, X.columns.tolist(), df, X_test_scaled, y_test, X_train_scaled, y_train


models, scaler, feature_names, df_training, X_test_scaled, y_test, X_train_scaled, y_train = train_models()


st.header("Model Evaluation (Test Set)")

report_rows = []
metrics_for_bars = []

for name, model in models.items():
    X_eval = X_test_scaled.toarray() if (name == 'Naive Bayes' and hasattr(X_test_scaled, 'toarray')) else X_test_scaled
    y_pred = model.predict(X_eval)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    row = {
        'Model': name,
        'Accuracy': acc,
        'Weighted Precision': report['weighted avg']['precision'],
        'Weighted Recall': report['weighted avg']['recall'],
        'Weighted F1': report['weighted avg']['f1-score'],
    }
    report_rows.append(row)

    metrics_for_bars.append({'Model': name, 'Metric': 'Accuracy', 'Value': row['Accuracy']})
    metrics_for_bars.append({'Model': name, 'Metric': 'Precision', 'Value': row['Weighted Precision']})
    metrics_for_bars.append({'Model': name, 'Metric': 'Recall', 'Value': row['Weighted Recall']})
    metrics_for_bars.append({'Model': name, 'Metric': 'F1', 'Value': row['Weighted F1']})

report_df = pd.DataFrame(report_rows).sort_values('Accuracy', ascending=False).reset_index(drop=True)


st.subheader("Classification Report (Weighted Averages)")
st.write(report_df.style.format({
    'Accuracy': '{:.3f}',
    'Weighted Precision': '{:.3f}',
    'Weighted Recall': '{:.3f}',
    'Weighted F1': '{:.3f}',
}).hide(axis='index').to_html(), unsafe_allow_html=True)


st.header("5-Fold Cross-Validation (Training Set)")
cv_scores = []
for name, model in models.items():
    X_cv = X_train_scaled.toarray() if (name == 'Naive Bayes' and hasattr(X_train_scaled, 'toarray')) else X_train_scaled
    scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='accuracy')
    cv_scores.append({'Model': name, 'CV_Accuracy': float(np.mean(scores))})

cv_df = pd.DataFrame(cv_scores).sort_values('CV_Accuracy', ascending=False).reset_index(drop=True)

st.write(cv_df.style.format({'CV_Accuracy': '{:.3f}'}).hide(axis='index').to_html(), unsafe_allow_html=True)


fig_cv, ax_cv = plt.subplots(figsize=(8, 4))
ax_cv.bar(cv_df['Model'], cv_df['CV_Accuracy'])
ax_cv.set_ylim(0, 1.05)
ax_cv.set_ylabel('Mean Accuracy (CV=5)')
ax_cv.set_title('5-Fold Cross-Validation (Training Set)')
plt.setp(ax_cv.get_xticklabels(), rotation=20, ha='right')
st.pyplot(fig_cv)

st.subheader("Model vs. Weighted Avg Metrics")
bar_df = pd.DataFrame(metrics_for_bars)
models_order = list(report_df['Model'].values)  
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1']

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models_order))
width = 0.18

for i, m in enumerate(metrics_list):
    vals = []
    for model_name in models_order:
        val = bar_df[(bar_df['Model'] == model_name) & (bar_df['Metric'] == m)]['Value'].values
        vals.append(val[0] if len(val) else 0.0)
    ax.bar(x + i*width, vals, width, label=m)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(models_order, rotation=20, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Model vs. Weighted Avg (Test Set)')
ax.legend()
st.pyplot(fig)


st.header("Elbow Method (K-Means on Training Features)")
k_min = st.number_input("Min k", min_value=2, max_value=3000, value=2, step=1)
k_max = st.number_input("Max k", min_value=k_min, max_value=3000, value=max(k_min+8, 10), step=1)

ks = list(range(int(k_min), int(k_max)+1))
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_train_scaled)
    inertias.append(km.inertia_)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(ks, inertias, marker='o')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Sum of Squared Distances')
ax2.set_title('Elbow Method')
st.pyplot(fig2)

st.caption("Implements 70:30 split, classification report (no index), Model vs Weighted Avg chart, and Elbow method chart.")
