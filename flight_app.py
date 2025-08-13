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
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

st.title("Flight Delay Predictor")
st.write("This application predicts if your flight will be delayed based on selected factors.")


@st.cache_data
def load_kaggle_data():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                "patrickzel/flight-delay-and-cancellation-dataset-2019-2023",
                "flights_sample_3m.csv")

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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, solver='liblinear'),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12),
        'SVM (Linear)': SVC(random_state=42, kernel='linear', probability=True),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}
    for name, model in models.items():
        
        X_fit = X_train_scaled.toarray() if (name == 'Naive Bayes' and hasattr(X_train_scaled, 'toarray')) else X_train_scaled
        model.fit(X_fit, y_train)
        trained_models[name] = model


    return trained_models, scaler, X.columns.tolist(), df, X_test_scaled, y_test, X_train_scaled, y_train


models, scaler, feature_names, df_training, X_test_scaled, y_test, X_train_scaled, y_train = train_models()


st.header("Flight Delay Prediction")

col1, col2 = st.columns(2)
with col1:
    dep_time_input = st.time_input("Departure Time (HH:MM)", value=datetime.strptime("08:00", "%H:%M").time())
    crs_dep_time = int(dep_time_input.strftime("%H%M"))
    airline = st.selectbox("Airline", sorted(df_training['AIRLINE'].unique().tolist()))
with col2:
    month = st.selectbox("Flight Month (1–12)", list(range(1, 13)))
    destination = st.selectbox("Destination Airport Code", sorted(df_training['DEST'].unique().tolist()))

if st.button("Predict Flight Status", type="primary", use_container_width=True):
    try:
        input_dict = {
            'CRS_DEP_TIME': crs_dep_time,
            'CRS_ELAPSED_TIME': 90,
            f'MONTH_{month}': 1,
            f'AIRLINE_{airline}': 1,
            f'DEST_{destination}': 1
        }
        input_df = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_df)

        all_predictions = [model.predict(input_scaled)[0] for model in models.values()]
        most_common = max(set(all_predictions), key=all_predictions.count)

        if most_common == 'Not Delayed':
            st.success("Your flight is expected to be on time or only slightly delayed.")
        else:
            st.error("Your flight is expected to be significantly delayed.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")



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
st.write(
    report_df.style.format({
        'Accuracy': '{:.3f}',
        'Weighted Precision': '{:.3f}',
        'Weighted Recall': '{:.3f}',
        'Weighted F1': '{:.3f}',
    }).hide(axis='index').to_html(),
    unsafe_allow_html=True
)

st.subheader("Model vs. Weighted Avg Metrics")
bar_df = pd.DataFrame(metrics_for_bars)
models_order = list(report_df['Model'].values)
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1']

fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
x = np.arange(len(models_order))
width = 0.18

for i, m in enumerate(metrics_list):
    vals = []
    for model_name in models_order:
        val = bar_df[(bar_df['Model'] == model_name) & (bar_df['Metric'] == m)]['Value'].values
        vals.append(val[0] if len(val) else 0.0)
    ax_bar.bar(x + i*width, vals, width, label=m)

ax_bar.set_xticks(x + width*1.5)
ax_bar.set_xticklabels(models_order, rotation=20, ha='right')
ax_bar.set_ylim(0, 1.05)
ax_bar.set_ylabel('Score')
ax_bar.set_title('Model vs. Weighted Avg (Test Set)')
ax_bar.legend()
st.pyplot(fig_bar)



st.header("5-Fold Cross-Validation")
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
ax_cv.set_title('5-Fold Cross-Validation')
plt.setp(ax_cv.get_xticklabels(), rotation=20, ha='right')
st.pyplot(fig_cv)


st.header("Elbow Method (K-Means on Training Features)")
k_min = st.number_input("Min k", min_value=2, max_value=3000, value=2, step=1)
k_max = st.number_input("Max k", min_value=k_min, max_value=3000, value=max(k_min+8, 10), step=1)

ks = list(range(int(k_min), int(k_max)+1))
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_train_scaled)
    inertias.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
ax_elbow.plot(ks, inertias, marker='o')
ax_elbow.set_xlabel('Number of Clusters (k)')
ax_elbow.set_ylabel('Sum of Squared Distances')
ax_elbow.set_title('Elbow Method')
st.pyplot(fig_elbow)


flight_counts = df_training['AIRLINE'].value_counts()
top_airlines = flight_counts.head(10).index
filtered_df = df_training[df_training['AIRLINE'].isin(top_airlines)]

flight_counts_top = filtered_df['AIRLINE'].value_counts().sort_index()
delay_rates_top = (
    filtered_df.groupby('AIRLINE')['FLIGHT_STATUS']
    .value_counts(normalize=True)
    .unstack()
    .get('Delayed', pd.Series(0, index=top_airlines))
)

combined_df = pd.DataFrame({
    'AIRLINE': flight_counts_top.index,
    'Flight Count': flight_counts_top.values,
    'Delay Rate': delay_rates_top.values
})

fig = px.bar(combined_df, x='AIRLINE', y='Flight Count',
             title="Top 10 Airlines: Flight Volume vs Delay Rate")

fig.add_scatter(x=combined_df['AIRLINE'], y=combined_df['Delay Rate'],
                mode='lines+markers', name='Delay Rate',
                yaxis='y2')

fig.update_layout(
    yaxis=dict(title='Flight Count'),
    yaxis2=dict(title='Delay Rate', overlaying='y', side='right', tickformat=".0%"),
    legend=dict(x=0.75, y=1.1),
)

st.plotly_chart(fig, use_container_width=True)
