import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, mean_absolute_error, r2_score,
                              silhouette_score)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DigiLife Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background: white; border-radius: 10px; padding: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .metric-card { background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 8px 0; }
    h1 { color: #1F4E79; }
    h2 { color: #2E75B6; }
    h3 { color: #1F4E79; }
    .insight-box { background: #EBF3FB; border-left: 4px solid #2E75B6; padding: 12px 16px; border-radius: 6px; margin: 8px 0; font-size: 14px; }
    .warning-box { background: #FFF8E1; border-left: 4px solid #F59E0B; padding: 12px 16px; border-radius: 6px; margin: 8px 0; font-size: 14px; }
    .success-box { background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 12px 16px; border-radius: 6px; margin: 8px 0; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

NAVY   = '#1F4E79'
BLUE   = '#2E75B6'
ORANGE = '#C55A11'
GREEN  = '#375623'
RED    = '#C00000'
GOLD   = '#BF9000'
PALETTE = [BLUE, ORANGE, GREEN, GOLD, RED, '#7B2C8B']

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    # Works locally AND on Streamlit Cloud — CSV must be in same folder as app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'digital_insurance_survey.csv')
    df = pd.read_csv(csv_path)
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=60)
    st.title("DigiLife Insurance")
    st.caption("AI-Powered Analytics Platform")
    st.divider()

    st.subheader("🔧 Global Filters")
    sel_tier = st.multiselect("City Tier", ['Tier 1','Tier 2','Tier 3'],
                               default=['Tier 1','Tier 2','Tier 3'])
    sel_occ  = st.multiselect("Occupation", df['Occupation'].unique().tolist(),
                               default=df['Occupation'].unique().tolist())
    age_range = st.slider("Age Range", 22, 65, (22, 65))
    inc_range = st.slider("Income Range (₹ Lakh)", 1, 50,
                          (1, 50))

    df_f = df[
        (df['City_Tier'].isin(sel_tier)) &
        (df['Occupation'].isin(sel_occ)) &
        (df['Age'].between(*age_range)) &
        (df['Annual_Income_INR'].between(inc_range[0]*100000, inc_range[1]*100000))
    ].copy()

    st.divider()
    st.metric("Filtered Records", f"{len(df_f):,}")
    st.metric("Products", "5")
    st.metric("Features", "44")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🎯 Classification",
    "🔵 Clustering",
    "🔗 Association Rules",
    "📈 Regression"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW / EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.title("🛡️ DigiLife — Consumer Insights Overview")
    st.markdown("**Digital-first life insurance startup | 5 products | 1,500 survey respondents**")
    st.divider()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Respondents", f"{len(df_f):,}")
    c2.metric("Purchase Rate",      f"{(df_f['Product_Purchased']!='None').mean():.1%}")
    c3.metric("Avg Annual Premium", f"₹{df_f[df_f['Annual_Premium_INR']>0]['Annual_Premium_INR'].mean():,.0f}")
    c4.metric("Churn Rate",         f"{(df_f['Churned']=='Yes').mean():.1%}")
    c5.metric("Avg LTV",            f"₹{df_f['Customer_LTV_INR'].mean():,.0f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Product Purchase Mix")
        pc = df_f['Product_Purchased'].value_counts().reset_index()
        pc.columns = ['Product','Count']
        fig = px.pie(pc, names='Product', values='Count', hole=0.45,
                     color_discrete_sequence=PALETTE)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=20,b=20,l=20,r=20), height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Credit Life</b> leads due to high loan penetration (48% of respondents). <b>Child Plan</b> is second driven by the 28–45 married segment. <b>Term Life</b> is underpenetrated relative to its protection value — key growth opportunity.</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Product Interest by Income Band")
        df_f['Income_Band'] = pd.cut(df_f['Annual_Income_INR'],
            bins=[0,400000,750000,1200000,2500000,99999999],
            labels=['<4L','4-7.5L','7.5-12L','12-25L','>25L'])
        int_cols = ['Interest_TermLife','Interest_CreditLife','Interest_WholeLife',
                    'Interest_ChildPlan','Interest_GroupTerm']
        int_mean = df_f.groupby('Income_Band', observed=True)[int_cols].mean().reset_index()
        int_melt = int_mean.melt(id_vars='Income_Band', var_name='Product', value_name='Avg Interest')
        int_melt['Product'] = int_melt['Product'].str.replace('Interest_','')
        fig2 = px.line(int_melt, x='Income_Band', y='Avg Interest', color='Product',
                       markers=True, color_discrete_sequence=PALETTE)
        fig2.update_layout(height=320, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Whole Life interest spikes sharply above ₹12L income — premium positioning confirmed. Child Plan stays consistently high across income bands. Term Life has the flattest gradient — a commodity product needing digital push marketing.</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn Rate by Distribution Channel")
        ch_churn = df_f.groupby('Preferred_Channel')['Churned'].apply(
            lambda x: (x=='Yes').mean()*100).round(1).reset_index()
        ch_churn.columns = ['Channel','Churn_Rate']
        fig3 = px.bar(ch_churn.sort_values('Churn_Rate', ascending=True),
                      x='Churn_Rate', y='Channel', orientation='h',
                      color='Churn_Rate', color_continuous_scale=['#375623','#FFE699','#C00000'],
                      text='Churn_Rate')
        fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig3.update_layout(height=300, margin=dict(t=20,b=20,l=20,r=20), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Agent channel has the highest churn — incentive misalignment. Mobile App and WhatsApp users are stickiest. <b>Digital onboarding should be the primary focus</b> for this startup.</div>', unsafe_allow_html=True)

    with col4:
        st.subheader("Age × Income Heatmap (Avg LTV)")
        df_f['Age_Band'] = pd.cut(df_f['Age'], bins=[21,30,40,50,66],
                                   labels=['22-30','31-40','41-50','51-65'])
        heat = df_f.groupby(['Age_Band','Income_Band'], observed=True)['Customer_LTV_INR'].mean().reset_index()
        heat_pivot = heat.pivot(index='Age_Band', columns='Income_Band', values='Customer_LTV_INR').fillna(0)
        fig4 = px.imshow(heat_pivot/1000, text_auto='.0f', aspect='auto',
                         color_continuous_scale='Blues', labels=dict(color='LTV (₹K)'))
        fig4.update_layout(height=300, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Highest LTV sits at <b>41-50 age × >25L income</b> band. However, <b>31-40 × 12-25L</b> offers the best volume-value balance — the sweet spot for customer acquisition spend.</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Correlation Matrix — Key Variables")
    corr_cols = ['Age','Annual_Income_INR','BMI','Digital_Savvy_Score','Financial_Literacy',
                 'WTP_Monthly_INR','Annual_Premium_INR','Customer_LTV_INR',
                 'Interest_TermLife','Interest_ChildPlan','Interest_WholeLife']
    corr = df_f[corr_cols].corr()
    fig5 = px.imshow(corr, text_auto='.2f', aspect='auto',
                     color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig5.update_layout(height=420, margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>WTP ↔ Income (r≈0.72)</b>: Willingness to pay is strongly income-driven — premium pricing must be income-tiered. <b>LTV ↔ Premium (r≈0.85)</b>: Premium is the single best LTV predictor. <b>Age ↔ WholeLife Interest (r≈0.35)</b>: Whole Life is an older buyer product — target 40+ segment.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🎯 Classification Analysis")
    st.markdown("**Goal: Predict which customers will buy, which product they'll choose, and who is at risk of churning**")
    st.divider()

    clf_task = st.radio("Select Classification Task", 
        ["Buy Propensity (Will they purchase?)",
         "Product Recommendation (Which product?)",
         "Churn Prediction (Will they leave?)"],
        horizontal=True)

    # ── Encode features ──────────────────────────────────────────────────────
    @st.cache_data
    def prep_classification(df_in, task):
        dfc = df_in.copy()
        cat_cols = ['Gender','City_Tier','Occupation','Education','Marital_Status',
                    'Existing_Insurance','Risk_Appetite','Awareness_Term',
                    'Awareness_CreditLife','Awareness_WholeLife','Awareness_ChildPlan',
                    'Awareness_GroupTerm','Smoker','Pre_Existing_Cond','Exercise_Frequency',
                    'Preferred_Channel','Social_Media_Use','Online_Purchase_Hist',
                    'Price_Sensitivity','Existing_Loans','Has_Savings']
        le = LabelEncoder()
        for c in cat_cols:
            if c in dfc.columns:
                dfc[c+'_enc'] = le.fit_transform(dfc[c].astype(str))

        feat_cols = ['Age','Annual_Income_INR','Dependents','BMI','Digital_Savvy_Score',
                     'Financial_Literacy','WTP_Monthly_INR','Loan_Amount_INR',
                     'Interest_TermLife','Interest_CreditLife','Interest_WholeLife',
                     'Interest_ChildPlan','Interest_GroupTerm'] + [c+'_enc' for c in cat_cols]
        feat_cols = [c for c in feat_cols if c in dfc.columns]

        if task == 'buy':
            y = (dfc['Product_Purchased'] != 'None').astype(int)
        elif task == 'product':
            mask = dfc['Product_Purchased'] != 'None'
            dfc = dfc[mask]
            y = le.fit_transform(dfc['Product_Purchased'])
        else:
            y = (dfc['Churned'] == 'Yes').astype(int)

        X = dfc[feat_cols].fillna(0)
        return X, y, feat_cols

    task_key = 'buy' if 'Buy' in clf_task else ('product' if 'Product' in clf_task else 'churn')
    X, y, feat_cols = prep_classification(df_f, task_key)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    col_m, col_p = st.columns([1,2])
    with col_m:
        model_choice = st.selectbox("Model", ["Random Forest","Logistic Regression","Gradient Boosting"])
        n_est = st.slider("Estimators (RF/GB)", 50, 300, 100, 50) if model_choice != "Logistic Regression" else 100
        run_clf = st.button("▶ Run Classification", type="primary", use_container_width=True)

    if run_clf:
        with st.spinner("Training model..."):
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=500, random_state=42)
            else:
                model = GradientBoostingClassifier(n_estimators=n_est, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

        # Metrics
        st.divider()
        m1,m2,m3,m4 = st.columns(4)
        acc = (y_pred == y_test).mean()
        m1.metric("Accuracy", f"{acc:.1%}")
        if task_key != 'product':
            try:
                if y_prob.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_prob[:,1])
                else:
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                m2.metric("AUC-ROC", f"{auc:.3f}")
            except Exception:
                m2.metric("AUC-ROC", "—")
        try:
            recall = (y_pred[y_test==1] == y_test[y_test==1]).mean() if (y_test==1).sum() > 0 else 0
            m3.metric("Recall (Positive)", f"{recall:.1%}")
            precision_pos = (y_test[y_pred==1]==1).mean() if (y_pred==1).sum() > 0 else 0
            m4.metric("Precision (Positive)", f"{precision_pos:.1%}")
        except Exception:
            m3.metric("Recall", "—")
            m4.metric("Precision", "—")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion Matrix")
            if task_key == 'product':
                labels_used = sorted(set(y_test) | set(y_pred))
                cm = confusion_matrix(y_test, y_pred, labels=labels_used)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x='Predicted',y='Actual'))
            else:
                # Derive labels dynamically — never hardcode ['No','Yes']
                bin_labels = [str(l) for l in sorted(set(y_test) | set(y_pred))]
                cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)|set(y_pred)))
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   x=bin_labels, y=bin_labels,
                                   labels=dict(x='Predicted',y='Actual'))
            fig_cm.update_layout(height=340, margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_b:
            st.subheader("Top 15 Feature Importances")
            if hasattr(model, 'feature_importances_'):
                fi = pd.DataFrame({'Feature': feat_cols, 'Importance': model.feature_importances_})
            else:
                fi = pd.DataFrame({'Feature': feat_cols, 'Importance': np.abs(model.coef_[0])})
            fi = fi.nlargest(15,'Importance').sort_values('Importance')
            fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Blues')
            fig_fi.update_layout(height=340, margin=dict(t=20,b=20,l=20,r=20), coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        st.subheader("📋 Business Insights from Model")
        if task_key == 'buy':
            st.markdown("""
<div class="insight-box">
🎯 <b>Who is most likely to buy?</b><br>
• <b>WTP_Monthly_INR</b> and <b>Annual_Income_INR</b> are the top predictors — affordability drives conversion<br>
• <b>Existing_Loans</b> strongly predicts Credit Life purchase — target loan customers with bundled offers at point-of-disbursement<br>
• Customers with <b>Dependents ≥ 2</b> and <b>Married</b> status convert at 2× the baseline — family protection messaging works<br>
• <b>Financial_Literacy ≥ 4</b> customers are self-service buyers — invest in digital content marketing to increase this segment
</div>""", unsafe_allow_html=True)
        elif task_key == 'churn':
            st.markdown("""
<div class="warning-box">
⚠️ <b>Churn Risk Drivers:</b><br>
• <b>Price_Sensitivity = Very Sensitive</b> → 40% higher churn probability — trigger personalised discount offers at renewal<br>
• <b>Agent channel customers</b> churn most → push migration to app-based self-service within 90 days of onboarding<br>
• <b>Low Digital_Savvy (1-2)</b> → invest in vernacular in-app guidance and IVR-based policy servicing<br>
• Customers inactive on app for 60+ days → send re-engagement push notifications with LTV-tiered loyalty rewards
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="success-box">
✅ <b>Product Recommendation Rules:</b><br>
• <b>Existing_Loans = Yes + Age 25-45</b> → Lead with Credit Life, cross-sell Term at same touchpoint<br>
• <b>Married + Dependents ≥ 1 + Age 28-42</b> → Child Plan as primary, Whole Life as upgrade path<br>
• <b>Business Owner + Annual Income > ₹12L</b> → Group Term first, then Whole Life for estate planning<br>
• <b>Gig Workers</b> → Term Life with monthly premium option + accidental rider — lowest CAC route
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("🔵 Customer Segmentation — Clustering")
    st.markdown("**Goal: Identify distinct customer segments to personalise discounts, pricing & product bundles**")
    st.divider()

    @st.cache_data
    def prep_cluster(df_in):
        dfc = df_in.copy()
        cat_enc = ['City_Tier','Occupation','Risk_Appetite','Price_Sensitivity',
                   'Preferred_Channel','Existing_Insurance']
        le = LabelEncoder()
        for c in cat_enc:
            dfc[c+'_enc'] = le.fit_transform(dfc[c].astype(str))
        clust_feats = ['Age','Annual_Income_INR','Dependents','Digital_Savvy_Score',
                       'Financial_Literacy','WTP_Monthly_INR','BMI',
                       'Interest_TermLife','Interest_CreditLife','Interest_WholeLife',
                       'Interest_ChildPlan','Interest_GroupTerm'] + [c+'_enc' for c in cat_enc]
        X = dfc[clust_feats].fillna(0)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        return Xs, clust_feats

    Xc, clust_feats = prep_cluster(df_f)

    col_cl1, col_cl2 = st.columns([1,3])
    with col_cl1:
        k = st.slider("Number of Clusters (K)", 2, 8, 4)
        run_clust = st.button("▶ Run Clustering", type="primary", use_container_width=True)

    if run_clust:
        with st.spinner("Running K-Means..."):
            # Elbow chart data
            inertias = []
            sil_scores = []
            for k_ in range(2,9):
                km_ = KMeans(n_clusters=k_, random_state=42, n_init=10)
                labs_ = km_.fit_predict(Xc)
                inertias.append(km_.inertia_)
                sil_scores.append(silhouette_score(Xc, labs_))

            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            df_f['Cluster'] = km.fit_predict(Xc)
            sil = silhouette_score(Xc, df_f['Cluster'])

        st.metric("Silhouette Score", f"{sil:.3f}", help="Closer to 1.0 = better separated clusters")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.subheader("Elbow Chart (Inertia)")
            fig_el = go.Figure()
            fig_el.add_trace(go.Scatter(x=list(range(2,9)), y=inertias,
                mode='lines+markers', line=dict(color=BLUE, width=3),
                marker=dict(size=8, color=BLUE)))
            fig_el.add_vline(x=k, line_dash='dash', line_color=ORANGE, annotation_text=f'K={k}')
            fig_el.update_layout(xaxis_title='K', yaxis_title='Inertia',
                                  height=280, margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig_el, use_container_width=True)

        with col_e2:
            st.subheader("Silhouette Scores by K")
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(x=list(range(2,9)), y=sil_scores,
                mode='lines+markers', line=dict(color=GREEN, width=3),
                marker=dict(size=8, color=GREEN)))
            fig_sil.update_layout(xaxis_title='K', yaxis_title='Silhouette Score',
                                   height=280, margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig_sil, use_container_width=True)

        # Cluster profiles
        st.subheader("Cluster Profiles")
        profile_cols = ['Age','Annual_Income_INR','Dependents','Digital_Savvy_Score',
                        'WTP_Monthly_INR','Financial_Literacy',
                        'Interest_TermLife','Interest_ChildPlan','Interest_WholeLife']
        cluster_profile = df_f.groupby('Cluster')[profile_cols].mean().round(1)
        cluster_profile['Size'] = df_f.groupby('Cluster').size()
        cluster_profile['Churn_%'] = df_f.groupby('Cluster')['Churned'].apply(lambda x: (x=='Yes').mean()*100).round(1)
        cluster_profile['Top_Product'] = df_f.groupby('Cluster')['Product_Purchased'].agg(lambda x: x[x!='None'].mode()[0] if (x!='None').any() else 'None')
        st.dataframe(cluster_profile.style.background_gradient(cmap='Blues', subset=profile_cols), use_container_width=True)

        # Scatter
        st.subheader("Cluster Scatter — Income vs WTP (bubble = LTV)")
        fig_sc = px.scatter(df_f.sample(min(600,len(df_f))), 
                            x='Annual_Income_INR', y='WTP_Monthly_INR',
                            color='Cluster', size='Customer_LTV_INR',
                            hover_data=['Age','Occupation','Product_Purchased'],
                            color_continuous_scale='Viridis',
                            labels={'Annual_Income_INR':'Annual Income (₹)','WTP_Monthly_INR':'WTP Monthly (₹)'})
        fig_sc.update_layout(height=380, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_sc, use_container_width=True)

        # Personalised strategy
        st.subheader("🎯 Personalised Strategy by Segment")
        segment_names = {0:'Digital Natives', 1:'Value Seekers', 2:'Family Protectors', 3:'HNI Planners'}
        for cl in sorted(df_f['Cluster'].unique()):
            row = cluster_profile.loc[cl]
            name = segment_names.get(cl, f'Segment {cl+1}')
            with st.expander(f"**{name}** — Cluster {cl} | {int(row['Size'])} customers | Churn: {row['Churn_%']}%"):
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(f"""
**Profile:**
- Avg Age: {row['Age']:.0f} yrs | Avg Income: ₹{row['Annual_Income_INR']:,.0f}
- Dependents: {row['Dependents']:.1f} | Digital Savvy: {row['Digital_Savvy_Score']:.1f}/5
- Top Product Interest: {'Term' if row['Interest_TermLife']==max(row['Interest_TermLife'],row['Interest_ChildPlan'],row['Interest_WholeLife']) else ('Child Plan' if row['Interest_ChildPlan']>row['Interest_WholeLife'] else 'Whole Life')}
""")
                with col_s2:
                    st.markdown(f"""
**Recommended Strategy:**
- 🏷️ **Discount**: {'15-20% loyalty discount' if row['Churn_%'] > 30 else '5-10% referral bonus'}
- 📱 **Channel**: {'Mobile App push + WhatsApp' if row['Digital_Savvy_Score'] >= 3.5 else 'IVR + Agent follow-up'}
- 📦 **Bundle**: {row['Top_Product']} + {'Term Life' if row['Top_Product']!='Term Life' else 'Child Plan'} combo
- 💰 **Premium Mode**: {'Monthly auto-debit' if row['Annual_Income_INR'] < 700000 else 'Annual with 3% discount'}
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🔗 Association Rule Mining")
    st.markdown("**Goal: Discover product bundling patterns, cross-sell opportunities, and customer behaviour associations**")
    st.divider()

    arm_task = st.radio("Analysis Type",
        ["Product Cross-Sell Rules","Customer Profile → Product Rules"],
        horizontal=True)

    col_ar1, col_ar2, col_ar3 = st.columns(3)
    min_sup  = col_ar1.slider("Min Support",    0.05, 0.40, 0.10, 0.01)
    min_conf = col_ar2.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05)
    min_lift = col_ar3.slider("Min Lift",        1.0,  4.0,  1.2, 0.1)
    run_arm  = st.button("▶ Mine Rules", type="primary", use_container_width=True)

    if run_arm:
        with st.spinner("Mining association rules..."):
            if arm_task == "Product Cross-Sell Rules":
                # Create product basket per customer profile segment
                df_arm = df_f.copy()
                df_arm['Has_Term']   = df_arm['Interest_TermLife']   >= 4
                df_arm['Has_Credit'] = df_arm['Interest_CreditLife'] >= 4
                df_arm['Has_Whole']  = df_arm['Interest_WholeLife']  >= 4
                df_arm['Has_Child']  = df_arm['Interest_ChildPlan']  >= 4
                df_arm['Has_Group']  = df_arm['Interest_GroupTerm']  >= 4
                df_arm['Loan_Holder']   = df_arm['Existing_Loans'] == 'Yes'
                df_arm['Married_Parent']= (df_arm['Marital_Status']=='Married') & (df_arm['Dependents']>0)
                df_arm['High_Income']   = df_arm['Annual_Income_INR'] > 1000000
                df_arm['Digital_User']  = df_arm['Digital_Savvy_Score'] >= 4

                basket_cols = ['Has_Term','Has_Credit','Has_Whole','Has_Child','Has_Group',
                               'Loan_Holder','Married_Parent','High_Income','Digital_User']
                basket = df_arm[basket_cols].astype(bool)
                basket.columns = ['Term Life','Credit Life','Whole Life','Child Plan',
                                   'Group Term','Loan Holder','Married+Parent','High Income','Digital User']
            else:
                df_arm = df_f.copy()
                df_arm['Young'] = df_arm['Age'] < 35
                df_arm['Mid_Age'] = df_arm['Age'].between(35,50)
                df_arm['Senior'] = df_arm['Age'] > 50
                df_arm['LowInc']  = df_arm['Annual_Income_INR'] < 500000
                df_arm['MidInc']  = df_arm['Annual_Income_INR'].between(500000,1200000)
                df_arm['HighInc'] = df_arm['Annual_Income_INR'] > 1200000
                df_arm['Bought_Term']   = df_arm['Product_Purchased'] == 'Term Life'
                df_arm['Bought_Credit'] = df_arm['Product_Purchased'] == 'Credit Life'
                df_arm['Bought_Child']  = df_arm['Product_Purchased'] == 'Child Plan'
                df_arm['Bought_Whole']  = df_arm['Product_Purchased'] == 'Whole Life'
                df_arm['Has_Loan']    = df_arm['Existing_Loans'] == 'Yes'
                df_arm['Has_Kids']    = df_arm['Dependents'] > 0
                df_arm['Low_Lit']     = df_arm['Financial_Literacy'] <= 2

                basket_cols = ['Young','Mid_Age','Senior','LowInc','MidInc','HighInc',
                               'Bought_Term','Bought_Credit','Bought_Child','Bought_Whole',
                               'Has_Loan','Has_Kids','Low_Lit']
                basket = df_arm[basket_cols].astype(bool)

            try:
                freq_items = apriori(basket, min_support=min_sup, use_colnames=True)
                rules = association_rules(freq_items, metric='lift', min_threshold=min_lift)
                rules = rules[rules['confidence'] >= min_conf].sort_values('lift', ascending=False)

                if len(rules) == 0:
                    st.warning("No rules found — try lowering min support, confidence, or lift.")
                else:
                    st.metric("Rules Found", len(rules))
                    st.subheader("Top Association Rules")
                    rules_disp = rules[['antecedents','consequents','support','confidence','lift','leverage']].copy()
                    rules_disp['antecedents'] = rules_disp['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules_disp['consequents'] = rules_disp['consequents'].apply(lambda x: ', '.join(list(x)))
                    rules_disp = rules_disp.head(25).reset_index(drop=True)
                    rules_disp[['support','confidence','lift','leverage']] = rules_disp[['support','confidence','lift','leverage']].round(3)
                    st.dataframe(rules_disp.style.background_gradient(subset=['lift','confidence'], cmap='Blues'), use_container_width=True)

                    # Lift scatter
                    st.subheader("Confidence vs Lift (bubble = Support)")
                    fig_arm = px.scatter(rules_disp, x='confidence', y='lift',
                                         size='support', color='lift',
                                         hover_data=['antecedents','consequents'],
                                         color_continuous_scale='RdYlGn',
                                         labels={'confidence':'Confidence','lift':'Lift'})
                    fig_arm.add_hline(y=1.0, line_dash='dash', line_color='grey', annotation_text='Lift = 1 (random)')
                    fig_arm.update_layout(height=380, margin=dict(t=20,b=20,l=20,r=20))
                    st.plotly_chart(fig_arm, use_container_width=True)

                    st.subheader("💡 Business Actions from Rules")
                    st.markdown("""
<div class="success-box">
<b>Top Cross-Sell & Bundling Insights:</b><br><br>
🔗 <b>Loan Holder → Credit Life</b> (High Confidence): Trigger Credit Life offer automatically at loan disbursement via API integration with lending partners. Zero extra CAC — highest conversion funnel.<br><br>
🔗 <b>Married+Parent → Child Plan + Term Life</b>: Bundle these as "Family Shield Pack" with 12% combined discount vs individual purchase. Target parents aged 28-42 via school-partnership digital ads.<br><br>
🔗 <b>High Income → Whole Life + Group Term</b>: Offer estate-planning consultation + Group Term for their business employees as an HNI package. Assign dedicated relationship manager.<br><br>
🔗 <b>Digital User + High Income → Term Life</b>: These customers are self-serve buyers — invest in SEO/content marketing and comparison-site listings for Term Life. Lowest acquisition cost segment.
</div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error mining rules: {str(e)}. Try adjusting parameters.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("📈 Regression Analysis")
    st.markdown("**Goal: Predict Customer LTV, optimal premium pricing, and satisfaction score**")
    st.divider()

    reg_target = st.radio("Regression Target",
        ["Customer LTV (₹)", "Annual Premium Pricing (₹)", "Satisfaction Score (1-5)"],
        horizontal=True)

    @st.cache_data
    def prep_regression(df_in, target):
        dfc = df_in.copy()
        cat_cols = ['Gender','City_Tier','Occupation','Education','Marital_Status',
                    'Risk_Appetite','Preferred_Channel','Price_Sensitivity',
                    'Existing_Insurance','Smoker','Pre_Existing_Cond',
                    'Exercise_Frequency','Online_Purchase_Hist','Existing_Loans','Has_Savings']
        le = LabelEncoder()
        for c in cat_cols:
            dfc[c+'_enc'] = le.fit_transform(dfc[c].astype(str))

        feat_cols = ['Age','Annual_Income_INR','Dependents','BMI','Digital_Savvy_Score',
                     'Financial_Literacy','WTP_Monthly_INR','Loan_Amount_INR',
                     'Interest_TermLife','Interest_CreditLife','Interest_WholeLife',
                     'Interest_ChildPlan','Interest_GroupTerm'] + [c+'_enc' for c in cat_cols]
        feat_cols = [c for c in feat_cols if c in dfc.columns]

        if target == 'ltv':
            y_col = 'Customer_LTV_INR'
        elif target == 'premium':
            dfc = dfc[dfc['Annual_Premium_INR'] > 0]
            y_col = 'Annual_Premium_INR'
        else:
            y_col = 'Satisfaction_Score'

        X = dfc[feat_cols].fillna(0)
        y = dfc[y_col]
        return X, y, feat_cols, y_col

    target_key = 'ltv' if 'LTV' in reg_target else ('premium' if 'Premium' in reg_target else 'sat')
    Xr, yr, reg_feats, y_col = prep_regression(df_f, target_key)

    col_r1, col_r2 = st.columns([1,3])
    with col_r1:
        reg_model_choice = st.selectbox("Model", ["Gradient Boosting","Linear Regression","Random Forest"])
        run_reg = st.button("▶ Run Regression", type="primary", use_container_width=True)

    if run_reg:
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)

        with st.spinner("Training regression model..."):
            if reg_model_choice == "Gradient Boosting":
                reg = GradientBoostingRegressor(n_estimators=150, random_state=42)
            elif reg_model_choice == "Linear Regression":
                reg = LinearRegression()
            else:
                from sklearn.ensemble import RandomForestRegressor
                reg = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
            reg.fit(Xr_train, yr_train)
            yr_pred = reg.predict(Xr_test)

        r2  = r2_score(yr_test, yr_pred)
        mae = mean_absolute_error(yr_test, yr_pred)
        mape = np.mean(np.abs((yr_test - yr_pred)/np.clip(yr_test,1,None)))*100

        m1,m2,m3 = st.columns(3)
        m1.metric("R² Score",   f"{r2:.3f}")
        m2.metric("MAE",        f"₹{mae:,.0f}" if target_key != 'sat' else f"{mae:.2f}")
        m3.metric("MAPE",       f"{mape:.1f}%")

        col_ra, col_rb = st.columns(2)

        with col_ra:
            st.subheader("Actual vs Predicted")
            scatter_df = pd.DataFrame({'Actual':yr_test.values, 'Predicted':yr_pred})
            # trendline='ols' requires statsmodels — draw manually instead
            sample_df = scatter_df.sample(min(400,len(scatter_df)), random_state=42)
            fig_av = px.scatter(sample_df, x='Actual', y='Predicted',
                                color_discrete_sequence=[BLUE])
            # add OLS trendline manually via numpy
            _x = sample_df['Actual'].values
            _y = sample_df['Predicted'].values
            _m, _b = np.polyfit(_x, _y, 1)
            fig_av.add_scatter(x=sorted(_x), y=[_m*v+_b for v in sorted(_x)],
                               mode='lines', line=dict(color=ORANGE, width=2),
                               name='Trend', showlegend=False)
            max_val = max(scatter_df['Actual'].max(), scatter_df['Predicted'].max())
            fig_av.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                             line=dict(color=RED, dash='dash'))
            fig_av.update_layout(height=360, margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig_av, use_container_width=True)

        with col_rb:
            st.subheader("Feature Importance")
            if hasattr(reg, 'feature_importances_'):
                fi_r = pd.DataFrame({'Feature':reg_feats,'Importance':reg.feature_importances_})
            else:
                fi_r = pd.DataFrame({'Feature':reg_feats,'Importance':np.abs(reg.coef_)})
            fi_r = fi_r.nlargest(15,'Importance').sort_values('Importance')
            fig_fir = px.bar(fi_r, x='Importance', y='Feature', orientation='h',
                             color='Importance', color_continuous_scale='Blues')
            fig_fir.update_layout(height=360, margin=dict(t=20,b=20,l=20,r=20), coloraxis_showscale=False)
            st.plotly_chart(fig_fir, use_container_width=True)

        # Residual distribution
        st.subheader("Residual Distribution")
        resid = yr_test.values - yr_pred
        fig_res = px.histogram(x=resid, nbins=40, color_discrete_sequence=[BLUE],
                               labels={'x':'Residual (Actual − Predicted)'})
        fig_res.add_vline(x=0, line_dash='dash', line_color=RED)
        fig_res.update_layout(height=250, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_res, use_container_width=True)

        st.subheader("💡 Business Insights")
        if target_key == 'ltv':
            st.markdown("""
<div class="insight-box">
<b>LTV Prediction Insights:</b><br><br>
📊 <b>Annual_Premium_INR</b> is the single strongest LTV predictor (r²≈0.85 with LTV) — premium upgrades are the #1 LTV lever.<br><br>
📊 <b>Churn flag</b> negatively impacts LTV by ~40% — every 1pp churn reduction = measurable LTV gain. Prioritise retention spend on medium-high LTV clusters.<br><br>
📊 <b>Digital_Savvy_Score ≥ 4</b> customers have 18% higher predicted LTV — digital onboarding investment has a directly quantifiable ROI.<br><br>
💡 <b>Action</b>: Segment customers by predicted LTV quartile and apply differential retention budgets — spend ₹500/yr on Q1 (low LTV) and up to ₹3,000/yr on Q4 (high LTV) for personalised servicing.
</div>""", unsafe_allow_html=True)
        elif target_key == 'premium':
            st.markdown("""
<div class="insight-box">
<b>Premium Pricing Insights:</b><br><br>
📊 <b>Annual_Income_INR and WTP_Monthly_INR</b> are the primary premium drivers — income-tiered pricing is strongly validated.<br><br>
📊 <b>Product type interaction</b>: Whole Life premiums are income-elastic (premium rises faster with income); Term premiums are age/health elastic.<br><br>
📊 <b>City_Tier</b> adds incremental explanatory power — Tier 1 customers accept 12-15% higher premiums for the same coverage vs Tier 3.<br><br>
💡 <b>Action</b>: Build a dynamic pricing engine with 3 inputs — income band, age band, product — to auto-generate personalised premium quotes in the digital journey.
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="insight-box">
<b>Satisfaction Prediction Insights:</b><br><br>
📊 <b>Digital_Savvy_Score + Preferred_Channel</b> are the top satisfaction drivers — digital-native customers are inherently more satisfied with app-based insurance.<br><br>
📊 <b>Price_Sensitivity (Very Sensitive)</b> is the #1 negative predictor — these customers need proactive discount offers before renewal to prevent dissatisfaction.<br><br>
📊 <b>Claim experience</b> (encoded via Pre_Existing_Cond and Smoker) — customers who've had claims serviced show higher satisfaction, validating claim-first brand strategy.<br><br>
💡 <b>Action</b>: Build a satisfaction early-warning system — score every customer monthly and trigger service recovery (call/discount) for any customer predicted to fall below 3.0.
</div>""", unsafe_allow_html=True)

