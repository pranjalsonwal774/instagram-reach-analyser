import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Streamlit page config
st.set_page_config(page_title="Instagram Reach Analyzer", layout="wide")
st.title("📊 Instagram Reach Analyzer")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Instagram.csv", encoding='latin1')
    data.dropna(inplace=True)
    return data

data = load_data()


# Missing data check
if data.isnull().sum().sum() > 0:
    st.warning("There are missing values in the dataset.")

# Sidebar
st.sidebar.header("Options")
show_distributions = st.sidebar.checkbox("Show Impression Distributions")
show_wordclouds = st.sidebar.checkbox("Show WordClouds")
show_relationships = st.sidebar.checkbox("Show Relationships")
show_model = st.sidebar.checkbox("Run ML Model for Impressions Prediction")

# Impressions Distribution
if show_distributions:
    st.subheader("📈 Distributions of Impressions")
    for col in ['From Home', 'From Hashtags', 'From Explore']:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of Impressions From {col.split()[-1]}")
        st.pyplot(fig)

    st.subheader("🧁 Impressions by Source (Pie Chart)")
    home = data["From Home"].sum()
    hashtags = data["From Hashtags"].sum()
    explore = data["From Explore"].sum()
    other = data["From Other"].sum()
    labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
    values = [home, hashtags, explore, other]
    fig = px.pie(values=values, names=labels, title='Sources of Impressions', hole=0.5)
    st.plotly_chart(fig)

# WordClouds
if show_wordclouds:
    st.subheader("☁️ WordClouds")

    caption_text = " ".join(i for i in data.Caption.fillna(''))
    hashtag_text = " ".join(i for i in data.Hashtags.fillna(''))
    stopwords = set(STOPWORDS)

    wc1 = WordCloud(stopwords=stopwords, background_color="white").generate(caption_text)
    wc2 = WordCloud(stopwords=stopwords, background_color="white").generate(hashtag_text)

    col1, col2 = st.columns(2)
    with col1:
        st.image(wc1.to_array(), caption="Caption WordCloud", use_container_width=True)

    with col2:
        st.image(wc2.to_array(), caption="Hashtag WordCloud", use_container_width=True)

# Relationships
if show_relationships:
    st.subheader("🔗 Relationships Between Metrics")

    metrics = [("Likes", "Likes"), ("Comments", "Comments"), ("Shares", "Shares"), ("Saves", "Saves")]
    for label, col in metrics:
        fig = px.scatter(data_frame=data, x="Impressions", y=col, size=col,
                         trendline="ols", title=f"Impressions vs {label}")
        st.plotly_chart(fig)

    fig = px.scatter(data_frame=data, x="Profile Visits", y="Follows", size="Follows",
                     trendline="ols", title="Profile Visits vs Followers")
    st.plotly_chart(fig)

    st.subheader("📊 Correlation with Impressions")
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()["Impressions"].sort_values(ascending=False)
    st.write(corr)

    st.subheader("🔁 Conversion Rate")
    conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
    st.metric(label="Conversion Rate (Follows/Profile Visits)", value=f"{conversion_rate:.2f}%")

# ML Modeling
# ================================
# Train Model Immediately (Global)
# ================================
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model_score = model.score(xtest, ytest)

# ================================
# Home Page ML Prediction Section
# ================================
st.subheader("🧠 Predict Impressions Based on Your Engagement")
st.caption(f"Model Score on Test Set: {model_score:.2f}")

likes = st.number_input("Likes", value=100, key="likes_input")
saves = st.number_input("Saves", value=50, key="saves_input")
comments = st.number_input("Comments", value=10, key="comments_input")
shares = st.number_input("Shares", value=5, key="shares_input")
profile_visits = st.number_input("Profile Visits", value=200, key="visits_input")
follows = st.number_input("Follows", value=60, key="follows_input")

if st.button("Predict Impressions", key="predict_button"):
    input_data = np.array([[likes, saves, comments, shares, profile_visits, follows]])
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Impressions: {int(prediction):,}")

    # Analysis
    st.subheader("📉 Post & Account Reach Analysis")
    avg_impressions = int(data["Impressions"].mean())
    delta = prediction - avg_impressions

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Average Impressions", value=f"{avg_impressions:,}")
    with col2:
        st.metric(label="Difference from Average", value=f"{int(delta):,}", delta=f"{delta:.2f}")

    st.markdown("### 🤔 Insights")
    if prediction < avg_impressions * 0.8:
        st.error("Your post is predicted to underperform. Try improving hashtags, captions, or visuals.")
    elif prediction > avg_impressions * 1.2:
        st.success("Nice! Your post is likely to outperform average content.")
    else:
        st.info("Performance is average. A few improvements could make a difference.")

    insights = []
    if likes < data['Likes'].mean():
        insights.append("🔻 Likes are below average.")
    if saves < data['Saves'].mean():
        insights.append("🔖 Encourage users to save your post.")
    if comments < data['Comments'].mean():
        insights.append("💬 Ask a question to boost comments.")
    if shares < data['Shares'].mean():
        insights.append("📤 Try content that prompts sharing.")
    if profile_visits < data['Profile Visits'].mean():
        insights.append("👀 Improve bio or link placement.")
    if follows < data['Follows'].mean():
        insights.append("➕ Add a clear call-to-action to follow.")

    if insights:
        st.markdown("### 📌 Suggestions to Improve Reach:")
        for tip in insights:
            st.write(tip)
    else:
        st.success("🎉 All engagement metrics are strong!")
