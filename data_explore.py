import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Load the data
data = pd.read_csv("Instagram.csv", encoding='latin1')
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop NaN values
data = data.dropna()

# Check data info
data.info()

# Plot distributions
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'], kde=True)
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'], kde=True)
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.histplot(data['From Explore'], kde=True)
plt.show()

# Summarize impressions
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

fig = px.pie(values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()

# WordCloud for Captions
data['Caption'] = data['Caption'].fillna('')
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# WordCloud for Hashtags
data['Hashtags'] = data['Hashtags'].fillna('')
text = " ".join(i for i in data.Hashtags)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Analyzing relationships
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title="Relationship Between Likes and Impressions")
figure.show()

figure = px.scatter(data_frame=data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title="Relationship Between Comments and Total Impressions")
figure.show()

figure = px.scatter(data_frame=data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title="Relationship Between Shares and Total Impressions")
figure.show()

figure = px.scatter(data_frame=data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title="Relationship Between Post Saves and Total Impressions")
figure.show()

# Calculate correlation only on numeric columns
numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
correlation = numeric_data.corr()
print(correlation["Impressions"].sort_values(ascending=False))

# Calculate conversion rate
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print("Conversion Rate:", conversion_rate)

# Relationship between Profile Visits and Follows
figure = px.scatter(data_frame=data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title="Relationship Between Profile Visits and Followers Gained")
figure.show()

# Prepare data for modeling
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

# Model training
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
print("Model Score:", model.score(xtest, ytest))

# Prediction example
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
predicted_impressions = model.predict(features)
print("Predicted Impressions:", predicted_impressions[0])