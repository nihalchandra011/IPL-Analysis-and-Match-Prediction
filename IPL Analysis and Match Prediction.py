#!/usr/bin/env python
# coding: utf-8

# # Indian Premier League Match Analysis and Prediction using Machine Learning

# In[1]:


#Importing the required libraries.
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[2]:


#Reading the two datasets stored as .csv files.
ipl_matches = pd.read_csv('matches.csv')
ipl_balls = pd.read_csv('deliveries.csv')

#Displaying the first five rows of the matches dataset.
ipl_matches.head()


# # Data Analysis

# In[3]:


#Match Results.
print("Number of Normal Matches                   : "+str(ipl_matches.shape[0] - ipl_matches[ipl_matches.result == 'tie'].id.count() - ipl_matches[ipl_matches.result == 'no result'].id.count() - ipl_matches[ipl_matches.dl_applied == 1].id.count()))
print("Number of Matches completed via DLS Method : "+ str(ipl_matches[ipl_matches.dl_applied == 1].id.count()))
print("Number of Tie Matches                      : "+str(ipl_matches[ipl_matches.result == 'tie'].id.count()))
print("Number of No Result Matches                : "+str(ipl_matches[ipl_matches.result == 'no result'].id.count()))


# From the above details, we can infer that out of a total of 756 matches:
# - There have been 19 matches where The Duckworth–Lewis–Stern(DLS) method has been implemented due to bad weather or lighting.
# - There have been 9 matches where the scores were level and a super over was used to decide the winning team.
# - There have been 4 washed-out games.

# In[4]:


#Importance of Winning the toss for winning a match.
print("No of matches where team winning the toss wins the match: "+str(ipl_matches[(ipl_matches.result == 'normal') & (ipl_matches.toss_winner == ipl_matches.winner)].id.count()))
print("No of matches where team winning the toss loses the match: "+str(ipl_matches[(ipl_matches.result == 'normal') & (ipl_matches.toss_winner != ipl_matches.winner)].id.count()))

#Visualization of the above using a Pie Chart.
l = ['Matches where team winning the toss wins the match', 'Matches where team winning the toss loses the match']
v = [ipl_matches[(ipl_matches.result == 'normal') & (ipl_matches.toss_winner == ipl_matches.winner)].id.count(), ipl_matches[(ipl_matches.result == 'normal') & (ipl_matches.toss_winner != ipl_matches.winner)].id.count()]

trace = go.Pie(labels = l, values = v, marker = dict(colors = ['orange' ,'black'], line = dict(color = "white", width =  1.3)), rotation = 90, hoverinfo = "label+value+text", hole = .5)
layout = go.Layout(dict(title = "Toss Winner vs Match Winner", plot_bgcolor  = "rgb(243,243,243)", paper_bgcolor = "rgb(243,243,243)"))

data = [trace]
figure = go.Figure(data = data,layout = layout)
py.iplot(figure) 


# The pie chart highlights that there is a higher chance that the team winning the toss wins the match, evident from the greater percentage share show in orange.

# In[5]:


#Venue-wise distribution of the above statistic.
fig = plt.figure() 
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
width = 0.35

ipl_matches[(ipl_matches.result == 'normal') & (ipl_matches.toss_winner == ipl_matches.winner)].groupby('venue')['venue'].count().plot(figsize = (20,10), kind = 'bar', color = 'orange', ax = ax, width = width, position = 1, xlabel = 'Venue', ylabel = 'Number of Matches', yticks = [0,5,10,15,20,25,30,35])
ipl_matches[(ipl_matches.result == 'normal') & (ipl_matches.toss_winner != ipl_matches.winner)].groupby('venue')['venue'].count().plot(figsize = (20,10), kind = 'bar', color = 'black', ax = ax2, width = width, position = 0, xlabel = 'Venue', ylabel = 'Number of Matches', yticks = [0,5,10,15,20,25,30,35])

plt.show()


# The above satistic gives us a good comparision between the number of matches won by the toss-winning team to the number of matches lost by the same for each venue. 
# The orange-colored bar shows the number of matches where the toss-winning team wins the match while the black-colored bar shows the number of matches where the toss-winning team loses the match.
# Few notable observations:
# - Venues like The Himachal Pradesh Cricket Association Stadium, Nehru Stadium and the Sheikh Zayed Stadium have a very positive corelation between winning the match and the team winning the toss. A captain might be dissapointed on losing the toss at such venues.
# - Venues like Eden Gardens, M Chinnaswamy Stadium, Rajiv Gandhi International Stadium and the Wankhade Stadium  have a stark negative corelation between winning the match and the team winning the toss. A captain might not be dissapointed on losing the toss at such venues.
# 
# Team Captains might factor-in these inferences while making toss decisions on the same grounds in the future.

# In[6]:


#Plot of Win percentage of each team throughout IPL
lab = ipl_matches["winner"].value_counts().keys().tolist()
val = ipl_matches["winner"].value_counts().values.tolist()

trace = go.Pie(labels = lab , values = val ,
               marker = dict(colors =  ['#000000', '#484337', '#5C5C5C', '#8B8B8B', '#8D8269', '#C5AF7E', '#A49345', '#BAA337', '#DAA53C', '#FFAA00', '#FFD000', '#FFE057', '#FDE475', '#FFF3BF', '#FBF6E2'],
               line = dict(color = "white", width =  1.3)), rotation = 120, hoverinfo = "label+value+text", hole = 0.5)

layout = go.Layout(dict(title = "Win Share of Each Team Throughout IPL", plot_bgcolor  = "rgb(243,243,243)", paper_bgcolor = "rgb(243,243,243)"))

data = [trace]
figure = go.Figure(data = data,layout = layout)
py.iplot(figure)


# We observe that Mumbai Indians have the highest win-precentage at 14.5% (also evident from their four IPL Titles), followed by Chennai Super Kings at 13.3% (Three IPL Titles) and then by Kolkata Knight Riders at 12.2% (Two IPL titles).

# In[7]:


#Players who were the Man of the Match for maximum times
#Top 10 man of the match bar graph
df1 = pd.DataFrame({"count":ipl_matches.groupby('player_of_match')['player_of_match'].count()}).reset_index()
df1 = df1.sort_values('count',ascending = False)
df1[0:10].plot.barh(figsize = (20,10), x = 'player_of_match', xlabel = 'Man of the Match', ylabel = 'Number of times', color = 'orange', fontsize = 15, xticks = [2,4,6,8,10,12,14,16,18,20])


# It is clear from the graph that Chris Gayle has won the Man of the Match Award most number of times (21), followed by AB De Villiers (20) and MS Dhoni and David Warner (17).

# In[8]:


#First five rows of the ball-by-ball score dataset.
ipl_balls.head()


# In[9]:


#Considering a specific batsman as an example - MS Dhoni.
#Snapshot of all the data related to MS Dhoni.

df_bats = ipl_balls[ipl_balls.batsman == 'MS Dhoni']
df_bats.head()


# In[10]:


#Teams under which MS Dhoni has played in the entire IPL and the corresponding number of matches played for that team.

df1 = pd.DataFrame({"count":df_bats.groupby(['batting_team','match_id'])['batting_team'].count()}).reset_index()
df1.groupby('batting_team')['batting_team'].count()


# We can observe that MS Dhoni has played 143 matches (10 seasons) as the Captain of Chennai Super Kings while a total of 27 matches under Rising Pune Supergiants (2 seasons).

# In[11]:


#Graph of runs scored by MS Dhoni in each match
df_bats.groupby('match_id')['batsman_runs'].sum().plot(figsize = (20,10), kind = 'bar', xlabel = 'Match ID', ylabel = 'Runs Scored', color = ['black','orange'])


# The above graph shows the variation in performance and form of MS Dhoni over time and the fluctuations in runs scored betweeen consecutive matches.

# In[12]:


#Comparison of different ways in which MS Dhoni has been dismissed.

df_bats[df_bats.player_dismissed == 'MS Dhoni'].groupby('dismissal_kind')['dismissal_kind'].count().plot(kind = 'bar', color = ['black','orange'], xlabel = 'Type of Dismissal', ylabel = 'Number of Times')


# The above statistic clearly highlights that MS Dhoni has got out most of the times by getting caught. This inference might help the bowler and captain of the opposite team to decide the kind of ball to be bowled and the suitable field placement.

# In[13]:


#Effectiveness of different bowlers against MS Dhoni.
ipl_balls[ipl_balls.player_dismissed == 'MS Dhoni'].groupby('bowler')['bowler'].count().plot(figsize = (20,10), kind = 'bar', color = ['black','orange'])


# We observe that two bolwers in particular, Zaheer Khan and Pragyan Ojha have proved to be most effective against MS Dhoni and hence the captain of the opposite team might make these bowlers bowl (if they're available) in the crucial overs of the match.

# In[14]:


#All time Purple Cap Holders - Top 10.

df1 = pd.DataFrame({'wickets':ipl_balls[ipl_balls.player_dismissed.notna()].groupby('bowler')['bowler'].count()}).reset_index()
df1= df1.sort_values('wickets',ascending = False)

df1[0:10].groupby('bowler')['wickets'].sum().plot(kind = 'bar',figsize = (20,10), color = ['black','orange'], fontsize = 15, xlabel = 'Bowler', ylabel = 'Number of Wickets')


# This bar ghraph brings out the fact that Lasith Malinga is the all-time Purple Cap holder (188 wickets), followed by Dwyane Bravo (168 wickets) and Amit Mishra (165 wickets).

# In[15]:


#Comparison of different ways in which Lasith Malinga has taken wicketss.

ipl_balls[ipl_balls.bowler == 'SL Malinga'].groupby('dismissal_kind')['dismissal_kind'].count().plot(kind = 'bar', xlabel = 'Type of Dismissal', ylabel = 'Number of times', fontsize = 12, color = ['black', 'orange'])


# We can observe from the above graph that Malinga has taken most wickets via a catch or bowled out the batsman. Again, of critical importance to cricket analysts and the team managememmt.

# # Data Preprocessing

# In[16]:


#Removing irrelavent attributes from the dataset.
ipl_matches.drop(columns=['player_of_match', 'dl_applied','result', 'umpire1','umpire2','date','season','id'], inplace=True)

#Removing missing values from the dataset.
ipl_matches.dropna(axis=0,subset=['winner'],inplace=True)
ipl_matches.head()


# In[17]:


#Brief Data Overview.
#No of rows
print ("Rows      : " ,ipl_matches.shape[0])

#No of columns
print ("Columns   : " ,ipl_matches.shape[1])

#Features/Attributes used to predict the winning team
print ("\nFeatures: " ,ipl_matches.columns.tolist())

#Count of missing valuews (if any)
print ("\nMissing values:  ",ipl_matches.isnull().sum().values.sum())

#Count of unique values of each feature
print ("\nUnique values :  \n",ipl_matches.nunique())


# In[18]:


#Conversion of catagorical to numerical values using Label Encoding.
#Team Names and their corresponding numerical values.
#0 - Chennai Super Kings
#1 - Deccan Chargers
#2 - Delhi Capitals
#3 - Delhi Daredevils
#4 - Gujrat Lions
#5 - Kings XI Punjab
#6 - Kochi Tuskers Kerala
#7 - Kolkata Knight Riders
#8 - Mumbai Indians
#9 - Pune Warriors
#10 - Rajasthan Royals
#11 - Rising Pune Supergiants
#12 - Royal Challengers Bangalore
#13 - Sunrisers Hyderabad

#Toss Result and their corresponding numerical values.
#0 - Bat first
#1 - Field first

#Preparation for label Encoding.
#Converting object datatype to catagory.
ipl_matches['team1']=ipl_matches['team1'].astype('category')
ipl_matches['team2']=ipl_matches['team2'].astype('category')
ipl_matches['winner']=ipl_matches['winner'].astype('category')
ipl_matches['venue']=ipl_matches['venue'].astype('category')
ipl_matches['toss_winner']=ipl_matches['toss_winner'].astype('category')
ipl_matches['toss_decision']=ipl_matches['toss_decision'].astype('category')

#Label Encoding.
ipl_matches['team1']=ipl_matches['team1'].cat.codes
ipl_matches['team2']=ipl_matches['team2'].cat.codes
ipl_matches['winner']=ipl_matches['winner'].cat.codes
ipl_matches['venue']=ipl_matches['venue'].cat.codes
ipl_matches['toss_winner']=ipl_matches['toss_winner'].cat.codes
ipl_matches['toss_decision']=ipl_matches['toss_decision'].cat.codes

#Dataset after conversion.
ipl_matches.head()


# In[19]:


#Correlation Matrix
correlation = ipl_matches.corr()

#Tick Labels
matrix_cols = correlation.columns.tolist()
corr_array  = np.array(correlation)

#Plotting using Heatmap
trace = go.Heatmap(z = corr_array, x = matrix_cols, y = matrix_cols,
                   colorscale=[[0, 'rgb(254, 147, 2)'], [1, 'rgb(0,0,0)']], colorbar = dict(title = "Pearson Correlation coefficient",
                   titleside = "right"))

layout = go.Layout(dict(title = "Correlation Matrix for variables",
                        autosize = False, height = 720, width = 800,
                        margin = dict(r = 0 ,l = 210,
                                       t = 25,b = 210),
                        yaxis = dict(tickfont = dict(size = 16)), xaxis = dict(tickfont = dict(size = 16))))

data = [trace]
fig = go.Figure(data=data, layout = layout)
py.iplot(fig)


# <b>Interpreting the Correlation Matrix</b><br>
# A correlation matrix is an effectie tool for figuring out patterns and relations between different features of the dataset.
# Each cell in the matrix has is represented by a value called the Pearson correlation coefficient (z score), which is a measure of the linear association between the two variables. It has a value between -1 and 1 where:
# - -1 indicates a perfectly negative linear correlation between the two variables
# - 0 indicates no linear correlation between two variables
# - 1 indicates a perfectly positive linear correlation between the two variables<br>
# 
# The further away the correlation coefficient is from zero, the stronger the relationship between the two variables.<br> 
# The cells are also represented by different shades of colors based on the Pearson Coefficient.<br>
#  
# Further from the above matrix, we discover that:
# - There is a positive correlation between team2 and toss_winner. And since we have already seen the positive correlation between the toss_winner and match winner, it can be inferred that team2 has been the won the match more number of times (evident from the corresponding positive z-score).
# - There is a strong negative correlation between win_by_runs and win_by_wickets due to the fact that in a match, a team either wins by runs or by wickets but never both.

# # Decision Tree Implementation

# In[20]:


#Assigning attributes to X and Y variables. 
feature_cols = [i for i in ipl_matches.columns]
X = ipl_matches.iloc[:,[0,1,2,3,4,6,7]].values
y = ipl_matches.iloc[:,5].values


# In[21]:


#Splitting the dataset into training and testing set (80:20).
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[22]:


#Feature Scaling.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[23]:


#Applying the Decision Tree Classifier.
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train,y_train)


# In[24]:


#Prediction.
y_pred = classifier.predict(X_test)

#Accuracy.
print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))

#f1_score.
print('F1 Score:', metrics.f1_score(y_test,y_pred, average='weighted'))


# We find that the Decision algorithm gives us a high accuracy and f-1 score, which implies that the classifier has correctly classified 92% of the times and has a very low misclassification error.

# In[ ]:




