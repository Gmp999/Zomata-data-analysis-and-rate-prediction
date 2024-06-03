import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import plotly.plotly as py
#import plotly.graph_objs as go

#py.offline.init_notebook_mode(connected=True)

#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('zomato.csv')
data.head()


# In[4]:


data.isna().sum()


# In[5]:


data.info()


# In[6]:


data=data[data.cuisines.isna()==False]


# In[7]:


data.isna().sum()


# In[8]:


data.info()


# In[9]:


data.drop(columns=["url", 'address','phone','listed_in(city)'], inplace  =True)


# In[10]:


data.rename(columns={'approx_cost(for two people)': 'average_cost'}, inplace=True)


# In[11]:


data.rename(columns={'listed_in(type)': 'listed_type'}, inplace=True)


# In[12]:


data.name.value_counts().head()


# In[13]:


plt.figure(figsize = (12,6))
ax = data.name.value_counts()[:20].plot(kind = 'bar')
ax.legend(['* Restaurants'])
plt.xlabel("Name of Restaurant")
plt.ylabel("Count of Restaurants")
plt.title("Name vs Number of Restaurant",fontsize =20, weight = 'bold')


# In[14]:


data.online_order.value_counts()


# In[15]:


ax = sns.countplot(x='online_order', data=data, hue='online_order')
plt.title('Number of Restaurants accepting online orders', weight='bold')
plt.xlabel('Online Orders')


# In[16]:


data['book_table'].value_counts()


# In[17]:


sns.countplot(x='book_table', data=data, palette="Set1", hue='book_table')
plt.title("No of Restaurant with Book Table Facility", weight='bold')
plt.xlabel('Book table facility')
plt.ylabel('No of restaurants')


# In[18]:


data['location'].value_counts()[:10]


# In[19]:


plt.figure(figsize=(12,6)) 
data['location'].value_counts()[:10].plot(kind = 'pie')
plt.title('Location', weight = 'bold')


# In[20]:


plt.figure(figsize = (12,6))
names = data['location'].value_counts()[:10].index
values = data['location'].value_counts()[:10].values
colors = ['gold', 'red', 'lightcoral', 'lightskyblue','blue','green','silver']
explode = (0.1, 0, 0, 0,0,0,0,0,0,0)  # explode 1st slice

plt.pie(values, explode=explode, labels=names, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title("Percentage of restaurants present in that location", weight = 'bold')
plt.show()


# In[21]:


plt.figure(figsize = (12,6))
data['location'].value_counts()[:10].plot(kind = 'bar', color = 'g')
plt.title("Location vs Count", weight = 'bold')


# In[22]:


data['location'].nunique()


# In[23]:


data['rest_type'].value_counts().head(10)


# In[24]:


plt.figure(figsize = (14,8))
data.rest_type.value_counts()[:15].plot(kind = 'pie')
plt.title('Restaurent Type', weight = 'bold')
plt.show()


# In[25]:


colors = ['#800080','red','#00FFFF','#FFFF00','#00FF00','#FF00FF']


# In[26]:


plt.figure(figsize = (12,6))
names = data['rest_type'].value_counts()[:6].index
values = data['rest_type'].value_counts()[:6].values
explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1)  # explode 1st slice

plt.title('Type of restaurant in percentage', weight = 'bold')
plt.pie(values, explode=explode, labels=names, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# In[27]:


data['average_cost'].value_counts()[:20]


# In[28]:


plt.figure(figsize = (12,8))
data['average_cost'].value_counts()[:20].plot(kind = 'pie')
plt.title('Avg cost in Restaurent for 2 people', weight = 'bold')
plt.show()


# In[29]:


colors  = ("red", "green", "orange", "cyan", "brown", "grey", "blue", "indigo", "beige", "yellow")


# In[30]:


fig= plt.figure(figsize=(18, 9))
explode = (0.1, 0, 0, 0,0,0,0,0,0,0) 

delplot = data['average_cost'].value_counts()[:10].plot(kind = 'pie',autopct='%1.1f%%',fontsize=20,shadow=True,explode = explode,colors = colors)

#draw circle
centre_circle = plt.Circle((0,0),0.80,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Average cost for 2 people in Rupees",fontsize = 15,weight = 'bold')


# In[31]:


dishes_data = data[data.dish_liked.notnull()]
dishes_data.dish_liked = dishes_data.dish_liked.apply(lambda x:x.lower().strip())


# In[32]:


dishes_data.isnull().sum()


# In[33]:


# count each dish to see how many times each dish repeated
dish_count = []
for i in dishes_data.dish_liked:
    for t in i.split(','):
        t = t.strip() # remove the white spaces to get accurate results
        dish_count.append(t)


# In[34]:


plt.figure(figsize=(12,6)) 
pd.Series(dish_count).value_counts()[:10].plot(kind='bar',color= 'c')
plt.title('Top 10 dished_liked in Bangalore',weight='bold')
plt.xlabel('Dish')
plt.ylabel('Count')


# In[35]:


data['rate'] = data['rate'].replace('NEW',np.NaN)
data['rate'] = data['rate'].replace('-',np.NaN)
data.dropna(how = 'any', inplace = True)


# In[36]:


data['rate'] = data.loc[:,'rate'].replace('[ ]','',regex = True)
data['rate'] = data['rate'].astype(str)
data['rate'] = data['rate'].apply(lambda r: r.replace('/5',''))
data['rate'] = data['rate'].apply(lambda r: float(r))


# In[37]:


plt.axvline(x= data.rate.mean(),ls='--',color='yellow')
plt.title('Average Rating for Bangalore Restaurants',weight='bold')
plt.xlabel('Rating')
plt.ylabel('No of Restaurants')
print(data.rate.mean())


# In[38]:


f,ax=plt.subplots(figsize=(18,8))
g = sns.pointplot(x=data["rest_type"], y=data["rate"], data=data)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title('Restaurent type vs Rate', weight = 'bold')
plt.show()


# In[39]:


#lets delete the nulll values
cuisines_data = data[data.cuisines.notnull()]
cuisines_data.cuisines = cuisines_data.cuisines.apply(lambda x:x.lower().strip())


# In[40]:


cuisines_count= []

for i in cuisines_data.cuisines:
    for j in i.split(','):
        j = j.strip()
        cuisines_count.append(j)


# In[41]:


plt.figure(figsize=(12,6)) 
pd.Series(cuisines_count).value_counts()[:10].plot(kind='bar',color= 'r')
plt.title('Top 10 cuisines in Bangalore',weight='bold')
plt.xlabel('cuisines type')
plt.ylabel('No of restaurants')


# In[42]:


plt.figure(figsize = (12,6))
sns.countplot(x=data['rate'], hue = data['online_order'])
plt.ylabel("Restaurants that Accept/Not Accepting online orders")
plt.title("rate vs oline order",weight = 'bold')


# In[43]:


data['online_order']= pd.get_dummies(data.online_order, drop_first=True)
data['book_table']= pd.get_dummies(data.book_table, drop_first=True)
data


# In[44]:


data.drop(columns=['dish_liked','reviews_list','menu_item','listed_type'], inplace  =True)


# In[45]:


data['rest_type'] = data['rest_type'].str.replace(',' , '') 
data['rest_type'] = data['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
data['rest_type'].value_counts().head()


# In[46]:


data['cuisines'] = data['cuisines'].str.replace(',' , '') 
data['cuisines'] = data['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
data['cuisines'].value_counts().head()


# In[47]:


from sklearn.preprocessing import LabelEncoder
T = LabelEncoder()                 
data['location'] = T.fit_transform(data['location'])
data['rest_type'] = T.fit_transform(data['rest_type'])
data['cuisines'] = T.fit_transform(data['cuisines'])
#data['dish_liked'] = T.fit_transform(data['dish_liked'].


# In[48]:


data["average_cost"] = data["average_cost"].str.replace(',' , '')


# In[49]:


data["average_cost"] = data["average_cost"].astype('float')


# In[50]:


data.head()


# In[51]:


x = data.drop(['rate','name'],axis = 1)


# In[52]:


y = data['rate']


# In[53]:


x.shape


# In[54]:


y.shape


# In[55]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 33)


# In[56]:


data.info()


# In[57]:


#standarizing
#taking numeric values
from sklearn.preprocessing import StandardScaler
num_values1=data.select_dtypes(['float64','int64']).columns
scaler = StandardScaler()
scaler.fit(data[num_values1])
data[num_values1]=scaler.transform(data[num_values1])


# In[58]:


data.head()


# In[59]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)


# In[60]:


lr.score(X_test, y_test)*100


# In[61]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_lr))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_lr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))


# In[62]:


from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)


# In[63]:


rfr.score(X_test,y_test)*100


# In[64]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rfr))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rfr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr)))


# In[65]:


####RIDGE
from sklearn.linear_model import Ridge
rdg = Ridge()
rdg.fit(X_train,y_train)
y_pred_rdg = rdg.predict(X_test)


# In[66]:


rdg.score(X_test,y_test)*100


# In[67]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rdg))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rdg))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rdg)))


# In[68]:


from sklearn.linear_model import Lasso
ls = Lasso()
ls.fit(X_train,y_train)
y_pred_ls = ls.predict(X_test)


# In[69]:


ls.score(X_test,y_test)*100


# In[70]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_ls))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_ls))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_ls)))


# In[71]:


from sklearn.svm import SVR
sv=SVR()
sv.fit(X_train,y_train)
y_pred_sv=sv.predict(X_test)


# In[72]:


sv.score(X_test,y_test)*100


# In[73]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_sv))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_sv))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_sv)))


# In[74]:


#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    knn = neighbors.KNeighborsRegressor(n_neighbors = K)

    knn.fit(X_train, y_train)  #fit the model
    y_pred_knn=knn.predict(X_test) #make prediction on test set


# In[75]:


knn.score(X_test,y_test)*100


# In[76]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_knn))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_knn))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_knn)))


# In[77]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge

# Load the California housing dataset
housing = fetch_california_housing()

# Split the data into features (X) and target (y)
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

# Create and fit the Bayesian Ridge model
nb = BayesianRidge()
nb.fit(X_train, y_train)

# Make predictions on test data
y_pred_nb = nb.predict(X_test)

# Evaluate the model's performance
r2 = r2_score(y_test, y_pred_nb)
print(f'R-squared score: {r2}')


# In[78]:


nb.score(X_test,y_test)*100


# In[79]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_nb))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_nb))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_nb)))


# In[80]:


onehot = pd.read_csv("zomato.csv")
onehot.head()


# In[81]:


onehot['rate'] = onehot['rate'].replace('NEW',np.NaN)
onehot['rate'] = onehot['rate'].replace('-',np.NaN)
onehot.dropna(how = 'any', inplace = True)

onehot['rate'] = onehot.loc[:,'rate'].replace('[ ]','',regex = True)
onehot['rate'] = onehot['rate'].astype(str)
onehot['rate'] = onehot['rate'].apply(lambda r: r.replace('/5',''))
onehot['rate'] = onehot['rate'].apply(lambda r: float(r))


# In[82]:


onehot['cuisines'] = onehot['cuisines'].str.replace(',' , '') 
onehot['cuisines'] = onehot['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
onehot['cuisines'].unique()


# In[83]:


onehot['rest_type'] = onehot['rest_type'].str.replace(',' , '') 
onehot['rest_type'] = onehot['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
onehot['rest_type'].value_counts().head()


# In[84]:


onehot['dish_liked'] = onehot['dish_liked'].str.replace(',' , '') 
onehot['dish_liked'] = onehot['dish_liked'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
onehot['dish_liked'].value_counts().head()


# In[85]:


dummy_rest_type=pd.get_dummies(onehot['rest_type'])
dummy_city=pd.get_dummies(onehot['location'])
dummy_cuisines=pd.get_dummies(onehot['cuisines'])
dummy_dishliked=pd.get_dummies(onehot['dish_liked'])


# In[86]:


final=pd.concat([onehot,dummy_rest_type,dummy_city,dummy_cuisines,dummy_dishliked],axis=1)


# In[87]:


final.drop(columns=['rest_type','location','cuisines','dish_liked','name','phone'] , inplace=True)
final.drop(columns=['reviews_list','menu_item','listed_in(type)','listed_in(city)'], inplace=True)
final.drop(columns=['url','address'], inplace=True)


# In[88]:


# Assuming 'online_order' and 'book_table' are the columns you want to one-hot encode
online_order_dummies = pd.get_dummies(final['online_order'], prefix='online_order')
book_table_dummies = pd.get_dummies(final['book_table'], prefix='book_table')

# Drop the original columns
final = final.drop(['online_order', 'book_table'], axis=1)

# Concatenate the one-hot encoded columns
final = pd.concat([final, online_order_dummies, book_table_dummies], axis=1)


# In[89]:


final['approx_cost(for two people)'] = final['approx_cost(for two people)'].str.replace(',' , '')


# In[90]:


x = final.drop(['rate'],axis=1)
y = final['rate']


# In[91]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 33)


# In[92]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)


# In[93]:


lr.score(X_test,y_test)*100


# In[94]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_lr))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_lr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))


# In[95]:


from sklearn.linear_model import Ridge
rdg = Ridge()
rdg.fit(X_train,y_train)
y_pred_rdg = rdg.predict(X_test)


# In[96]:


rdg.score(X_test,y_test)*100


# In[97]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rdg))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rdg))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rdg)))


# In[98]:


from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)


# In[99]:


rfr.score(X_test,y_test)*100


# In[100]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rfr))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rfr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr)))


# In[101]:


from sklearn.linear_model import Lasso
ls = Lasso()
ls.fit(X_train,y_train)
y_pred_ls = ls.predict(X_test)


# In[102]:


ls.score(X_test,y_test)*100


# In[103]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_ls))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_ls))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_ls)))


# In[104]:


Randpred = pd.DataFrame({ "actual": y_test, "pred": y_pred_rfr })
Randpred


# In[110]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

# Generate predictions using the Random Forest Regressor
preds_rf = rfr.predict(X_test)

# Plotting histogram for true values
plt.hist(y_test, bins=20, alpha=0.5, label="True Values", color="red")

# Plotting histogram for predicted values
plt.hist(preds_rf, bins=20, alpha=0.5, label="Predicted Values", color="green")

# Add title and labels
plt.title("True rate vs Predicted rate", size=20, pad=15)
plt.xlabel('Rating', size=15)
plt.ylabel('Frequency', size=15)

# Add legend
plt.legend()

# Display the plot
plt.show()





# In[106]:


data.to_csv('zum.csv',index=False)
