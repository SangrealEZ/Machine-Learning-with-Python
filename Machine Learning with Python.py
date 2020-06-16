# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[114]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[115]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[116]:


df = pd.read_csv('loan_train.csv')
df.read()


# In[117]:


df.shape


# ### Convert to date time object 

# In[118]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[119]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[120]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[121]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[122]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[123]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[124]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[125]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[126]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[127]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[128]:


df[['Principal','terms','age','Gender','education']].head()

#### Use one hot encoding technique to convert categorical varables to binary variables and append them to the feature Data Frame 
# In[129]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[130]:


mydata = Feature
mydata[0:5]


# What are our labels?

# In[131]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[132]:


mydata = preprocessing.StandardScaler().fit(mydata).transform(mydata)
mydata[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[133]:


# split dataset into train + test data
from sklearn.model_selection import train_test_split
mydata_train, mydata_test, y_train, y_test = train_test_split(mydata,y, test_size = 0.2, random_state = 123)
print('Train data:', mydata_train.shape, y_train.shape)
print('Test data:',mydata_test.shape, y_test.shape)


# In[134]:


# import KNN library
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[135]:


# find k (below 10)
k = 10
mean_acc = np.zeros((k-1))
std_acc = np.zeros((k-1))
confusionMatrix = []
for n in range(1,k):
    neighbors = KNeighborsClassifier(n_neighbors = n).fit(mydata_train, y_train)
    yhat = neighbors.predict(mydata_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)
    std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[136]:


# plot out 
plt.plot(range(1,k), mean_acc, 'g')
plt.fill_between(range(1,k),mean_acc-1*std_acc, mean_acc+1*std_acc, alpha = 0.1)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (k)')
plt.tight_layout()
plt.show()


# In[137]:


# looks like accuracy reaches max when k = 4, and then it starts to decline
# setting k = 4 for modeling
print( "The best accuracy was with", mean_acc.max().round(4), "with k=", mean_acc.argmax()+1) 


# In[138]:


plt.plot(range(1,k),mean_acc,'g')
plt.fill_between(range(1,k),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[139]:


# train model and predict
k = 4
KNN_model = KNeighborsClassifier(n_neighbors = k).fit(mydata_train, y_train)
KNN_model


# In[140]:


# predicting
yhat = KNN_model.predict(mydata_test)
yhat[0:5]


# In[141]:


# print out accuracy
print('KNN Training set accuracy:', metrics.accuracy_score(y_train, KNN_model.predict(mydata_train)))
print('KNN Test set accuracy:', metrics.accuracy_score(y_test, KNN_model.predict(mydata_test)))


# # Decision Tree

# In[142]:


# import Decision Tree library
from sklearn.tree import DecisionTreeClassifier


# In[143]:


# re-split data into train and test
mydata_train, mydata_test, y_train, y_test = train_test_split(mydata, y, test_size = 0.2, random_state = 123)


# In[144]:


# build the Decision Tree classifier
tree_model = DecisionTreeClassifier(criterion = 'entropy', max_depth= 4)
tree_model


# In[145]:


# fit the training dataset
tree_model.fit(mydata_train, y_train)


# In[146]:


# predict on test set and print out result
tree_pred = tree_model.predict(mydata_test)
print (tree_pred [0:5])
print (y_test [0:5])


# In[147]:


# evaluate accuracty
import matplotlib.pyplot as plt
print('Decistion Tree Classifier Accuracy: ', metrics.accuracy_score(y_test, tree_pred))


# # Support Vector Machine

# In[148]:


# re-split data into train and test set
mydata_train, mydata_test, y_train, y_test = train_test_split(mydata, y, test_size = 0.2, random_state = 123)


# In[149]:


# import library
from sklearn import svm


# In[150]:


# build model
svm_model = svm.SVC(kernel = 'rbf')
svm_model.fit (mydata_train, y_train)


# In[151]:


# predict new values
yhat = svm_model.predict(mydata_test)
yhat [0:5]


# In[152]:


# evaluate model
# import library
from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[153]:


# create a function to build confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[154]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=['PAIDOFF','COLLECTION'])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['PAIDOFF','COLLECTION'],normalize= False,  title='Confusion matrix')


# In[155]:


# calculate f1 score
from sklearn.metrics import f1_score
f1_score = f1_score(y_test, yhat, average='weighted') 
print('F1 Score: ',f1_score)


# In[156]:


# calculate Jaccard Index
from sklearn.metrics import jaccard_similarity_score
jaccard_index = jaccard_similarity_score(y_test, yhat)
print('Jaccard Index: ', jaccard_index)


# # Logistic Regression

# In[157]:


# re-split data into train and test set
mydata_train, mydata_test, y_train, y_test = train_test_split(mydata, y, test_size = 0.2, random_state = 123)


# In[158]:


# import library & build logistic model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
logistic_model = LogisticRegression(C = 0.01, solver = 'liblinear').fit(mydata_train, y_train)
logistic_model


# In[159]:


# predict values
yhat = logistic_model.predict(mydata_test)
yhat [0:5]


# In[160]:


# check probability
yhat_prob = logistic_model.predict_proba(mydata_test)
yhat_prob[0:5]


# In[161]:


# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=['PAIDOFF','COLLECTION'])
np.set_printoptions(precision=2)

# plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['PAIDOFF','COLLECTION'],normalize= False,  title='Confusion matrix')
print (classification_report(y_test, yhat))


# In[162]:


# calculate Jaccard score
print('Jaccard Index: ',jaccard_similarity_score(y_test, yhat))


# In[163]:


# calculate log loss
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# # Model Evaluation using Test set

# In[164]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[165]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[166]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[167]:


# pre-processing test set
# convert to date-time object
test_df['effective_date'] = pd.to_datetime(df['effective_date'])
test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df.head()


# In[168]:


# create a binary object to identify weekend
test_df['dayofweek'] = df['effective_date'].dt.dayofweek
test_df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3) else 0)
test_df.head()


# In[169]:


# convert categorical variables to numerical
# Gender: male = 0, female = 1
test_df['Gender'].replace(to_replace = ['male','female'], value = [0,1],inplace = True)
test_df.head()


# In[170]:


# select features
test_df_feature = test_df[['Principal','terms','age','Gender','weekend']]
# one hot encoding for Education
test_df_feature = pd.concat([test_df_feature,pd.get_dummies(test_df['education'])],axis = 1)
# drop 'Master or Above'
test_df_feature.drop(['Master or Above'], axis = 1, inplace = True)
test_df_feature.head()


# In[171]:


# define feature set and y
test_x = test_df_feature
test_y = test_df['loan_status'].values
test_y[0:5]


# In[172]:


# normalize data
test_x = preprocessing.StandardScaler().fit(test_x).transform(test_x)
test_x[0:5]


# In[173]:


# evaluate KNN
yhat_KNN = KNN_model.predict(test_x)
# print out accuracy
jac_KNN = jaccard_similarity_score(test_y, yhat_KNN).round(4)
f1_KNN = f1_score(test_y, yhat_KNN, average = 'weighted').round(4)
log_loss_KNN = 'NA'


# In[174]:


# evaluate Decision Tree
yhat_tree = tree_model.predict(test_x)
# print out accuracy
jac_tree = jaccard_similarity_score(test_y, yhat_tree).round(4)
f1_tree =  f1_score(test_y,yhat_tree,average = 'weighted').round(4)
log_loss_tree = 'NA'


# In[175]:


# evaluate SVM
yhat_svm = svm_model.predict(test_x)
# print out accuracy
jac_svm = jaccard_similarity_score(test_y, yhat_svm).round(4)
f1_svm = f1_score(test_y, yhat_svm, average = 'weighted').round(4)
log_loss_svm = 'NA'


# In[176]:


# evaluate logistics
yhat_log = logistic_model.predict(test_x)
yhat_prob = logistic_model.predict_proba(test_x)
# print out accuracy
jac_log = jaccard_similarity_score(test_y, yhat_log).round(4)
f1_log =  f1_score(test_y,yhat_log, average = 'weighted').round(4)
log_loss_log = log_loss(test_y,yhat_prob).round(4)


# In[177]:


# build a data frame to report model accuracy
accuracy_matrix = pd.DataFrame({'Algorithm':['KNN','Decision Tree','SVM','Logistic Regression'],
    'Jaccard':[jac_KNN, jac_tree, jac_svm, jac_log],
    'F-1 Score': [f1_KNN, f1_tree, f1_svm, f1_log],
    'Log Loss':[log_loss_KNN, log_loss_tree, log_loss_svm, log_loss_log]})
accuracy_matrix


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
