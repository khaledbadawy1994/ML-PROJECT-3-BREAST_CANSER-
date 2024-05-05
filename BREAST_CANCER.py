# ML-PROJECT-3-BREAST_CANSER-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv("/content/drive/MyDrive/dataBREAST_CANSER.csv")
data.shape

data.info()

data.diagnosis.value_counts()

data.head(10)

data.columns

data.describe()

data.isnull().sum()

data.diagnosis.unique()

data.columns

data['Unnamed: 32'].unique()

data['diagnosis'].value_counts(normalize=True)

data['diagnosis'].value_counts()

data = data.drop(['id','Unnamed: 32'], axis= 1)

data.columns

data1=data.drop(['diagnosis'], axis=1)

data1.skew()

#The skew result show a positive (right) or negative (left) skew. Values closer to zero show less skew. From the graphs, we can see that radius_mean, perimeter_mean, area_mean, concavity_mean and concave_points_mean are useful in predicting cancer type due to the distinct grouping between malignant and benign cancer types in these features. We can also see that area_worst and perimeter_worst are also quite useful.

#Approach: We will first select the independent features (X) that are highly (positively or negatively) correlated with the dependent feature (y), and then drop the the common coutliers of these highly correlated features.¶

corr = data1.corr()
plt.figure(figsize = (20,20))
sns.heatmap(corr,annot = True,
            linewidth = 0.5,
            cmap = "BuPu")

#Discussion: Observe the diagnosis_M feature(which is our target) and note that features with the highest correlation with diagnosis_M lie between 0.7 and 0.8, we will select these features (as they are having the most impact) and find the the recurring outlier using percentile method.

max_thresold2 = data['perimeter_mean'].quantile(0.99)
max_thresold2
data[data['perimeter_mean']>max_thresold2]

min_thresold2 = data['perimeter_mean'].quantile(0.01)
min_thresold2
data[data['perimeter_mean']<min_thresold2]

max_thresold3 = data['area_mean'].quantile(0.99)
max_thresold3
data[data['area_mean']>max_thresold3]

min_thresold3 = data['area_mean'].quantile(0.01)
min_thresold3
data[data['area_mean']<min_thresold3]

max_thresold = data['radius_worst'].quantile(0.99)
max_thresold
data[data['radius_worst']>max_thresold]

min_thresold = data['radius_worst'].quantile(0.01)
min_thresold
data[data['radius_worst']<min_thresold]

max_thresold5 = data['perimeter_worst'].quantile(0.99)
max_thresold5
data[data['perimeter_worst']>max_thresold5]

min_thresold5 = data['perimeter_worst'].quantile(0.01)
min_thresold5
data[data['perimeter_worst']<min_thresold5]

max_thresold6 = data['area_worst'].quantile(0.99)
max_thresold6
data[data['area_worst']>max_thresold6]

min_thresold10 = data['area_worst'].quantile(0.01)
min_thresold10
data[data['area_worst']<min_thresold10]

#Discussion: So, what we just did from cell 28 till 44? We are getting top 1% and bottom 1% of the features that are highly correlated with our output, we observe that the values with row# 82, 180, 352, and 461 are repeatedly coming as outlier in the top 1% and the values with row# 46, 101, 151, 538, 539 and 568 are repeatedly coming as outlier in the bottom 1%, it would be better to drop these rows.

data = data.drop(data.index[[46,82,101,151,180,352,461,538,539,568]])

target = data.diagnosis
input_col = data.iloc[:,1:]

fig = plt.figure(figsize=(12,18))
for i in range(len(input_col.columns)):
    fig.add_subplot(9,4,i+1)
    sns.distplot(input_col.iloc[:,i], kde=True, hist=True)
    plt.xlabel(input_col.columns[i])
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12,18))
for i in range(len(input_col.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=input_col.iloc[:,i])
    plt.xlabel(input_col.columns[i])
plt.tight_layout()
plt.show()

plt.figure(figsize = (20, 15))
sns.set(style="darkgrid")
plotnumber = 1

for column in data:
    if plotnumber <=13 :
        ax = plt.subplot(5, 6, plotnumber)
        sns.histplot(data[column],kde=True)
        plt.xlabel(column)

    plotnumber += 1
plt.show()

plt.figure(figsize = (20, 20))
plotnumber = 1

for column in data:
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.boxplot(x=data[column])
        plt.xlabel(column)

    plotnumber += 1
plt.title("Distribution")
plt.show()

area = data[['area_mean','area_se','area_worst','diagnosis']]
sns.pairplot(area, hue='diagnosis', markers=["o", "s"])

perimeter = data[['perimeter_mean','perimeter_se','perimeter_worst','diagnosis']]
sns.pairplot(perimeter, hue='diagnosis', markers=["o", "s"])

texture = data[['texture_mean','texture_se','texture_worst','diagnosis']]
sns.pairplot(texture, hue='diagnosis', markers=["o", "s"])

compactness = data[['compactness_mean','compactness_se','compactness_worst','diagnosis']]
sns.pairplot(compactness, hue='diagnosis', markers=["o", "s"])

concavity = data[['concavity_mean','concavity_se','concavity_worst','diagnosis']]
sns.pairplot(concavity, hue='diagnosis', markers=["o", "s"])

symmetry = data[['symmetry_mean','symmetry_se','symmetry_worst','diagnosis']]
sns.pairplot(symmetry, hue='diagnosis', markers=["o", "s"])

fractal_dimension = data[['fractal_dimension_mean','fractal_dimension_se','fractal_dimension_worst','diagnosis']]
sns.pairplot(fractal_dimension, hue='diagnosis', markers=["o", "s"])

smoothness = data[['smoothness_mean','smoothness_se','smoothness_worst','diagnosis']]
sns.pairplot(smoothness, hue='diagnosis', markers=["o", "s"])

plt.figure(figsize = (20,10))
sns.set_theme(style="darkgrid")

radius = data[['radius_mean','radius_se','radius_worst','diagnosis']]
sns.pairplot(radius, hue='diagnosis', markers=["o", "s"])

y=data.diagnosis

ax = sns.countplot(y)
Benign, Malignant = y.value_counts(normalize = True)
print(f'The percentage of Benign case is : {Benign*100}\n\n')
print(f'The percentage of Malignant case is : {Malignant*100}\n\n')

y=pd.DataFrame(y)
y

# Visualizing class distribution
class_counts = data['diagnosis'].value_counts()

# Plotting a pie chart
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=['#FF9999', '#66B2FF'])
plt.title('Distribution of Diagnosis')
plt.show()

# Counting observations in each class
benign_count, malignant_count = data['diagnosis'].value_counts()
print('Count of cells labeled as Benign: ', benign_count)
print('Count of cells labeled as Malignant: ', malignant_count)
print('')
print('Percentage of cells labeled Benign: ', round(benign_count / len(data) * 100, 2), '%')
print('Percentage of cells labeled Malignant: ', round(malignant_count / len(data) * 100, 2), '%')

#Visualizing Multidimensional Relationships
plt.style.use('fivethirtyeight')
sns.set_style("white")
sns.pairplot(data[[data.columns[0], data.columns[1],data.columns[2],data.columns[3],
                     data.columns[4], data.columns[5]]], hue = 'diagnosis' , size=3)

#create the correlation matrix heat map
plt.figure(figsize=(10,6))
sns.heatmap(data[[data1.columns[0], data1.columns[1],data1.columns[2],data1.columns[3],
                     data1.columns[4], data1.columns[5]]].corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
plt.suptitle('Correlation Matrix')

#It can be seen that many of the features are highly correlated with each other 1) radius_mean,perimeter_mean_area_mean,radius_worst,perimeter_worst,area_worst are highly correlated with each other.

#2) radius_se,perimeter_se and area_se are highly correlated with each other.

#3) texture_mean and texture_worst are highly correlated

#4) compactness,concavity,concave_points mean and worst values are also correlated with each other without any pattern which is w quite confusing !

mean_data=data.drop([  'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst','concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'],axis=1)
sns.pairplot(data=mean_data,hue='diagnosis')
mean_data=data.iloc[0:11].T
mean_data

se_data=data.drop([ 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst','concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se'],axis=1)
sns.pairplot(data=se_data,hue='diagnosis')
se_data=data.iloc[11:21].T

se_data

worst_data=data.drop([ 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se','concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'],axis=1)
sns.pairplot(data=worst_data,hue='diagnosis')
worst_data=data.iloc[21:31].T
worst_data

worst_data

data2=worst_data.drop(['diagnosis'], axis=0)

corr=data2.corr()
corr

sns.displot(data=data,x='radius_mean',col='diagnosis',kde=True)

sns.set_theme()
sns.relplot(data=data,
            x='radius_mean',y='texture_mean',col='diagnosis',
            )

#Jointplot comparing radius and texture based on diagnosis
sns.jointplot(data=data,x='radius_mean',y='texture_mean',hue='diagnosis',palette='bwr_r',kind='scatter')

sns.relplot(data=data,
            x='area_mean',y='perimeter_mean',col='diagnosis',
            color = 'darkorange')

sns.jointplot(data=data,x='perimeter_mean',y='area_mean',hue='diagnosis',palette='bwr_r',kind='hist')

sns.relplot(data=data,x='smoothness_mean',y='compactness_mean',col='diagnosis')

sns.relplot(data=data, x='concavity_mean',y='concave points_mean',col='diagnosis')

sns.relplot(data=data,
            x='concave points_se',y='symmetry_se',col='diagnosis'
            )

sns.relplot(data=data,
            x='area_se',y='smoothness_se',col='diagnosis',hue='compactness_se',color='r'
            )

sns.relplot(data=data, x='radius_se',y='texture_se',col='diagnosis'
            )

sns.relplot(data=data,x='symmetry_mean',y='fractal_dimension_mean',col='diagnosis')

sns.relplot(data=data,
            x='fractal_dimension_se',y='radius_worst',col='diagnosis',hue='texture_worst'
            )

sns.relplot(data=data,x='texture_mean',y='area_mean',col='diagnosis')

sns.relplot(data=data,
            x='perimeter_worst',y='area_worst',col='diagnosis',color='g'
            )

#Jointplot comparing concativity and smoothness based on diagnosis
sns.jointplot(data=data,x='concavity_mean',y='smoothness_mean',hue='diagnosis',palette='bwr_r',kind='kde')

# count plot of the diagnosis
plt.figure(figsize = (8, 4), dpi = 100)
sns.countplot(data = data, x = 'diagnosis')
plt.show()

predictors=data.columns[1:]

data[predictors].hist(figsize=(18,18))

We can see that most of the features are right skewed . Therefore, it would be better if we scaled the features.

for col in predictors:
    sns.scatterplot(x=data[col],y=data['diagnosis'])
    plt.show()

#These scatter plots show that-

tumors with high radius, area, perimeter and concave points are usually malignant

Radius_mean

sns.boxplot(data['radius_mean'])

Texture_mean

sns.boxplot(data['texture_mean'])

Area_worst

sns.boxplot(data['area_worst'])

Area se

sns.boxplot(data['area_se'])

#Handling outliers

data = data[(data['radius_mean'] < 23) & (data['texture_mean'] < 35) & (data['area_worst'] < 2300) & (data['area_se'] < 150)]

#we can also check outliers as below:
for i in data.select_dtypes("number").columns:

    plt.figure()
    plt.title(f'{i}')
    plt.boxplot(data[i], vert = False);

def outliers(data, ft):
    Q1 = data[ft].quantile(0.25)
    Q3 = data[ft].quantile(0.75)
    IQR = Q3-Q1

    low = Q1 - 1.5 * IQR
    top = Q3 + 1.5 * IQR

    ls = data.index[ (data[ft] < low ) |  (data[ft]  > top) ]
    return ls

index_list = []
for i in ["radius_mean", "texture_mean","perimeter_mean", "area_mean", "radius_se","perimeter_se",
         "texture_se", "area_se","area_worst", "symmetry_worst","fractal_dimension_worst"]:
    index_list.extend(outliers(data, i))

def remove (data, ls):
    ls = sorted(set(ls))
    df = data.drop(ls)
    return df

data = remove(data, index_list)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
data[["radius_mean", "texture_mean"]] = min_max_scaler.fit_transform(
    data[["radius_mean", "texture_mean"]])
data.head()

from sklearn.model_selection import train_test_split
y = data.diagnosis
X = data.drop(columns ="diagnosis", axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

from sklearn.preprocessing import StandardScaler
s= StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)

#build a logistic regression classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#Make predictions on test data
predictions = classifier.predict(X_test)

#plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)
sns.heatmap(cm,annot = True)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))

num_folds = 10
results = []
names = []

for name, model in models_list:
    kfold = KFold(n_splits=num_folds)
    start = time.time()
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

#import warnings

# Standardize the dataset
pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
                                                                        DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
                                                                      GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
                                                                       KNeighborsClassifier())])))
results = []
names = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

# KNN CLASSIFICATION
from sklearn.neighbors import KNeighborsClassifier

# Finding optimum K value using elbow method
error_rate=[]
for i in range(1,51):
    knn_model=KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train,y_train)
    y_pred=knn_model.predict(X_test)
    error_rate.append(np.mean(y_pred!=y_test))

# Finding optimum K value using elbow method
error_rate=[]
for i in range(1,51):
    knn_model=KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train,y_train)
    y_pred=knn_model.predict(X_test)
    error_rate.append(np.mean(y_pred!=y_test))

figure=plt.figure(figsize=(10,10))
plt.plot(range(1,51),error_rate,linestyle='dashed',marker='o')

knn_mod=accuracy_score(y_test,pred_results)

knn_model=KNeighborsClassifier(n_neighbors=37)
knn_model.fit(X_train,y_train)
pred_results=knn_model.predict(X_test)
accuracy_score(y_test,pred_results)

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier_model=RandomForestClassifier()
classifier_model.fit(X_train,y_train)
pred_results=classifier_model.predict(X_test)
accuracy_score(y_test,pred_results)

rfc=accuracy_score(y_test,pred_results)

from sklearn.ensemble import GradientBoostingClassifier

gb_model=GradientBoostingClassifier()

gb_model.fit(X_train,y_train)

pred_results=gb_model.predict(X_test)

accuracy_score(y_test,pred_results)

gb_tech=accuracy_score(y_test,pred_results)
Gradient Boosting also resulted in an accuracy of 92%

from sklearn.svm import SVC
SVC_Gaussian = SVC(kernel='rbf', gamma=0.1,C=10)
SVC_Gaussian.fit(X_train,y_train)
y_pred = SVC_Gaussian.predict(X_test)
print(classification_report(y_test, y_pred))

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# prepare the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
model = SVC(C=2.0, kernel='rbf')
start = time.time()
model.fit(X_train_scaled, y_train)
end = time.time()
print( "Run Time: %f" % (end-start))

# estimate accuracy on test dataset
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)

print("Accuracy score %f" % accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

We can see that we achieve an accuracy of 95.28% on the held-out test dataset. From the confusion matrix, there is only 3 case of mis-classification. The performance of this algorithm is expected to be high given the symptoms for breast cancer should exchibit certain clear patterns.

#Ridge Classifier¶

# Ridge Classifier with VIF features and hyperparameter tuning
from sklearn.linear_model import RidgeClassifier
steps = [('scaler', StandardScaler()),
         ('ridge', RidgeClassifier())]
pipeline = Pipeline(steps)

parameters = dict(ridge__alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1,
                  error_score = 0.0)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
ridge_accuracy = accuracy_score(y_pred, y_test) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(ridge_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()

#Gradient Boosting

# Gradient Boosting Classifier  with VIF features  and hyperparameter tuning

steps = [('scaler', StandardScaler()),
         ('gbc', GradientBoostingClassifier())]
pipeline = Pipeline(steps)

parameters = dict(gbc__n_estimators = [10,100,200],
                  gbc__loss = ['deviance', 'exponential'],
                  gbc__learning_rate = [0.001, 0.1, 1, 10]
)


cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1,
                  error_score = 0.0
                  )

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
gb_accuracy = accuracy_score(y_pred, y_test) * 100
print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(gb_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()

print(y_test)

print(predictions)

#Random_forest

Finding best n_estimators value

from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators':[15, 20, 30, 40, 50, 100, 150, 200, 300, 400]}

GR = GridSearchCV(RandomForestClassifier(random_state=42),
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=-1)
GR = GR.fit(X_train, y_train)
y_test_pred_gr = GR.predict(X_test)
print(GR.best_estimator_)
print("")
print(classification_report(y_test, y_test_pred_gr))
rf=RandomForestClassifier(n_estimators=400).fit(X_train,y_train)
RandomForestClassifier(n_estimators=150, random_state=42)

#Random Forest Classifier gives an accuracy of 94%

#DecisionTree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))

#Decision Tree gives an accuracy of just 91%

from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth':range(1, dt.tree_.max_depth+1, 2),
              'max_features': range(1, len(dt.feature_importances_)+1)}

GR = GridSearchCV(DecisionTreeClassifier(random_state=42),
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=-1)

GR = GR.fit(X_train, y_train)

GR.best_estimator_.tree_.node_count, GR.best_estimator_.tree_.max_depth

y_test_pred_gr = GR.predict(X_test)

#Decision Tree gives an accuracy of just 92%

#Feature importances

feature_imp = pd.Series(dt.feature_importances_,index=data.columns[1:]).sort_values(ascending=False)

ax = feature_imp.plot(kind='bar', figsize=(16, 6))
ax.set(ylabel='Relative Importance');
ax.set(xlabel='Feature');

#This shows that only -

#perimeter_worst,concave_points_worst,compactness_worst,texture_worst,texture_mean,area_se, radius_mean,fractal_dimension_worst,symmetry_worst,smoothness_worst,compactness_se,perimeter_mean

#could have been used as they are the only important features

#Naive_bayes

for i in range(1,20): # Loop to try all error rates from 1 to 20
    rfe = RandomForestClassifier(n_estimators=i*10) # Create rfc with number of estimators with value i*10
    rfe.fit(X_train,y_train) # Fit the model
    errpred = rfe.predict(X_test) # Predict the value
    err.append(np.mean(errpred != y_test)) #Add the value to the array

# Plotting the value of estimators error rate using the method we created above to make it easier to choose an estimator value
plt.figure(figsize=(20,10)) # Size of the figure
plt.plot(range(1,20),err,color='blue',linestyle='dotted',marker='o',markerfacecolor='red',markersize=10)#plotting the values
plt.title = 'Number of estimators VS Error Rates' #title
plt.xlabel = 'Estimators' #X label
plt.ylabel= 'Error Rate' # Y label
plt.show()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred1 = gnb.predict(X_test)

print(confusion_matrix(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))

#Naive bayes gives an accuracy of just 94%

#Random forest classifier

rfc = RandomForestClassifier(n_estimators=180,max_features='auto', max_depth=8, n_jobs=-1)
rfc.fit(X_train,y_train)
rfcpred = rfc.predict(X_test)

print('Random Forest' + '\n')
print(classification_report(y_test,rfcpred))

#Confusion matrix
sns.heatmap(confusion_matrix(y_test,rfcpred), annot=True,cmap='Greens',fmt='g',linewidth=2,linecolor='black')

#Supervised Learning with Support Vector Machine

from sklearn import svm
svc_clr = svm.SVC(kernel='linear')
svc_clr.fit(X_train, y_train)
y_pred_scv = svc_clr.predict(X_test)
accuracy_score(y_test, y_pred_scv)

print(classification_report(y_test, y_pred_scv))

# This is formatted as code
#Support Vector Machine gives an accuracy of just 95%

#Feature scaling

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#Train with Standard scaled Data

svc_clr2 = SVC()
svc_clr2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_clr2.predict(X_test_sc)
accuracy_score(y_test, y_pred_svc_sc)

print(classification_report(y_test, y_pred_svc_sc))
             
#Support Vector Machine gives an accuracy of 97%

#Conclusions 1) Since this is a medical problem our main objective is to accuractly predict the malignant tumors i.e the label 1 . Therefore, the main aim should be to decrease the number of false negatives and have the highest possible recall as possible. This can be achieved by any of the 98% classifiers . However, I would reccomend using the gradient boosted algorithm as it has the highest recall for both the classes.

#2) Only some of the features are really important out of the 33 features with which we started . We can clearly see grid seach showing the maximum number of features used were 3 and the feature importances showed by the decision tree.

#Suggestions 1) My suggestion would be to try feature selection in the early stages and use only those features that have a high relative importance and then train the model. Maybe that will increase the accuracy even more .

#2) After that stacking classifier can be used which eg. using a voting classifier with logistic regression and gradient boosting which should increase the accuracy even more.
