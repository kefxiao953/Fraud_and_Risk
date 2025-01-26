# 引入模块
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import seaborn as sns
import missingno as msno
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

import matplotlib.gridspec as gridspec

# 分类器库
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_val_score
import collections

# 其它库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, precision_score, precision_recall_curve, roc_curve, recall_score, roc_auc_score 
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split, RandomizedSearchCV,StratifiedShuffleSplit,cross_val_predict

import itertools

#==========================================EDA============================================
#EDA是我们进行数据项目的第一步，在进行分析之前，首先我们应该去了解数据，这就是EDA（探索性数据分析），
#需要了解的，一般包括以下这些内容：
#1.基础信息：数据量（有多少行）,数据特征(有多少列，每一列表示什么意思，每一列的数据类型),
#           数据意义(每行数据表示的含义）
#2.数据分布：了解一下Numerical数据列的数据分布，包括中位数，2分位数，4分位数，方差，标准差等等
#3.统计信息：对影响业务的数据列进行统计。
#3.数据问题：哪些数据列存在异常，什么样的异常。
#4.数据间的关系：行与行之间的关系，列与列之间的关系
#5.根据本项目的业务目标，探索数据其他信息
#=========================================================================================
#读取数据
data = pd.read_csv("creditcard.csv")

#=======================观察基础信息=============================
data.columns = [col.strip() for col in data.columns]  # 去除列名中的空格或不可见字符
#观察前5条数据
print(data.head())
#观察数据类型
data.info() 

#=======================观察数据分布=============================
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amount_val, ax=ax[0])
ax[0].set_title('Distribution of Transaction Amount')
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1])
ax[1].set_title('Distribution of Transaction Time')
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()

#=======================计算统计信息=============================
#既然是欺诈检测，那么我们自然想到要看一下数据中有多少是fraud的数据
#所以我们做一下普通交易和欺诈交易的数量统计
count_classes = pd.value_counts(data['Class'])
print("count_classes:"+str(count_classes)) 

#画成图像看一下
sns.countplot(x='Class', data=data)
plt.title("Fraud Class Distributions")
plt.show() #如果是在jupyter notebook之下，则不需要这句话
#经过观察图像，我们发现，这份数据中存在极度的样本不均衡问题，
#因此我们在后续处理中，需要着重关注这个问题
#而正是因为我们进行了恰当的EDA操作，才发现了这个问题。

#=======================检查数据问题，测空值=============================
# 使用missingno查看数据集中的空值
# 矩阵图 (msno.matrix(data))：
# 这个图显示了数据集中的空值情况。白色的线表示缺失值，黑色表示存在值。
# 这种图表可以很好地展示数据缺失的分布和模式
msno.matrix(data)
#plt.show()

# 可以自己试一下下面这几种测空值的方式：
# 条形图 msno.bar(data)
# 热力图 msno.heatmap(data)
# 树状图 msno.dendrogram(data)
#=======================观察数据问题=============================

#=======================根据业务，进行相应的数据探索============================

#=======================Data Preprocess=====================================

# 大多数数据已经被缩放，再缩放剩下的Amount和Time列
from sklearn.preprocessing import StandardScaler, RobustScaler

# 使用RobustScaler
rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)

print("=============")
data.columns = [col.strip() for col in data.columns]  # 去除列名中的空格或不可见字符
data.describe()
print(data.head())

#进入modeling部分
#先进行训练数据集的划分，一般按照8:2的比例来划分
#======================Train Test Dividing=====================================
X = data.drop('Class', axis=1)
y = data['Class']

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2,train_size=0.8,random_state=42)

for train_index, test_index in sss.split(X,y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# 查看训练集和测试集是否具有分布相同
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)
print(train_counts_label)
print(test_counts_label)

#=========================Modeling=============================================
#=========================先进行Train Test spliting=============================================
# 重新洗牌
data = data.sample(frac=1)

# 分类为欺诈的数量有492行；由于数据已随机打乱，取分类为正常的数据前492行，两者进行合并
fraud_data = data.loc[data['Class'] == 1]
non_fraud_data = data.loc[data['Class'] == 0][:492]
normal_distributed_df = pd.concat([fraud_data, non_fraud_data])

# 将合并后的数据打乱
data_undersample = normal_distributed_df.sample(frac=1, random_state=42)

data_undersample.head()

#=========================欠采样=========================
print('Distribution of the Classes in the subsample dataset')
print(data_undersample['Class'].value_counts()/len(data_undersample))

sns.countplot(x='Class', data=data_undersample)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# 原始数据集
corr = data.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

#欠采样数据集
data_undersample_corr = data_undersample.corr()
sns.heatmap(data_undersample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('UnderSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()

#=========================欠采样下的modeling操作=========================
#========================再次划分Train/test dataset====================
X_undersample = data_undersample.drop('Class', axis=1)
y_undersample = data_undersample['Class']

for train_index, test_index in sss.split(X_undersample,y_undersample):
    print("Train:", train_index, "Test:", test_index)
    X_train_undersample, X_test_undersample = X_undersample.iloc[train_index], X_undersample.iloc[test_index]
    y_train_undersample, y_test_undersample = y_undersample.iloc[train_index], y_undersample.iloc[test_index]

X_train_undersample = X_train_undersample.values
X_test_undersample = X_test_undersample.values
y_train_undersample = y_train_undersample.values
y_test_undersample = y_test_undersample.values

#========================设定多种分类器==========================
classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

#=======================依次对每个分类器进行效果测试================
#欠采样集训练模型
for key, classifier in classifiers.items():
    classifier.fit(X_train_undersample, y_train_undersample)
    training_score = cross_val_score(classifier, X_train_undersample, y_train_undersample, cv=5)
    prediction = classifier.predict(X_test_undersample)  #用验证集预测
        
    accuracy_lst=classifier.score(X_test_undersample, y_test_undersample)
    precision_lst=precision_score(y_test_undersample, prediction)
    recall_lst=recall_score(y_test_undersample, prediction)
    f1_lst=f1_score(y_test_undersample, prediction)
    auc_lst=roc_auc_score(y_test_undersample, prediction)
        
    print('---' * 45)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
    print("accuracy: {}".format(accuracy_lst))
    print("precision: {}".format(precision_lst))
    print("recall: {}".format(recall_lst))
    print("f1: {}".format(f1_lst))
    print('---' * 45)

#======================寻找最佳参数模型========================
from sklearn.model_selection import GridSearchCV

# 逻辑回归
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train_undersample, y_train_undersample)
# 获得最佳参数的逻辑回归模型
log_reg = grid_log_reg.best_estimator_

#K近邻
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train_undersample, y_train_undersample)
# K近邻最佳参数
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train_undersample, y_train_undersample)
# SVC最佳参数
svc = grid_svc.best_estimator_

# 决策树
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train_undersample, y_train_undersample)
# 决策树最佳参数
tree_clf = grid_tree.best_estimator_

#=====================得到最佳参数模型后进行交叉验证评估======================
log_reg_score = cross_val_score(log_reg, X_train_undersample, y_train_undersample, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors, X_train_undersample, y_train_undersample, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train_undersample, y_train_undersample, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train_undersample, y_train_undersample, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')

#=====================得到最佳参数并交叉验证评估======================
# 分类器以及所对应预测得的决策值,代入训练集进行评估
log_reg_pred = cross_val_predict(log_reg, X_train_undersample, y_train_undersample, cv=5,
                             method="decision_function")
knears_pred = cross_val_predict(knears_neighbors, X_train_undersample, y_train_undersample, cv=5)
svc_pred = cross_val_predict(svc, X_train_undersample, y_train_undersample, cv=5,
                             method="decision_function")
tree_pred = cross_val_predict(tree_clf, X_train_undersample, y_train_undersample, cv=5)

df = pd.DataFrame({'log_reg':log_reg_pred,
                'knears':knears_pred,
                'svc':svc_pred,
                'tree':tree_pred,})
                
print('Logistic Regression: ', roc_auc_score(y_train_undersample, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train_undersample, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train_undersample, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train_undersample, tree_pred))

#=====================画出ROC曲线======================
# ROC曲线,全称Receiver Operating Characteristic Curve(受试者特征曲线)
# 是一种用于评估二分类模型性能的图形化工具。
# 它通过绘制真阳性率（True Positive Rate, TPR，也称为灵敏度）与
# 假阳性率（False Positive Rate, FPR）之间的关系来展示模型在不同分类阈值下的性能。
# ROC曲线越靠近左上角，说明模型的诊断或预测效果越好。
# 曲线下面积（Area Under Curve, AUC）是ROC曲线的一个重要评价指标，AUC值越大，模型的诊断或预测效果越好。

#真正类：正确分类的欺诈交易 (True Positive)
#假正类：错误分类的欺诈交易
#真负类：正确分类的非欺诈交易
#假负类：错误分类的非欺诈交易
#精度：真正类/（真正类+假正类）
#召回：真正类/（真正类+假负类）
#精确度表示我们的模型在检测欺诈交易中的精确度（确定性），而召回是我们的模型能够检测到的欺诈案件的数量。
#精度/召回的权衡：我们的模型越精确，它将检测到的事件数量就越少。

# 绘制ROC曲线
log_fpr, log_tpr, log_thresold = roc_curve(y_train_undersample, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train_undersample, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train_undersample, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train_undersample, tree_pred)

def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train_undersample, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train_undersample, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train_undersample, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train_undersample, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()

graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()

#重新对逻辑回归模型用测试集评估
log_reg_score = cross_val_score(log_reg, original_Xtest, original_ytest, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

log_reg_pred = cross_val_predict(log_reg, original_Xtest, original_ytest, cv=5,
                                 method="decision_function")

log_fpr, log_tpr, log_thresold = roc_curve(original_ytest, log_reg_pred)

#定义一个绘制ROC曲线的函数，横轴假正类率，纵轴真正类率
def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure(figsize=(12,8))
    plt.title('Logistic Regression ROC Curve', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
       
logistic_roc_curve(log_fpr, log_tpr)
plt.show()

precision, recall, threshold = precision_recall_curve(original_ytest, log_reg_pred)

y_pred = log_reg.predict(original_Xtest)

print('---' * 45)
print('Overfitting: \n')
print('Recall Score: {:.2f}'.format(recall_score(original_ytest, y_pred)))
print('Precision Score: {:.2f}'.format(precision_score(original_ytest, y_pred)))
print('F1 Score: {:.2f}'.format(f1_score(original_ytest, y_pred)))
print('Accuracy Score: {:.2f}'.format(accuracy_score(original_ytest, y_pred)))
print('---' * 45)

undersample_y_score = log_reg.decision_function(original_Xtest)

from sklearn.metrics import average_precision_score

undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)
print('Average precision-recall score: {0:0.2f}'.format(undersample_average_precision))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)

plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('UnderSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          undersample_average_precision), fontsize=16)
plt.show()

#=====================过采样下的SMOTE======================
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

log_reg_sm = LogisticRegression()
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)

# SMOTE过采样及交叉验证
log_reg_params = {"penalty": ['l1','l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'solver':['liblinear']}
print("===============$$$$$$$$$$$$$$================")
for train, test in skf.split(original_Xtrain, original_ytrain):
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_log_reg.best_estimator_  #模型调参，最佳参数模型
    prediction = best_est.predict(original_Xtrain[test])  #用训练集中的验证集预测
    
    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

print('---' * 45)
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 45)


labels = ['No Fraud', 'Fraud']
smote_prediction = best_est.predict(original_Xtest)
print(classification_report(original_ytest, smote_prediction, target_names=labels))

y_score = best_est.decision_function(original_Xtest)
average_precision = average_precision_score(original_ytest, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
      
      
fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, y_score)

plt.step(recall, precision, color='r', alpha=0.2,where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='#F59B00')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          average_precision), fontsize=16)
       
# 对训练集进行SMOTE过采样处理后，此数据集用来训练模型
sm = SMOTE(random_state=42)
print("here 1 ")
# 获得过采样数据训练集
Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)  
Xsm_train = Xsm_train[:1000]
ysm_train = ysm_train[:1000]
print("here 2")
print("Xsm_train length")
print(len(Xsm_train))
print("ysm_train length")
print(len(ysm_train))
# 逻辑回归
t0 = time.time()
print("here 3 ")
grid_log_reg_sm = GridSearchCV(LogisticRegression(), log_reg_params)
print("here 4 ")
grid_log_reg_sm.fit(Xsm_train, ysm_train)
print("here 5 ")
log_reg_sm = grid_log_reg_sm.best_estimator_
print("here 6 ")
t1 = time.time()
print("Fitting oversample data took :{} sec".format(t1 - t0))

#======================测试========================
# SMOTE训练模型
y_pred_log_reg  = log_reg_sm.predict(original_Xtest)  #用测试集预测
log_reg_cf = confusion_matrix(original_ytest, y_pred_log_reg)
print(log_reg_cf)
          
fig, ax = plt.subplots(1, 1,figsize=(16,8))

sns.heatmap(log_reg_cf, ax=ax, annot=True, cmap=plt.cm.Blues, fmt='g')
ax.set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax.set_xticklabels(['', ''], fontsize=14, rotation=90)
ax.set_yticklabels(['', ''], fontsize=14, rotation=360)
plt.show()




