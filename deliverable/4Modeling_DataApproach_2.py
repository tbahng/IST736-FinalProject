# In[74]:

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import PorterStemmer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score


# In[2]:

df = pd.read_csv('https://raw.githubusercontent.com/tbahng/IST736-FinalProject/master/data/data.csv')

df = df[df['created_at'] <= '2019-11-24'] # Last candidate announced

# 2020 Candidate count
len(df.handle[(df.candidate_2020 == True)].unique())
# 2016 & 2020 Candidate count
len(df.handle[(df.candidate_2020 == True) | (df.candidate_2016 == True)].unique())

# In[4]:

#removing the websites from tweet data
df['text'] = df['text'].str.replace('http\S+|www.\S+|rt|amp', '', case=False)

# In[2020 Candidates]:

df = df.groupby('handle').agg({'text':'sum', 'candidate_2020':'first'}).reset_index()

df['candidate_2020'] = df['candidate_2020'].apply(lambda x: 1 if x is True else 0)
print("Final Dataframe shape: ", df.shape)

sum(df.candidate_2020)

FinalDF = df

# In[]

############ MODELING SECTION ###############


# In[Count Vec | MNB]

cvec1 = CountVectorizer()
cvec1_fit = cvec1.fit_transform(FinalDF.text)
Colnames1 = cvec1.get_feature_names()
cvec1_df = pd.DataFrame(cvec1_fit.toarray(), columns = Colnames1, index = FinalDF.candidate_2020)

train, test = train_test_split(cvec1_df, test_size = 0.3, stratify = cvec1_df.index)
# Check for class balance
sum(test.index)/test.shape[0] 
sum(FinalDF.candidate_2020)/len(FinalDF.candidate_2020)

MNB1 = MultinomialNB()
MNB1.fit(train, train.index)
pred1 = MNB1.predict(test)
cm1 = confusion_matrix(test.index, pred1)

print("\nThe confusion matrix is:")
print(cm1,"\n")
print(classification_report(test.index, pred1))
cv1 = cross_val_score(MNB1, test, test.index, cv = 5)
print("The 5-fold CV score is:",round(np.mean(cv1),2))

cvec_coef = pd.DataFrame(np.exp(MNB1.feature_log_prob_)).T
cvec_coef["Word"] = Colnames1

ran = []
no_ran = []
i = 0 
for k in range(0,len(Colnames1)):
    if cvec_coef[0][i] > cvec_coef[1][i]:
        no_ran.append([cvec_coef[0][i],cvec_coef["Word"][i]])
    else:
        ran.append([cvec_coef[1][i],cvec_coef["Word"][i]])
    i += 1

ran = sorted(ran, reverse = True)[:10]
no_ran = sorted(no_ran, reverse = True)[:10]
ran_tot =  no_ran + ran
ran_tot_c, ran_tot_w = [*zip(*ran_tot)]

mycolors = 10*"Firebrick".split(" ") + 10*"DarkGreen".split(" ")

fig, ax = plt.subplots()
ax.bar(ran_tot_w, ran_tot_c, color = mycolors)
ax.set_xticklabels(ran_tot_w, rotation=60)
ax.set_ylabel("Plain Odds")
ax.set_title("Top Most Indictative Words | Model #4")
colors = {'Ran':'DarkGreen', "Didn't Run":'Firebrick'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)


cvec_y_probas = MNB1.predict_proba(test)
fpr, tpr, _ = roc_curve(test.index,  cvec_y_probas[:,1])
auc = roc_auc_score(test.index, cvec_y_probas[:,1])
fpr2, tpr2, _ = roc_curve(test.index, cvec_y_probas[:,0])
auc2 = roc_auc_score(test.index, cvec_y_probas[:,0])

plt.plot(fpr,tpr,label="Ran, AUC = "+str(round(auc,2)), color = "DarkGreen")
plt.plot(tpr2,fpr2,label="Didn't Run, AUC = "+str(round(1-auc2,2)), color = "Firebrick")
plt.plot([0, 1.1], [0, 1.1], color='grey', lw=2, linestyle='--')
plt.xlim([-0.005, 1.003])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model #4 | ROC')
plt.legend(loc="lower right")
plt.show()


# In[Count Vec2 | MNB]

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(stem) for stem in analyzer(doc) if stem not in STOPWORDS)

cvec2 = CountVectorizer(analyzer = stemmed_words, min_df = 0.10)
cvec2_fit = cvec2.fit_transform(FinalDF.text)
Colnames2 = cvec2.get_feature_names()
cvec2_df = pd.DataFrame(cvec2_fit.toarray(), columns = Colnames2, index = FinalDF.candidate_2020)

train2, test2 = train_test_split(cvec2_df, test_size = 0.3, stratify = cvec2_df.index)
# Check for class balance
sum(test2.index)/test2.shape[0] 
sum(FinalDF.candidate_2020)/len(FinalDF.candidate_2020)

MNB2 = MultinomialNB()
MNB2.fit(train2, train2.index)
pred2 = MNB2.predict(test2)
cm2 = confusion_matrix(test2.index, pred2)

print("\nThe confusion matrix is:")
print(cm2,"\n")
print(classification_report(test2.index, pred2))
cv2 = cross_val_score(MNB2, test2, test2.index, cv = 5)
print("The 5-fold CV score is:",round(np.mean(cv2),2))

cvec_coef = pd.DataFrame(np.exp(MNB2.feature_log_prob_)).T
cvec_coef["Word"] = Colnames2

ran = []
no_ran = []
i = 0 
for k in range(0,len(Colnames2)):
    if cvec_coef[0][i] > cvec_coef[1][i]:
        no_ran.append([cvec_coef[0][i],cvec_coef["Word"][i]])
    else:
        ran.append([cvec_coef[1][i],cvec_coef["Word"][i]])
    i += 1

ran = sorted(ran, reverse = True)[:10]
no_ran = sorted(no_ran, reverse = True)[:10]
ran_tot =  no_ran + ran
ran_tot_c, ran_tot_w = [*zip(*ran_tot)]

mycolors = 10*"Firebrick".split(" ") + 10*"DarkGreen".split(" ")

fig, ax = plt.subplots()
ax.bar(ran_tot_w, ran_tot_c, color = mycolors)
ax.set_xticklabels(ran_tot_w, rotation=60)
ax.set_ylabel("Plain Odds")
ax.set_title("Top Most Indictative Words | Model #5")
colors = {'Ran':'DarkGreen', "Didn't Run":'Firebrick'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)


cvec_y_probas = MNB2.predict_proba(test2)
fpr, tpr, _ = roc_curve(test2.index,  cvec_y_probas[:,1])
auc = roc_auc_score(test2.index, cvec_y_probas[:,1])
fpr2, tpr2, _ = roc_curve(test2.index, cvec_y_probas[:,0])
auc2 = roc_auc_score(test2.index, cvec_y_probas[:,0])

plt.plot(fpr,tpr,label="Ran, AUC = "+str(round(auc,2)), color = "DarkGreen")
plt.plot(tpr2,fpr2,label="Didn't Run, AUC = "+str(round(1-auc2,2)), color = "Firebrick")
plt.plot([0, 1.1], [0, 1.1], color='grey', lw=2, linestyle='--')
plt.xlim([-0.005, 1.003])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model #5 | ROC')
plt.legend(loc="lower right")
plt.show()


# In[Count Vec3 | SVM]

param_grid = {'C': [0.1, 1, 10, 50, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],'kernel': ['rbf', 'poly', 'sigmoid', "linear"], "degree":[1, 2, 3, 4, 5, 6]}
grid = GridSearchCV(svm.SVC(probability = True), param_grid, refit=True, verbose = 2)
grid.fit(train2, train2.index)
print(grid.best_estimator_)

svm1 = svm.SVC(kernel = "linear", C = 0.1, probability = True)
# kernel options = ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
# degree (for poly kernel) = [number] (int)
# gamma (for = ‘poly’, ‘rbf’, ‘sigmoid’) = "scale" or "auto" or a number (float)
train2, test2 = train_test_split(cvec2_df, test_size = 0.3, stratify = cvec2_df.index)
svm1.fit(train2, train2.index)
pred_svm1 = svm1.predict(test2)
cm_svm1 = confusion_matrix(test2.index, pred_svm1)

print("\nThe confusion matrix is:")
print(cm_svm1,"\n")
print(classification_report(test2.index, pred_svm1))

cv_svm1 = cross_val_score(svm1, test2, test2.index, cv = 5)
print("The 5-fold CV score is:",round(np.mean(cv_svm1),2))


most_important = pd.DataFrame({"coef":svm1.coef_.T[:,0], "word":Colnames2})
most_important = most_important.sort_values(by="coef", ascending = False)
most_important_ran = most_important[:10]
most_important_no_run = most_important[-10:]
most_important_no_run.coef = abs(most_important_no_run.coef)
most_important_no_run = most_important_no_run.sort_values(by = "coef", ascending = False)
most_important = most_important_no_run.append(most_important_ran)


most_important.plot.bar(x="word",y="coef", legend = False, title="Top Most Indicative Words | Model #6", rot = 60, color = mycolors)
colors = {'Ran':'DarkGreen', "Didn't Run":'Firebrick'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.show()


cvec_svm_y_probas = svm1.decision_function(test2)
fpr, tpr, _ = roc_curve(test2.index,  cvec_svm_y_probas)
auc = roc_auc_score(test2.index, cvec_svm_y_probas)

plt.plot(fpr,tpr,label="AUC = "+str(round(auc,2)))
plt.plot([0, 1.1], [0, 1.1], color='grey', lw=2, linestyle='--')
plt.xlim([-0.005, 1.003])
plt.ylim([0.0, 1.008])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model #6 | ROC')
plt.legend(loc="lower right")
plt.show()

