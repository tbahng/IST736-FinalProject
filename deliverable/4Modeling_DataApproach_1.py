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

# 2020 Candidate count
len(df.handle[(df.candidate_2020 == True)].unique())
# 2016 & 2020 Candidate count
len(df.handle[(df.candidate_2020 == True) | (df.candidate_2016 == True)].unique())

# In[4]:

#removing the websites from tweet data
df['text'] = df['text'].str.replace('http\S+|www.\S+|rt|amp', '', case=False)


# In[2020 Candidates]:

#2020 candidates with 2018 tweets

tweets_2018 = df[(df['created_at'] >= '2018-01-01') & (df['created_at'] <= '2018-12-31') & (df.candidate_2020 == True)]
candidates_2018 = list(tweets_2018['handle'].unique())

#2020 candidates w/o 2018 tweets
not_2018 = df[~df.handle.isin(candidates_2018) & (df.candidate_2020 == True)]
candidates_not2018 = list(not_2018['handle'].unique())

len(candidates_2018) + len(candidates_not2018) == len(df.handle[df.candidate_2020 == True].unique())

count_dist = tweets_2018.groupby('handle')['text'].count().reset_index(name='count')
print('Median Tweets for Candidates: ', count_dist['count'].median())

#tweets for those not retrieved from 2018
count_dist2 = not_2018.groupby('handle')['text'].count().reset_index(name='count')
print('Median Tweets for Candidates: ', count_dist2['count'].median())

#For those that did not have tweets in 2018, grab the first 198 tweets
not_2018_ordered = not_2018.groupby(["handle"]).apply(lambda x: x.sort_values(["created_at"], ascending = True)).reset_index(drop=True)

#grab the top 198 from everyone
final_not_2018 = not_2018_ordered.groupby('handle').head(count_dist['count'].median())

#merge the dataframes and convert the target variable
merge_df = pd.concat([final_not_2018, tweets_2018], axis=0, sort=False)

final_df = merge_df.groupby('handle').agg({'text':'sum', 'candidate_2020':'first'}).reset_index()

final_df['candidate_2020'] = final_df['candidate_2020'].apply(lambda x: 1 if x is True else 0)
print("Final Dataframe shape: ", final_df.shape)

sum(final_df.candidate_2020)

# In[Non-2020 Candidates]

#2020 non-candidates with 2018 tweets

tweets_2018_nc = df[(df['created_at'] >= '2018-01-01') & (df['created_at'] <= '2018-12-31') & (df.candidate_2020 == False)]
candidates_2018_nc = list(tweets_2018_nc['handle'].unique())

#2020 non-candidates w/o 2018 tweets
not_2018_nc = df[~df.handle.isin(candidates_2018_nc) & (df.candidate_2020 == False)]
candidates_not2018_nc = list(not_2018_nc['handle'].unique())

len(candidates_2018_nc) + len(candidates_not2018_nc) == len(df.handle[df.candidate_2020 == False].unique())

count_dist3 = tweets_2018_nc.groupby('handle')['text'].count().reset_index(name='count')
print('Median Tweets for Non-Candidates: ', count_dist3['count'].median())

#tweets for those not retrieved from 2018
count_dist4 = not_2018_nc.groupby('handle')['text'].count().reset_index(name='count')
print('Median Tweets for Non-Candidates: ', count_dist4['count'].median())

#For those that did not have tweets in 2018, grab the first 198 tweets
not_2018_ordered_nc = not_2018_nc.groupby(["handle"]).apply(lambda x: x.sort_values(["created_at"], ascending = True)).reset_index(drop=True)

ordered_nc = tweets_2018_nc.groupby(["handle"]).apply(lambda x: x.sort_values(["created_at"], ascending = True)).reset_index(drop=True)

#grab the top 198 from everyone
final_not_2018_nc = not_2018_ordered_nc.groupby('handle').head(count_dist['count'].median())

final_2018_nc = ordered_nc.groupby('handle').head(count_dist['count'].median())

#merge the dataframes and convert the target variable
merge_df_nc = pd.concat([final_not_2018_nc, final_2018_nc], axis=0, sort=False)

final_df_nc = merge_df_nc.groupby('handle').agg({'text':'sum', 'candidate_2020':'first'}).reset_index()

final_df_nc['candidate_2020'] = final_df_nc['candidate_2020'].apply(lambda x: 1 if x is True else 0)
print("Final Dataframe shape: ", final_df_nc.shape)

sum(final_df_nc.candidate_2020)

# Grabbing a random 34 non-candidates to even the class balance
final_df_nc2 = train_test_split(final_df_nc, test_size = 34)[1]

# In[Merge candidates and non-candidates]

FinalDF = final_df.append(final_df_nc2, ignore_index = True)


# In[]

############ MODELING SECTION ###############


# In[Count Vec | MNB | Base]

cvec1 = CountVectorizer()
cvec1_fit = cvec1.fit_transform(FinalDF.text)
Colnames1 = cvec1.get_feature_names()
cvec1_df = pd.DataFrame(cvec1_fit.toarray(), columns = Colnames1, index = FinalDF.candidate_2020)


train, test = train_test_split(cvec1_df, test_size = 0.3)
# Check for class balance
print(sum(test.index)/test.shape[0])


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
for k in range(0,len(cvec_coef.Word)):
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
ax.set_title("Top Most Indictative Words | Model #1")
colors = {'Ran':'DarkGreen', "Didn't Run":'Firebrick'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)


cvec_y_probas = MNB1.predict_proba(test)
fpr, tpr, _ = roc_curve(test.index,  cvec_y_probas[:,1])
auc = roc_auc_score(test.index, cvec_y_probas[:,1])
fpr2, tpr2, _ = roc_curve(test.index, cvec_y_probas[:,0])
auc2 = roc_auc_score(test.index, cvec_y_probas[:,0])

plt.plot(fpr,tpr,label="Didn't Run, AUC = "+str(round(auc,2)), color = "Firebrick")
plt.plot(tpr2,fpr2,label="Ran, AUC = "+str(round(1-auc2,2)),  color = "DarkGreen")
plt.plot([0, 1.1], [0, 1.1], color='grey', lw=2, linestyle='--')
plt.xlim([-0.005, 1.003])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model #1 | ROC')
plt.legend(loc="lower right")
plt.show()



# In[Count Vec2 | MNB]  # SKIP

cvec2 = CountVectorizer(stop_words = STOPWORDS)
cvec2_fit = cvec2.fit_transform(FinalDF.text)
Colnames2 = cvec2.get_feature_names()
cvec2_df = pd.DataFrame(cvec2_fit.toarray(), columns = Colnames2, index = FinalDF.candidate_2020)


train2, test2 = train_test_split(cvec2_df, test_size = 0.3)
# Check for class balance
sum(test2.index)/test2.shape[0] 


MNB2 = MultinomialNB()
MNB2.fit(train2, train2.index)
pred2 = MNB2.predict(test2)
cm2 = confusion_matrix(test2.index, pred2)

print("\nThe confusion matrix is:")
print(cm2,"\n")
print(classification_report(test2.index, pred2))

cv2 = cross_val_score(MNB2, test2, test2.index, cv = 5)
print("The 5-fold CV score is:",round(np.mean(cv2),2))




# In[Count Vec3 | MNB]

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(stem) for stem in analyzer(doc) if stem not in STOPWORDS)

cvec3 = CountVectorizer(analyzer = stemmed_words, min_df = 0.10)
cvec3_fit = cvec3.fit_transform(FinalDF.text)
Colnames3 = cvec3.get_feature_names()
cvec3_df = pd.DataFrame(cvec3_fit.toarray(), columns = Colnames3, index = FinalDF.candidate_2020)

train3, test3 = train_test_split(cvec3_df, test_size = 0.3)
# Check for class balance
print(round(sum(test3.index)/test3.shape[0], 2))


MNB3 = MultinomialNB()
MNB3.fit(train3, train3.index)
pred3 = MNB3.predict(test3)
cm3 = confusion_matrix(test3.index, pred3)

print("\nThe confusion matrix is:")
print(cm3,"\n")
print(classification_report(test3.index, pred3))

cv3 = cross_val_score(MNB3, test3, test3.index, cv = 5)
print("The 5-fold CV score is:",round(np.mean(cv3),2))


cvec_coef = pd.DataFrame(np.exp(MNB3.feature_log_prob_)).T
cvec_coef["Word"] = Colnames3
cvec_coef = cvec_coef[cvec_coef.Word != "rt"]
cvec_coef = cvec_coef[cvec_coef.Word != "amp"]
cvec_coef = cvec_coef.reset_index()

ran = []
no_ran = []
i = 0 
for k in range(0,len(cvec_coef.Word)):
    if cvec_coef[0][i] > cvec_coef[1][i]:
        no_ran.append([cvec_coef[0][i],cvec_coef["Word"][i]])
    else:
        ran.append([cvec_coef[1][i],cvec_coef["Word"][i]])
    i += 1

ran = sorted(ran, reverse = True)[:10]
no_ran = sorted(no_ran, reverse = True)[:10]
ran_tot =  no_ran + ran
ran_tot_c, ran_tot_w = [*zip(*ran_tot)]


fig, ax = plt.subplots()
ax.bar(ran_tot_w, ran_tot_c, color = mycolors)
ax.set_xticklabels(ran_tot_w, rotation=60)
ax.set_ylabel("Plain Odds")
ax.set_title("Top Most Indictative Words | Model #2")
colors = {'Ran':'DarkGreen', "Didn't Run":'Firebrick'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)


cvec_y_probas = MNB3.predict_proba(test3)
fpr, tpr, _ = roc_curve(test3.index,  cvec_y_probas[:,1])
auc = roc_auc_score(test3.index, cvec_y_probas[:,1])
fpr2, tpr2, _ = roc_curve(test3.index, cvec_y_probas[:,0])
auc2 = roc_auc_score(test3.index, cvec_y_probas[:,0])

plt.plot(fpr,tpr,label="Didn't Run, AUC = "+str(round(auc,2)), color = "Firebrick")
plt.plot(tpr2,fpr2,label="Ran, AUC = "+str(round(1-auc2,2)),  color = "DarkGreen")
plt.plot([0, 1.1], [0, 1.1], color='grey', lw=2, linestyle='--')
plt.xlim([-0.005, 1.003])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model #2 | ROC')
plt.legend(loc="lower right")
plt.show()




# In[Count Vec5 | MNB] # SKIP

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(stem) for stem in analyzer(doc) if stem not in STOPWORDS)

cvec4 = CountVectorizer(analyzer = stemmed_words, min_df = 0.10)
cvec4_fit = cvec4.fit_transform(FinalDF.text)
Colnames4 = cvec4.get_feature_names()
cvec4_df = pd.DataFrame(cvec4_fit.toarray(), columns = Colnames4, index = FinalDF.candidate_2020)
# keep words only (removing numbers and other characters that count vec missed)
cvec4_df = cvec4_df.filter(regex = r'(\b[a-z]+\b)')

train4, test4 = train_test_split(cvec4_df, test_size = 0.3)
# Check for class balance
print(round(sum(test4.index)/test4.shape[0], 2))


MNB4 = MultinomialNB()
MNB4.fit(train4, train4.index)
pred4 = MNB4.predict(test4)
cm4 = confusion_matrix(test4.index, pred4)

print("\nThe confusion matrix is:")
print(cm4,"\n")
print(classification_report(test4.index, pred4))

cv4 = cross_val_score(MNB4, test4, test4.index, cv = 5)
print("The 5-fold CV score is:",round(np.mean(cv4),2))


# In[TF-IDF Vec1 | MNB] # SKIP

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(stem) for stem in analyzer(doc) if stem not in STOPWORDS)

cvec5 = TfidfVectorizer(analyzer = stemmed_words, min_df = 0.10)
cvec5_fit = cvec5.fit_transform(FinalDF.text)
Colnames5 = cvec5.get_feature_names()
cvec5_df = pd.DataFrame(cvec5_fit.toarray(), columns = Colnames5, index = FinalDF.candidate_2020)

train5, test5 = train_test_split(cvec5_df, test_size = 0.3)
# Check for class balance
print(round(sum(test5.index)/test5.shape[0], 2))


MNB5 = MultinomialNB()
MNB5.fit(train5, train5.index)
pred5 = MNB5.predict(test5)
cm5 = confusion_matrix(test5.index, pred5)

print("\nThe confusion matrix is:")
print(cm5,"\n")
print(classification_report(test5.index, pred5))

cv5 = cross_val_score(MNB5, test5, test5.index, cv = 5)
print("The 5-fold CV score is:",round(np.mean(cv5),2))


# In[Count Vec3 | SVM]

param_grid = {'C': [0.1, 1, 10, 50, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],'kernel': ['rbf', 'poly', 'sigmoid', "linear"], "degree":[1,2,3,4,5,6]}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose = 2)
grid.fit(train3, train3.index)
print(grid.best_estimator_)

svm1 = svm.SVC(C = 10, kernel = "rbf", gamma = 0.00001, probability = True) 
# kernel options = ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
# degree (for poly kernel) = [number] (int)
# gamma (for = ‘poly’, ‘rbf’, ‘sigmoid’) = "scale" or "auto" or a number (float)


svm1.fit(train3, train3.index)
pred_svm1 = svm1.predict(test3)
cm_svm1 = confusion_matrix(test3.index, pred_svm1)

print("\nThe confusion matrix is:")
print(cm_svm1,"\n")
print(classification_report(test3.index, pred_svm1))

cv_svm1 = cross_val_score(svm1, test3, test3.index, cv = 5)
print("The 5-fold CV score is:",round(np.mean(cv_svm1),2))


cv_svm1_probas = svm1.predict_proba(test3)
auc_svm1 = roc_auc_score(test3.index, cv_svm1_probas[:,1])
auc_svm1
fpr, tpr, _ = roc_curve(test3.index,  cv_svm1_probas[:,1])

plt.plot(fpr,tpr,label="Ran, AUC = "+str(round(auc_svm1,2)))
plt.plot([0, 1.1], [0, 1.1], color='grey', lw=2, linestyle='--')
plt.xlim([-0.005, 1.003])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model #3 | ROC')
plt.legend(loc="lower right")
plt.show()

