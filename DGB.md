# 方案一：svm
## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, 
             max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

clf = svm.LinearSVC(C=4,dual=False)

## 提交分数：**0.77929**

## 耗时
### 读取 63.5
### 向量化 7257
### 训练  21235

## 预测分数 **0.9948**

# 方案二：svm
## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, 
             max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

**clf = svm.LinearSVC(C=5,dual=False)**

## 提交结果：**0.779634**

## 耗时   26203.1
### 读取 63.7
### 向量化 7347.4
### 训练  18757.8

## 预测分数 **0.9957**

# 方案三：svm,lv,朴素贝叶斯各自预测，预测出的概率再放到一个svm中的版本(未执行)
## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, max_df=0.9,
            use_idf=1,smooth_idf=1, sublinear_tf=1)
            
clf1 = svm.LinearSVC(C=4,dual=False)

clf2 = LogisticRegression(C=40,dual=True)

clf3 = MultinomialNB()

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2, 1, 1], voting='soft')


## 提交结果


## 耗时   
### 读取 
### 向量化 
### 训练  


# 方案四：svm,lv,朴素贝叶斯各自预测，预测出的概率再放到一个svm中的版本(未执行)
## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, max_df=0.9,
                            use_idf=1,smooth_idf=1, sublinear_tf=1)
                            
clf1 = svm.LinearSVC(C=4,dual=False)

clf2 = LogisticRegression(C=40,dual=True)

clf3 = MultinomialNB()


sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],  
                          use_probas=True,  
                          average_probas=False,  
                          meta_classifier=clf1) 
## 提交结果


## 耗时   
### 读取 
### 向量化 
### 训练  


# 方案五：svm

## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,
                      use_idf=1,smooth_idf=1, sublinear_tf=1)
                     
**C=0.5**

## 提交结果 **0.777-**
        

## 耗时   9894
### 读取  64
### 向量化  7300
### 训练     2434


##预测分数  0.98