
11
# 方案一：svm (article+word_seq)
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

# 方案二：svm (article+word_seq)
## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, 
             max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

**clf = svm.LinearSVC(C=5,dual=False)**

## 提交结果：**0.779634** 

## 耗时   26203.1
### 读取 63.7
### 向量化 7347.4
### 训练  18757.8

## 预测分数 **0.9958**

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


# 方案五：svm (word_seq)

## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,
                      use_idf=1,smooth_idf=1, sublinear_tf=1)
                     
**C=0.5**

## 提交结果 **0.777-**
 
## 耗时   9894
### 读取  64
### 向量化  7300
### 训练     2434

## 预测分数  0.98


# 方案六：svm (word_seq)

## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2, max_df=0.9,
                    use_idf=1,smooth_idf=1, sublinear_tf=1)
                     
**C=4**

## 提交结果 **0.777-**
 
## 耗时   2645
### 读取  87
### 向量化  1322
### 训练     1273

## 预测分数  0.994

# 方案七：svm (article+word_seq)
## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, 
             max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

**clf = svm.LinearSVC(C=4,dual=False)**

## 提交分数：**0.77929 - 0.779634之间**

## 耗时 23854
### 读取 64
### 向量化 7711
### 训练  16079

## 预测分数 **0.9958**

# 方案八：svm (article+word_seq)
## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, 
             max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

**clf = svm.LinearSVC(C=6,dual=False)**

## 提交分数： **0.77929 - 0.779634之间**

## 耗时   28007
### 读取  83
### 向量化 7356
### 训练  20568

## 预测分数 **0.9966**


# 方案九：lg (article+word_seq)
## 参数
vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=2, 
             max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

**lg = LogisticRegression(C=40,dual=True)**

## 提交分数：**0.77- **排第六

## 耗时 21113
### 读取 67
### 向量化 7396
### 训练  13525

## 预测分数 **0.995828**









