---
layout: post
title: Use out-of-core training to detect fake news
subtitle: An approach to deal with a data set too big to fit in memory
gh-repo: thiagolcmelo/thiagolcmelo.github.io
gh-badge: [star, follow]
tags: [machine-learning, scikit-learn, fake-news-detector]
thumbnail-img: /assets/img/fake-news.png
comments: true
---

It is not a secret that the bad guys are using this horrible weapon in a scale never saw before. Our legal systems are not fast enough to stop them, because they are too fast. Machine learning techniques are certainly a way out of this war.

Unfortunately we have to work in this context, and our focus in this article is not to find the best technique to solve this challenge. In this article we aim to suggest a way to handle data sets that are too big to fit in memory.

The selected data set is not really that big, but it will serve the purpose.

## The data set

We will use a Kaggle's data set named [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). It has 44,898 texts labeled as True or Fake.

The data set consists of two csv files. If we download it to a directory named `kaggle`, we can use the following code to split the articles in individual files that we will use during our out-of-core training.

{% highlight python linenos %}
from csv import reader
from os import makedirs, path
from re import sub

class Article:
    def __init__(self, title: str, text: str):
        self.title = title
        self.text = text
    
    @property
    def clean(self):
        sanitizer = '[^A-Za-z]+'
        clean = sub(sanitizer, ' ', f'{self.title} {self.text}')
        clean = clean.lower()
        return sub('\s+', ' ', clean)



def split_dataset(base_folder: str = 'kaggle'):
    fake_folder = path.join(base_folder, 'fake')
    true_folder = path.join(base_folder, 'true')
    makedirs(fake_folder, exist_ok=True)
    makedirs(true_folder, exist_ok=True)
    
    for file in ['Fake', 'True']:
        with open(path.join('kaggle', f'{file}.csv'), 'r',
                  encoding='utf-8',
                  errors='ignore') as f:
            csv_reader = reader(f, delimiter=',', quotechar='"')
            _ = next(csv_reader)  # headers
            for i, row in enumerate(csv_reader):
                article_file = path.join(base_folder,
                                         file.lower(),
                                         f'{i:05d}.txt')
                with open(article_file, 'w+') as a:
                    a.write(Article(row[0], row[1]).clean)


split_dataset()
{% endhighlight %}

For sanity check, we can run:

{% highlight python linenos %}
from glob import glob

print('Fake articles:', len(glob(path.join('kaggle', 'fake', '*'))))
# Fake articles: 23481

print('True articles:', len(glob(path.join('kaggle', 'true', '*'))))
# True articles: 21417
{% endhighlight %}


## Pre-processing

The approach here will be the one of encoding each article as an array of words' occurrences. In this array, each position corresponds to one specific word.

To begin with, we create a transformer to encode our articles. When the encoder is fitted for the first time, it persists the words dictionary in disk inside a directory called `.dictionaries`. Before fitting again, the encoder will check first if a dictionary with the same name already exists. The parameter `refit` allows to fit again ignoring an existing dictionary.

{% highlight python linenos %}
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ArticleEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 dictionary_name: str,
                 binary: bool = False,
                 top: int = 1000,
                 min_length: int = 2,
                 max_length: int = 100,
                 refit: bool = False):
        makedirs('.dictionaries', exist_ok=True)
        self.dictionary = None
        self.dictionary_name = dictionary_name
        self.dictionary_filename = path.join('.dictionaries',
                                             f'{self.dictionary_name}.npy')
        self.binary = binary
        self.top = top
        self.min_length = min_length
        self.max_length = max_length
        self.refit = refit
    
    def fit(self, X, y=None):
        if path.exists(self.dictionary_filename) and not self.refit:
            self.dictionary = np.load(self.dictionary_filename)
        
        else:
            dictionary = defaultdict(int)
            
            for article_path in X:
                with open(article_path, 'r') as article:
                    for word in article.read().split(' '):
                        dictionary[word] += 1
            
            descending_dictionary = sorted(dictionary.items(),
                                           key=lambda v: v[1],
                                           reverse=True)
            
            self.dictionary = np.array([
                word for (word, occur) in descending_dictionary
                if self.min_length <= len(word) <= self.max_length
            ][:self.top])
            
            if self.dictionary_name:
                np.save(path.join('.dictionaries', self.dictionary_name),
                        self.dictionary)

        return self
    
    def transform(self, X):
        return np.array(list(map(self.encode_article, X)))
        
    def encode_article(self, article_path: str):
        encoded = np.zeros(self.dictionary.size)
        
        with open(article_path, 'r') as article:
            words = article.read().split(' ')
        
        for word in words:
            index = np.where(self.dictionary == word)[0]
            if index.size == 1:  # we ignore unknown words
                if self.binary:
                    encoded[index[0]] = 1
                else:
                    encoded[index[0]] += 1
        return encoded
{% endhighlight %}

Next step is to split our articles in train and test data sets:

{% highlight python linenos %}
from sklearn.model_selection import train_test_split


true_articles = glob(path.join('kaggle', 'true', '*'))
fake_articles = glob(path.join('kaggle', 'fake', '*'))

all_articles = np.concatenate((np.array(true_articles),
                               np.array(fake_articles)))
all_labels = np.concatenate((np.zeros(len(true_articles)),
                             np.ones(len(fake_articles))))

articles_train, articles_test, labels_train, labels_test = \
    train_test_split(all_articles,
                     all_labels,
                     test_size=0.2,
                     random_state=42,
                     stratify=all_labels)
{% endhighlight %}

And we fit our encoder using only the training set:

{% highlight python linenos %}
encoder = ArticleEncoder(dictionary_name='dictionary_1000')
encoder.fit(articles_train)
{% endhighlight %}

## Training a model

Because we are assuming our data set is too big to fit in memory at once, we need to choose a model that allows the training process to be done in small batches. A good candidate for that is the `SGDClassifier` which applies the gradient descent algorithm to minimize the loss function of a linear model.

We create 5 folds out of the training set:

{% highlight python linenos %}
from sklearn.model_selection import StratifiedKFold


kfold = StratifiedKFold(n_splits=5).split(articles_train,
                                          labels_train)
{% endhighlight %}

This function will help us by printing a good looking percentage bar.

{% highlight python linenos %}
def print_epoch_bar(cur_epoch: int, n_epochs: int, acc: list, loss: list):
    pct = 100 * cur_epoch / n_epochs
    h_pct = int(pct / 2)
    acc_ = 100 * sum(acc) / len(acc)
    loss_ = sum(loss) / len(loss)
    print('{} {} {}'.format('{:.2f} % | '.format(pct).rjust(11, ' '),
                            '{}'.format("=" * h_pct + '>').ljust(51, ' '),
                            f'| acc: {acc_:.2f} %, loss: {loss_:.4f}'))
{% endhighlight %}

We create some arrays to hold the training and validation metrics:

{% highlight python linenos %}
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hinge_loss,
    precision_score,
    recall_score
)

train_accuracy, val_accuracy = [], []
train_f1, val_f1 = [], []
train_hinge, val_hinge = [], []
train_precision, val_precision = [], []
train_recall, val_recall = [], []
{% endhighlight %}

And then we have a **big** for loop that will train estimators for each one the folds.

{% highlight python linenos %}
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


for k, (train, val) in enumerate(kfold):
    print(f'Fold {k}')
    articles_train_ = articles_train[train]
    articles_val_ = articles_train[val]
    labels_train_ = labels_train[train]
    labels_val_ = labels_train[val]
    
    # train
    print('training')
    print()
    n_epochs = 100
    batch_size = 100
    pipeline = Pipeline([
        ('encoder', ArticleEncoder(dictionary_name='dictionary_1000')),
        ('sgd_clf', SGDClassifier(warm_start=True,
                                  max_iter=1,
                                  penalty='l1',
                                  alpha=0.01,
                                  loss='hinge'))
    ])
    
    
    train_accuracy_ = []
    train_f1_ = []
    train_hinge_ = []
    train_precision_ = []
    train_recall_ = []
    
    for epoch in range(n_epochs):
        idx = np.random.randint(0, articles_train_.size, batch_size)
        articles = articles_train_[idx]
        labels = labels_train_[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(articles, labels)
        
        label_pred = pipeline.predict(articles)
        train_accuracy_.append(accuracy_score(labels, label_pred))
        train_f1_.append(f1_score(labels, label_pred))
        train_hinge_.append(hinge_loss(labels, label_pred))
        train_precision_.append(precision_score(labels, label_pred))
        train_recall_.append(recall_score(labels, label_pred))
    
        print_epoch_bar(epoch, n_epochs, train_accuracy_, train_hinge_)
    
    # validate
    print('validating')
    label_pred = pipeline.predict(articles_val_)
    
    val_accuracy_ = accuracy_score(labels_val_, label_pred)
    val_f1_ = f1_score(labels_val_, label_pred)
    val_hinge_ = hinge_loss(labels_val_, label_pred)
    val_precision_ = precision_score(labels_val_, label_pred)
    val_recall_ = recall_score(labels_val_, label_pred)
    
    print(', '.join([f'loss: {hinge_:04f}',
                     f'precision: {100 * precision_:.2f}',
                     f'recall: {100 * recall_:.2f}']))
    print()
    val_accuracy.append(val_accuracy_)
    val_f1.append(val_f1_)
    val_hinge.append(val_hinge_)
    val_precision.append(val_precision_)
    val_recall.append(val_recall_)
    
    train_accuracy.append(np.mean(train_accuracy_))
    train_f1.append(np.mean(train_f1_))
    train_hinge.append(np.mean(train_hinge_))
    train_precision.append(np.mean(train_precision_))
    train_recall.append(np.mean(train_recall_))
{% endhighlight %}

If we now print a summary of what we have accomplished so far:

{% highlight python linenos %}
def get_interval(values: list, pct: bool = True):
    avg = np.mean(values) * (100.0 if pct else 1.0)
    std = np.std(values) * (100.0 if pct else 1.0)
    formated = '({:.2f} +/- {:.2f})'.format(avg, std / np.sqrt(len(values)))
    if pct:
        return f'{formated} %'
    return formated


print('cross validation result:')
print()    
print('Avg. train accuracy: {}'.format(get_interval(train_accuracy)))
print('Avg. train precision: {}'.format(get_interval(train_precision)))
print('Avg. train recall: {}'.format(get_interval(train_recall)))
print('Avg. train f1: {}'.format(get_interval(train_f1)))
print('Avg. train hinge loss: {}'.format(get_interval(train_hinge, pct=False)))
print()    
print('Avg. val accuracy: {}'.format(get_interval(val_accuracy)))
print('Avg. val precision: {}'.format(get_interval(val_precision)))
print('Avg. val recall: {}'.format(get_interval(val_recall)))
print('Avg. val f1: {}'.format(get_interval(val_f1)))
print('Avg. val hinge loss: {}'.format(get_interval(val_hinge, pct=False)))
{% endhighlight %}

The result is:

```
cross validation result:

Avg. train accuracy: (94.27 +/- 0.17) %
Avg. train precision: (95.40 +/- 0.18) %
Avg. train recall: (93.84 +/- 0.42) %
Avg. train f1: (94.23 +/- 0.27) %
Avg. train hinge loss: (0.54 +/- 0.00)

Avg. val accuracy: (91.85 +/- 1.66) %
Avg. val precision: (98.94 +/- 0.48) %
Avg. val recall: (85.40 +/- 3.58) %
Avg. val f1: (91.48 +/- 1.86) %
Avg. val hinge loss: (0.56 +/- 0.02)
```

Not terrible, but it is overfitting.

A few highlights on the code above:

1. The parameter `warm_start` is the one that allows the model to be trained in small batches.
2. Using `penalty=l1` extends for Lasso regularization, which is interesting when we have a large number of features (many words) that might not have the same importance.
3. The parameter `alpha` determines the strength of the regularization contrain.
4. The choice `loss=hinge` is an attempt do have something like a Support Vector Machine, which is great finding a safe boundary that separates classes, but not suitable for out-of-core training.

In order to fight the overfitting, we could try to increase the `alpha` parameter in the `SGDClassifier`, however the low overall accuracy during training already tells us that the capacity of the model is not big enough.

Let's see how it performs on the test set:

{% highlight python linenos %}
from sklearn.metrics import classification_report

print('cross validation result:')
label_pred = pipeline.predict(articles_test)
print(classification_report(labels_test,
                            label_pred,
                            target_names=['True', 'Fake'],
                            digits=4))
{% endhighlight %}

And the result is:

```
              precision    recall  f1-score   support

        True     0.8212    0.9788    0.8931      4284
        Fake     0.9765    0.8056    0.8828      4696

    accuracy                         0.8882      8980
   macro avg     0.8989    0.8922    0.8880      8980
weighted avg     0.9024    0.8882    0.8877      8980
```

## Conclusion

We saw a simple strategy to train a model when the data set doesn't fit in memory all at once. Althoght it has room for improvement, it might be a good scaffolding for a simpler problem.

This problem would be better addressed by NLP, and we might return to it in the future, but for now we might say we have better feel of the problem's complexity.
