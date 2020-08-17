---
layout: post
title: Train you own spam detector
subtitle: A basic and powerful strategy to separate spam from ham
gh-repo: thiagolcmelo/thiagolcmelo.github.io
gh-badge: [star, fork, follow]
tags: [machine-learning, scikit-learn, spam-detector]
cover-img: /assets/img/letters-in-computer.jpg
thumbnail-img: /assets/img/road-sign-spam.jpg
share-img: /assets/img/letters-in-computer.jpg
comments: true
---

You certainly heard many times about using machine learning to separate spam from ham. Do you want to have a glimpse of how it works?

We chose to stick to the Python standard library as much as possible, but there are great libraries out there to accomplish everything done here.

Having said that, time to get our hands dirty reinventing some wheels!


## The data set

[Apache SpamAssassin](https://spamassassin.apache.org/) Project maintains a nice collection of old e-mail messages that we can use. The messages age from early 2000's, and probably the scammers are way smarter now, so please don't use this in any production environment :)

Download the data set using the following code:


{% highlight python linenos %}
from os import makedirs, path, remove, rename, rmdir
from tarfile import open as open_tar
from urllib import request, parse


def download_corpus(dataset_dir: str = 'data'):
    base_url = 'https://spamassassin.apache.org'
    corpus_path = 'old/publiccorpus'
    files = {
        '20021010_easy_ham.tar.bz2': 'ham',
        '20021010_hard_ham.tar.bz2': 'ham',
        '20021010_spam.tar.bz2': 'spam',
        '20030228_easy_ham.tar.bz2': 'ham',
        '20030228_easy_ham_2.tar.bz2': 'ham',
        '20030228_hard_ham.tar.bz2': 'ham',
        '20030228_spam.tar.bz2': 'spam',
        '20030228_spam_2.tar.bz2': 'spam',
        '20050311_spam_2.tar.bz2': 'spam' 
    }
    
    downloads_dir = path.join(dataset_dir, 'downloads')
    ham_dir = path.join(dataset_dir, 'ham')
    spam_dir = path.join(dataset_dir, 'spam')


    makedirs(downloads_dir, exist_ok=True)
    makedirs(ham_dir, exist_ok=True)
    makedirs(spam_dir, exist_ok=True)
    
    for file, spam_or_ham in files.items():
        # download file
        url = parse.urljoin(base_url, f'{corpus_path}/{file}')
        tar_filename = path.join(downloads_dir, file)
        request.urlretrieve(url, tar_filename)
        
        # list e-mails in compressed file
        emails = []
        with open_tar(tar_filename) as tar:
            tar.extractall(path=downloads_dir)
            for tarinfo in tar:
                if len(tarinfo.name.split('/')) > 1:
                    emails.append(tarinfo.name)
        
        # move e-mails to ham or spam dir
        for email in emails:
            directory, filename = email.split('/')
            directory = path.join(downloads_dir, directory)
            rename(path.join(directory, filename),
                   path.join(dataset_dir, spam_or_ham, filename))
        
        rmdir(directory)


download_corpus()
{% endhighlight %}

We have a corpus of 6952 hams and 2399 spams.

{% highlight python linenos %}
from glob import glob
from os import path

ham_dir = path.join('data', 'ham')
spam_dir = path.join('data', 'spam')

print('hams:', len(glob(f'{ham_dir}/*')))  # hams: 6952
print('spams:', len(glob(f'{spam_dir}/*')))  # spams: 2399
{% endhighlight %}


## Parsing messages

If you open any of these individual files, you will see they are very hard to read. This is because they are in [MIME format]()https://en.wikipedia.org/wiki/MIME. Python has a standard library that helps us to extract only the part that we care about, namely subject and body.

Let's create a class to represent a message. This class will hold the subject, the body, and will have a method to retrieve a clean string containing only letters.

{% highlight python linenos %}
from re import sub


class SimpleEmail:
    def __init__(self, subject: str, body: str):
        self.subject = subject
        self.body = body
    
    @property
    def clean(self):
        sanitizer = '[^A-Za-z]+'
        clean = sub(sanitizer, ' ', f'{self.subject} {self.body}')
        clean = clean.lower()
        return sub('\s+', ' ', clean)
    
    def __str__(self):
        subject = f'subject: {self.subject}'
        body_first_line = self.body.split('\n')[0]
        body = f'body: {body_first_line}...'
        return f'{subject}\n{body}'

    def __repr__(self):
        return self.__str__()
{% endhighlight %}

When we first started, we thought these messages would be heavy to load at once in memory, and because of that we built this generator for reading messages from a directory. In the end the messages are not that heavy…

{% highlight python linenos %}
from email import message_from_file
from glob import glob


class EmailIterator:
    def __init__(self, directory: str):
        self._files = glob(f'{directory}/*')
        self._pos = 0
    
    def __iter__(self):
        self._pos = -1
        return self
    
    def __next__(self):
        if self._pos < len(self._files) - 1:
            self._pos += 1
            return self.parse_email(self._files[self._pos])
        raise StopIteration()
    
    @staticmethod
    def parse_email(filename: str) -> SimpleEmail:
        with open(filename,
                  encoding='utf-8',
                  errors='replace') as fp:
            message = message_from_file(fp)
        
        subject = None
        for item in message.raw_items():
            if item[0] == 'Subject':
                subject = item[1]
        
        if message.is_multipart():
            body = []
            for b in message.get_payload():
                body.append(str(b))
            body = '\n'.join(body)
        else:
            body = message.get_payload()
        
        return SimpleEmail(subject, body)
{% endhighlight %}


## Pre-processing

We can load everything in memory, we will be fine :)

{% highlight python linenos %}
import numpy as np


ham_emails = EmailIterator('data/ham')
spam_emails = EmailIterator('data/spam')

hams = np.array([email.clean for email in ham_emails])
spams = np.array([email.clean for email in spam_emails])
{% endhighlight %}

Since we have an unbalanced data set (6952 hams and 2399 spams) we need to take care when splitting it in training and test sets. We can use Scikit-Learn's `StratifiedShuffleSplit` for that. It will make sure that we have a homogeneous distribution of hams and spams in both training and test sets.

{% highlight python linenos %}
from sklearn.model_selection import StratifiedShuffleSplit


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

emails = np.concatenate((hams, spams))
labels = np.concatenate((np.zeros(hams.size), np.ones(spams.size)))

for train_index, test_index in split.split(emails, labels):
    emails_train, labels_train = \
        emails[train_index], labels[train_index]
    emails_test, labels_test = \
        emails[test_index], labels[test_index]
{% endhighlight %}

Using the messages in the training set, we build a dictionary with the occurrences of each word across all messages.

{% highlight python linenos %}
from collections import defaultdict


dictionary = defaultdict(int)

for email in emails_train:
    for word in email.split(' '):
        dictionary[word] += 1
{% endhighlight %}

And then we select only the top 1000 most frequent ones (you can experiment varying this number). Also, notice we are ignoring single letters (`len(word) > 1`).

{% highlight python linenos %}
top = 1000
descending_dictionary = sorted(dictionary.items(),
                               key=lambda v: v[1],
                               reverse=True)
dictionary = [
    word for (word, occur) in descending_dictionary
    if len(word) > 1
][:top]
{% endhighlight %}

The idea now is that we will encode each message (subject + body) into an array where each index indicates how many times a given word appears there. For instance, if our dictionary was only:

```python
["yes", "no", "have", "write", "script", "myself", "to"]
```

And a certain message is "I would prefer not to have to write a script myself", it would be encoded as:

```python
[0, 0, 1, 1, 1, 1, 2]
```

Which means:

```python
[
    0,  # zero occurrence(s) of word yes
    0,  # zero occurrence(s) of word no
    1,  # one  occurrence(s) of word have
    1,  # one  occurrence(s) of word write
    1,  # one  occurrence(s) of word script
    1,  # one  occurrence(s) of word myself
    2   # two  occurrence(s) of word to
]
```

You could also do "0" or "1" for "not occur" or "occur", respectively. The following function encodes a given message using the approach just described.

{% highlight python linenos %}
def encode_email(email: SimpleEmail,
                 dictionary_: list,
                 binary: bool = False) -> np.array:
    encoded = np.zeros(dictionary_.size)
    words = email.split(' ')
    
    for word in words:
        index = np.where(dictionary_ == word)[0]
        if index.size == 1:  # we ignore unknown words
            if binary:
                encoded[index[0]] = 1
            else:
                encoded[index[0]] += 1
    return encoded
{% endhighlight %}

And then we encode our messages.

{% highlight python linenos %}
from functools import partial


dictionary = np.array(dictionary)
_encode_email = partial(encode_email, dictionary_=dictionary)

encoded_train = np.array(list(map(_encode_email, emails_train)))
encoded_test = np.array(list(map(_encode_email, emails_test)))
{% endhighlight %}


## Training a model

Think about when you read an e-mail yourself and try to judge whether it is a spam or not. A good approach is to search for certain words or combination of words that you previously saw in spam e-mails. Words are just words, they are valid entities of a given language, there is nothing suspicious about them. But because you saw some of them in a certain combination and in a certain context, using a certain grammatical style, that is what makes them suspicious.

A good model for tackling such a task of comparing a message with other messages is the K Nearest Neighbors. It is a good guess because it doesn't fit a mathematical expression, it is instance based, which means it will use a measure of distance to compare a new message with all the other known messages, and flag spam or ham according to how similar this new message is with its closest "neighbours".

Let's give one shot.

{% highlight python linenos %}
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier


knn_clf = KNeighborsClassifier()
labels_pred = cross_val_predict(knn_clf,
                                encoded_train,
                                labels_train,
                                cv=5)

print('accuracy:', accuracy_score(labels_train, labels_pred))
# accuracy: 0.9648395721925134

print('precision:', precision_score(labels_train, labels_pred))
# precision: 0.9630872483221476

print('recall:', recall_score(labels_train, labels_pred))
# recall: 0.897342365815529

print('f1:', f1_score(labels_train, labels_pred))
# f1: 0.9290531427029943
{% endhighlight %}

That is cool right? A recall of 89% just out of the box. Well, this also means that 11% of the spams are escaping our detection, and because of the 96% precision, it also means that we are flagging as spam 4% of the valid messages. We can try hyperparameters tuning to improve this.

From our experience, companies prefer to flag a message as spam even when it is not than to allow a spam message into someone's inbox. For that reason we believe we seek to improve recall here.

{: .box-warning}
**Warning:** The following code might take one hour or two to run depending on your machine.

{% highlight python linenos %}
from sklearn.model_selection import GridSearchCV

params_grid = [{
    'n_neighbors': [2, 5, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['minkowski'],
    'metric_params': [{'p': 2}, {'p': 3}, {'p': 4}]
}]

search = GridSearchCV(knn_clf, params_grid, n_jobs=6,
                      scoring='recall', cv=5, verbose=1)
search.fit(encoded_train, labels_train)
{% endhighlight %}

The best parameters using this grid search are:

```python
{
    'algorithm': 'ball_tree',
    'metric': 'minkowski',
    'metric_params': {'p': 2},
    'n_neighbors': 2,
    'weights': 'distance'
}
```

Using the best estimator:

{% highlight python linenos %}
labels_pred = cross_val_predict(search.best_estimator_,
                                encoded_train,
                                labels_train,
                                cv=5)


print('accuracy:', accuracy_score(labels_train, labels_pred))
# accuracy: 0.9740641711229947

print('precision:', precision_score(labels_train, labels_pred))
# precision: 0.9551451187335093

print('recall:', recall_score(labels_train, labels_pred))
# recall: 0.9431995831162063

print('f1:', f1_score(labels_train, labels_pred))
# f1: 0.9491347666491873
{% endhighlight %}

The precision decreased a little bit (from 96% to 95%), but the recall improved a lot (from 89% to 94%).

Let's now see how it performs in the test set.

{% highlight python linenos %}
knn_clf = KNeighborsClassifier(algorithm='ball_tree',
                               n_neighbors=2,
                               weights='distance')
knn_clf.fit(encoded_train, labels_train)
labels_pred = knn_clf.predict(encoded_test)

print('accuracy:', accuracy_score(labels_test, labels_pred))
# accuracy: 0.982896846606093

print('precision:', precision_score(labels_test, labels_pred))
# precision: 0.9666666666666667

print('recall:', recall_score(labels_test, labels_pred))
# recall: 0.9666666666666667

print('f1:', f1_score(labels_test, labels_pred))
# f1: 0.9666666666666667
{% endhighlight %}

Nice right? It seems to be generalizing quite well.


## Automating previous steps

First thing, we need an easy way to encode messages, and also vary some encoding parameters as hyperparameters if we want to. For that, let's create a `MessageEncoder` estimator.

{% highlight python linenos %}
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class MessageEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, binary: bool = False, top: int = 1000):
        self.dictionary_ = None
        self.binary = binary
        self.top = top
    
    def fit(self, X, y=None):
        dictionary = defaultdict(int)
        
        for email in X:
            for word in email.split(' '):
                dictionary[word] += 1
        
        descending_dictionary = sorted(dictionary.items(),
                                       key=lambda v: v[1],
                                       reverse=True)
        
        self.dictionary = np.array([
            word for (word, occur) in descending_dictionary
            if len(word) > 1
        ][:self.top])
        
        return self
    
    def transform(self, X):
        return np.array(list(map(self.encode_message, X)))
        
    def encode_message(self, message: str):
        encoded = np.zeros(self.dictionary.size)
        words = message.split(' ')
        
        for word in words:
            index = np.where(self.dictionary == word)[0]
            if index.size == 1:  # we ignore unknown words
                if self.binary:
                    encoded[index[0]] = 1
                else:
                    encoded[index[0]] += 1
        return encoded
{% endhighlight %}

Cool, now we can use it like:

{% highlight python linenos %}
encoder = MessageEncoder()
encoder.fit(emails_train)
encoded_emails_train = encoder.transform(emails_train)
{% endhighlight %}

Or we can use it directly in a pipeline:

{% highlight python linenos %}
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('encode_messages', MessageEncoder()),
    ('knn_clf', KNeighborsClassifier()),
])
{% endhighlight %}

It makes possible to run grid search and cross validation in our pipeline.

{: .box-warning}
**Warning:** The following code might take four to five hours to run depending on your machine.

{% highlight python linenos %}
from sklearn.model_selection import GridSearchCV

params_grid = [{
    'encode_messages__binary': [True, False],
    'encode_messages__top': [500, 1000],
    'knn_clf__n_neighbors': [2, 5, 10],
    'knn_clf__weights': ['uniform', 'distance'],
    'knn_clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}]

pipe_search = GridSearchCV(pipeline, params_grid, n_jobs=1,
                           scoring='recall', cv=5, verbose=2)
pipe_search.fit(emails_train, labels_train)
{% endhighlight %}

And the outcome is:

```python
{
    'encode_messages__binary': True,
    'encode_messages__top': 1000,
    'knn_clf__algorithm': 'auto',
    'knn_clf__n_neighbors': 2,
    'knn_clf__weights': 'distance'
}
```

The difference is the `binary` parameter of the `MessageEncoder`. It seems that flagging absence or presence of a word is better than counting occurrences.

Finally, we can then check if we are still in good shape:

{% highlight python linenos %}
from sklearn.metrics import classification_report

labels_test_pred = pipe_search.best_estimator_.predict(emails_test)
print(classification_report(labels_test,
                            labels_pred,
                            target_names=['ham', 'spam'],
                            digits=4))
{% endhighlight %}
                         
```
              precision    recall  f1-score   support

         ham     0.9877    0.9799    0.9838      1391
        spam     0.9430    0.9646    0.9537       480

    accuracy                         0.9759      1871
   macro avg     0.9653    0.9722    0.9687      1871
weighted avg     0.9762    0.9759    0.9760      1871
```


## Conclusion

Although this is not a production ready model, it is nice that we can have such a nice recall without a lot of effort.

Thanks for reading so far, I hope you enjoyed having a few code samples you can copy and paste :)

Please let me know if something is not clear.
