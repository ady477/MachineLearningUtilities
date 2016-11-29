# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 22:46:53 2016

@author: t_tiwaad
"""

import numpy as np
#import lda
import sklearn.decomposition.LatentDirichletAllocation as lda



## Text documents
titles = lda.datasets.load_reuters_titles()

# load Data Sets : Term Document Matrix
X = lda.datasets.load_reuters()

#Vocabulary
vocab = lda.datasets.load_reuters_vocab()


#model
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available

#All topics
topic_word = model.topic_word_  # model.components_ also works

#Printing top 8 words in the topic
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))