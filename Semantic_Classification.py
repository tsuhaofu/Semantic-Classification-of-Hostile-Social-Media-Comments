import os
import pandas as pd 
import  numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from wordcloud import WordCloud, ImageColorGenerator
import seaborn as sns
import json
import string
import re
import warnings
from pprint import pprint
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,  TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel,wrappers

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models # don't skip this
warnings.filterwarnings("ignore",category=DeprecationWarning)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def plot_10_most_common_words(count_data, count_vectorizer):  
    import matplotlib.pyplot as plt  
    words = count_vectorizer.get_feature_names()  
    total_counts = np.zeros(len(words))  
    for t in count_data:  
        total_counts+=t.toarray()[0]  
      
    count_dict = (zip(words, total_counts))  
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]  
    words = [w[0] for w in count_dict]  
    counts = [w[1] for w in count_dict]  
    x_pos = np.arange(len(words))   
      
    plt.figure(2, figsize=(12, 12/1.6180))  
    plt.subplot(title='10 most common words')  
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})  
    sns.barplot(x_pos, counts, palette='husl')  
    plt.xticks(x_pos, words, rotation=90)   
    plt.xlabel('words')  
    plt.ylabel('counts')  
    plt.show()  

#Import Data
json_df = pd.read_json (r'C:\Users\kevin\Desktop\統計計算\Dataset for Detection of Cyber-Trolls.json', lines=True,orient='columns')
json_df.shape
json_df.head()
x = []
y = []
stopword = stopwords.words('english')
stopword.append('im')
stopword.append('http')
alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)
lemmatizer = WordNetLemmatizer()
for i in range(len(json_df)):
    x.append(json_df.content[i])
    y.append(int(json_df.annotation[i]['label'][0]))
#Remove punctuation
#string.punctuation
for p in string.punctuation:
    for i in range(len(x)):
        x[i] = x[i].replace(p," ")

for i in range(len(x)):
#Remove digits
    x[i] = re.sub(r'\d+', '', x[i])
#Case normalization
    x[i] = x[i].lower()
#Remove extra spaces
    x[i] = re.sub(' +', ' ',x[i] )
    x[i] = x[i].lstrip().rstrip()
#Word tokenize
    x[i] = word_tokenize(x[i])
#Remove stopwords
    x[i] = [sw for sw in x[i] if sw not in stopword]
#Remove a-z
    x[i] = [letter for letter in x[i] if letter not in alphabet_list]
#Lemmatize
    lemmatize_x = []
    wordnet_tagged = map(lambda x: (x[0], get_wordnet_pos(x[1])), pos_tag(x[i]))
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatize_x.append(word)
        else:        
            lemmatize_x.append(lemmatizer.lemmatize(word, pos = tag))
    x[i] = lemmatize_x

#Combine x,y as df
new_dict = {
        "content" : x,
        "label" : y
       }
df = pd.DataFrame.from_dict(new_dict)
df = df[~df.content.str.len().eq(0)] #Remove empty contents
df.head()
#Countplot
plt.figure(figsize=(12,6))
sns.countplot(x = "label", data = df)
plt.show()

#WordCloud
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)
# Now we use the WordCloud function from the wordcloud library 
all_words = ''
for document in df['content'][df['label']==1]:
    words = ' '.join(text for text in document)
    all_words =  ' '.join([all_words, words])
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words)
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wc) 
plt.axis("off") 
plt.tight_layout(pad = 0)   
plt.show() 

#LDA
 # Create Dictionary
id2word = corpora.Dictionary(df['content'])
id2word[0]
 # Create Corpus
texts = df['content']
 # Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
 # Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
 # Build LDA model
perplexity_list = []
coherence_list = []
coherence_list3236 = []
coherence_list2731 = []
for num_topics in range(1,55):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=424,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    
    print(num_topics)
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_list2731.append(coherence_lda)
    print('\nCoherence Score: ', coherence_lda)
 # Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

 # Show graph
plt.plot(list(range(27,31)), coherence_list2731)
plt.xlabel("Num Topics")
plt.ylabel("Coherence Score")
plt.legend(("Coherence Score"), loc='best')
plt.title("Choosing the optimal model with Coherence Scores")
plt.show()

 #Save LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=31, 
                                           random_state=424,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
lda_model.save(r'C:\Users\kevin\Desktop\統計計算\lda.model')
 # Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
 # Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
 #Load lad model
model =  gensim.models.LdaModel.load(r'C:\Users\kevin\Desktop\統計計算\lda.model')
 #Topic Vector
topic_vectors = []
for i in range(len(df['content'])):
    top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(31)]
    topic_vectors.append(topic_vec)
topic_vectors[:100]

#PCA
pca_list = []
for i in range(20):
    pca = PCA(n_components=i)
    pc= pca.fit_transform(topic_vectors)
    pca_list.append(sum(pca.explained_variance_ratio_))
# Show graph
plt.plot(list(range(20)), pca_list)
plt.xlabel("Num Components")
plt.ylabel("sum of explained_variance_ratio")
plt.legend(("explained_variance_ratio"), loc='best')
plt.title("Choosing the optimal model with sum of explained_variance_ratio")
plt.show()

pca = PCA(n_components=i)
pc= pca.fit_transform(topic_vectors)
pca_list.append(sum(pca.explained_variance_ratio_))
pc_df = pd.DataFrame(data = pc
             , columns = ['principal component 1', 'principal component 2'])
pc_df = pd.concat([pc_df, df[['label']]], axis = 1)
 #Visualize 2D Projection
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pc_df['label'] == target
    ax.scatter(pc_df.loc[indicesToKeep, 'principal component 1']
               , pc_df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
plt.show()
ax.grid()

#SVM
 # Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(pc, df['label'], test_size=0.2,random_state=424) # 80% training and 20% test
SVM = svm.SVC()
 #Train the model using the training sets
SVM.fit(x_train, y_train)

 #Predict the response for train dataset
y_pred = SVM.predict(x_train)
 # Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train, y_pred))
 # Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_train, y_pred))
 # Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_train, y_pred))


 #Predict the response for test dataset
y_pred = SVM.predict(x_test)
metrics.confusion_matrix(y_test, y_pred) 
 # Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
 # Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
 # Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
topic_vectors[0]