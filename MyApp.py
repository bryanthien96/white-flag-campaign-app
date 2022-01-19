# to run the app: streamlit run c:/Users/bryan/OneDrive/Desktop/MyApp2.py on cmd prompt

import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import re

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
from streamlit import components

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

st.set_page_config(page_title='White Flag Campaign in Malaysia',
                   page_icon='https://i.pinimg.com/564x/3f/28/4b/3f284bf2fa045059fa82fa119fcd4208.jpg',
                   layout="wide")

st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Header.jpg'), use_column_width=True)

st.header("Introduction")
with st.expander('Details'):
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://img.i-scmp.com/cdn-cgi/image/fit=contain,width=1098,format=auto/sites/default/files/styles/1200x800/public/d8/images/methode/2021/07/04/d0db1694-dbdf-11eb-9660-0b62a055b768_image_hires_154804.jpg?itok=iX84qcta&v=1625384901", caption="Image taken from South China Morning Post.")
    with col2:
        st.image("https://media2.malaymail.com/uploads/articles/2021/2021-07/white_flag_kuching_01072021.JPG", caption="Image taken from MalayMail.")
    st.markdown("The **White Flag Campaign** was initiatated by an entrepreneur and politician, named Nik Faizah Nik Othman who wrote a Facebook post to encourage people who needed help to raise a white flag outside of their home as a signal for people to help them if they were financially impacted by the COVID-19 pandemic.")

st.header("Text Exploration")
with st.expander('Details'):
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Hashtags.png'), caption='Top 15 hashtags in lollipop chart.')
    with col2:
        st.markdown(f'<h1 style="color:#4e4e94;font-size:27px;"><u>{"Top 15 Hashtags in Lollipop Chart"}<ins></h1>', unsafe_allow_html=True)
        st.markdown("Unsurprisingly, the hashtags of #BenderaPutih and #benderaputih have topped the chart as they are used for the tweets collection in this study. **#RakyatJagaRakyat** _(citizens take care of citizens)_ comes in third place, followed by **#KitaJagaKita** _(we take care of ourselves)_. Apart from that, **#KerajaanGagal** _(government failure)_, **#KerajaanPembunuh** _(murderer government)_, **#KerajaanBangsat** _(unpleasant government)_ and **#KerajaanBodoh** _(stupid government)_ are observed from the chart as well. Seemingly, the words associated with the government are all negative.")
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/General Wordcloud.png'), caption='Word cloud of tweets')
    with col2:
        st.markdown(f'<h1 style="color:#4e4e94;font-size:27px;"><u>{"World Cloud of Tweets"}<ins></h1>', unsafe_allow_html=True)
        st.markdown("The **dominant words** in this dataset consist of _“people”_, _“need”_, _“help”_, _“white”_, _“flag”_ and _“campaign”_.")
    st.markdown(f'<h1 style="color:#4e4e94;font-size:27px;"><u>{"Bigrams & Trigrams in Horizontal Bar Charts"}<ins></h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Bigrams.png'), caption='Top 15 bigrams in horizontal bar chart.')
    with col2:
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Trigrams.png'), caption='Top 15 trigrams in horizontal bar chart.')
    st.markdown("By looking at the chart of bigrams, _“white flag”_, _“let us”_ and _“need help”_ have topped the chart. On the other hand, _“raise white flag”_, _“white flag campaign”_ and _“let us help”_ appeared to be the most frequently discussed keywords among the trigrams. It seems that most Twitter users in Malaysia are somewhat encouraging people to ask for help in this campaign.")

st.header("Sentiment Analysis")
with st.expander('Details'):
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Pie Chart.png'), caption='Distribution of sentiments in pie chart.')
    with col2:
        st.markdown(f'<h1 style="color:#4e4e94;font-size:27px;"><u>{"Distribution of Sentiments in Pie Chart"}<ins></h1>', unsafe_allow_html=True)
        st.markdown("The results of sentiment analysis showed that **majority of the tweets had a positive tone** which account for roughly 62% of the pie chart, followed by 26% of negative tweets and 12% of neutral tweets.")
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Line Chart.png'), caption='Time series of sentiments in line chart.')  
    with col2:
        st.markdown(f'<h1 style="color:#4e4e94;font-size:27px;"><u>{"Time Series of Sentiments in Line Chart"}<ins></h1>', unsafe_allow_html=True)
        st.markdown("The number of positive tweets have always exceeded the number of both negative and neutral tweets. Apparently, the attention of this campaign only lasted about half a month as the frequency of tweets slowly decline after 15th July 2021. Besides, the peak of positive tweets happened around 28th to 30th June 2021, which was the time after the initiator, Nik Faizah Nik Othman published the post on Facebook.") 
    st.markdown(f'<h1 style="color:#4e4e94;font-size:27px;"><u>{"World Cloud of Positive, Negative & Neutral Tweets"}<ins></h1>', unsafe_allow_html=True)  
    col1, col2, col3 = st.columns(3)
    with col1:    
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Positive Wordcloud.png'), caption='Word cloud of positive tweets.')
    with col2:
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Negative Wordcloud.png'), caption='Word cloud of negative tweets.')
    with col3:
        st.image(Image.open('c:/Users/bryan/OneDrive/Desktop/Neutral Wordcloud.png'), caption='Word cloud of neutral tweets.')
    st.markdown("Based on the word clouds generated, it can be seen that the most common words found across all sentiments are _“people”_, _“white”_, _“flag”_ and _“campaign”_. Besides, words like _“help”_, _“please”_, _“food”_, _“need”_, _“support”_, _“thanks”_, etc. are found in the positive word cloud. Evidently, majority of the public seems to be quite supportive of the campaign generated. On the other hand, the words _“help”_ and _“government”_ emerged to be the most dominant words in the negative word cloud, while some other negative words included _“failure”_, _“fail”_, _“stupid”_, _“hijack”_, etc. By looking at the negative word cloud and the most frequent hashtags used in the tweets, it seems like most of the tweets related to the government carried a negative sentiment. Lastly, the neutral word cloud included words like _“need”_, _“food”_, _“raise”_, etc.")

## Topic Modelling
df_clean = pd.read_excel("c:/Users/bryan/OneDrive/Desktop/Abs Clean Data (tweets) - V2.xlsx", parse_dates=['date',])
df_clean.drop(["Unnamed: 0"], axis=1, inplace=True)

data = df_clean.abs_clean_tweets.values.tolist()

# Create dic to count the words
count_dict_alex = {}
for doc in df_clean['abs_clean_tweets']:
    for word in doc.split():
        if word in count_dict_alex.keys():
            count_dict_alex[word] +=1
        else:
            count_dict_alex[word] = 1
            
# Remove words that occur less than 10 times
low_value = 10
bad_words = [key for key in count_dict_alex.keys() if count_dict_alex[key] < low_value]
# Create a list of lists - Each document is a string broken to a list of words
corpus = [doc.split() for doc in df_clean['abs_clean_tweets']]
clean_list = []
for document in corpus:
    clean_list.append([word for word in document if word not in bad_words])

corpora_dict = corpora.Dictionary(clean_list)
corpus = [corpora_dict.doc2bow(line) for line in clean_list]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=corpora_dict,
                                           num_topics=5,
                                           random_state=1,
                                           update_every=1, 
                                           chunksize=100,
                                           passes=100,
                                           iterations=500,
                                           alpha=0.3,
                                           eta=0.9, 
                                           per_word_topics=True)

st.header("Topic Modelling")
with st.expander('Details'):
    df = pd.DataFrame({
    'Topics': ["Encourage people to not be embarrassed to raise a white flag as a sign of seeking help", "Criticism against the government", "Citizens offer help in any way possible on Twitter", "Miscellaneous", "The organization of food bank and donation for those in need"],
    'Keywords': ["flag, white, help, raise, ask, need, house, mean, home, fly, call, sign, front, neighbor, media", "people, campaign, government, initiative, minister, politicians, good, political, support, fail, take, hijack, use, want, failure", "help, need, us, please, let, anyone, want, may, know, God, share, thank, one, spread, support", "people, make, like, want, even, go, know, work, think, say, one, many, right, still", "food, need, assistance, provide, bank, family, families, send, buy, donate, receive, rice, donations, thank"]
    })
    st.table(df)
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, corpora_dict)
    html_string = pyLDAvis.prepared_data_to_html(vis)
    components.v1.html(html_string, width=1300, height=800)

# st.header("Sentiment Analysis by Topic")
# with st.expander('Details'):
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image("xxx")
#     with col2:
#         st.text("Sentiments by Topic in Radar Chart")
