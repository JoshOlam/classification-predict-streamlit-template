"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from numpy import result_type
import scipy
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy import sparse
import base64
import csv
import string
from extras import clean
st.set_page_config(page_icon="resources/imgs/globe.png", page_title="Greener Cloud Global Warming Model")



primary_clr = st.get_option("theme.primaryColor")
txt_clr = st.get_option("theme.textColor")
    # I want 3 colours to graph, so this is a red that matches the theme:
second_clr = "#d87c7c"



# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
 
	options = ["General Information", 'Visuals', "Classify Tweets", "Contact Us"]
	selection = st.sidebar.image(
    "https://imgur.com/VIIqPzu.png",
   	width=300,)
	selection = st.sidebar.selectbox("Choose an Option Here:", options)

	# Building out the "Information" page
	if selection == "General Information":
		st.image("https://imgur.com/zUvnHT1.png")
		image = Image.open('resources/imgs/cloud_2.jpg')
		st.image(image, caption = '...touching a Greener Cloud', use_column_width=True)
		st.subheader("""Project Title: Climate Change Tweet Classification
							Date: June, 2022""")

		st.info("""General Information:
		This is an accurate and robust solution that gives companies (or individuals) access to a broad base of consumer sentiment,
		spanning multiple demographic and geographic categories - thus increasing their insights and informing future 
		marketing strategies.""")
		
		st.info('''You are required to input text (ideally a tweet relating to climate change),
		and the model will classify it according to whether or not they believe in 
		climate change.
		Below, you will find information about the data source (feel free to play around with the different options available).
		You can classify your tweets on the 'Classify Tweets' page and don't forget to check us out on the 
		"Contact Us" page; all of which you can navigate to in the sidebar.''')

		st.markdown('''Each tweet is labelled as one of the following classes:
		-  2 (News): the tweet links to factual news about climate change;
		-  1 (Pro): the tweet supports the belief of man-made climate change;
		-  0 (Neutral): the tweet neither supports nor refutes the belief of man-made climate change;
		- -1 (Anti): the tweet does not believe in man-made climate change.''')

		st.subheader("Raw Twitter data and label")
		data_display = ['Select option', 'Header', 'Random_row', 'Full_data']
		source_selection = st.selectbox('Select desired display:', data_display)

		if source_selection == 'Header':
			st.write('Display of the columns in the dataset:')
			st.write(raw.columns)

		if source_selection == 'Random_row':
			st.write('Display of a random row in the dataset:')
			st.write(raw.sample())
			st.write('You can re-select this same option from the dropdown to view another random row.')

		if source_selection == 'Full_data':
			st.write("Display of the full data(don't get overwhelmed :smiley:):")
			st.write(raw)
   
   	# Building out the Visuals page
	if selection == "Visuals":
		st.image("https://imgur.com/HL0NhVQ.png")


		visual_options = ["Visuals (Home)", "Bar Graphs", "Word Clouds", "Model performance", "PieChart"]
		visual_options_selection = st.selectbox("Which visual category would you like?",
		visual_options)

		if visual_options_selection == "Visuals (Home)":
			st.image('https://imgur.com/5QEtky5.png', width=730)
   
		if visual_options_selection == "Model performance":
			per_listed = ['F1_measure', 'Fit_time']
			per_list = st.selectbox('I would like to view the...', per_listed)

			if per_list == 'F1_measure':
				st.subheader('F1 scores of the various models used')
				st.image('https://imgur.com/o1zYqC8.png', width=730)
			if per_list == 'Fit_time':
				st.subheader('Fit time of the various models used')
				st.image('https://imgur.com/N4DVf2s.png', width=730)
       
		if visual_options_selection == "PieChart":
			st.subheader("Percentage Distribution of Our Train Dataset")
			st.image("https://imgur.com/oKishVn.png", width=730)

		if visual_options_selection == "Bar Graphs":
			st.image('https://i.imgur.com/p3J5Gcw.png')

			bar_nav_list = ['Sentiment distribution of raw data', 
			'Most common words in various sentiment classes (raw data)', 
			'Most common words in various sentiment classes (cleaned data)']

			bar_nav = st.selectbox('I would like to view the...', bar_nav_list)


			if bar_nav == 'Sentiment distribution of raw data':
				st.subheader('Sentiment Distribution of Raw Data')
				st.image('https://i.imgur.com/JT9HzVW.png', width=700)
				st.write("This graph shows how the raw data was distruted amongst the various sentiment classes.")
				st.write("The classes can be interpreted as follows:")				
				st.write("2: News  --  the tweet links to factual news about climate change")
				st.write("1: Pro  --  the tweet supports the belief of man-made climate change")
				st.write("0: Neutral  --  the tweet neither supports nor refutes the belief of man-made climate change")
				st.write("-1: Anti  --  the tweet does not believe in man-made climate change")
		
			if bar_nav == 'Most common words in various sentiment classes (raw data)':

				raw_common_words_list = ['All tweets', 'Negative tweets', 'Positive tweets', 
				'News-related tweets', 'Neutral tweets']
				raw_common_words = st.radio('Raw Data Sentiment Classes:', raw_common_words_list)

				if raw_common_words == 'All tweets':
					st.subheader('Common Words in all Tweets')
					st.image('https://i.imgur.com/HSdxDP9.png', width=700)

				if raw_common_words == 'Negative tweets':
					st.subheader('Common Words in Negative Tweets')
					st.image('https://i.imgur.com/FqnrN1Y.png', width=700)

				if raw_common_words == 'Positive tweets':
					st.subheader('Common Words in Positive Tweets')
					st.image('https://i.imgur.com/glF9Z0M.png', width=700)

				if raw_common_words == 'News-related tweets':
					st.subheader('Common Words in News-related Tweets')
					st.image('https://i.imgur.com/fWDhrTL.png', width=700)

				if raw_common_words == 'Neutral tweets':
					st.subheader('Common Words in Neutral Tweets')
					st.image('https://i.imgur.com/LEnkE9V.png', width=700)


			if bar_nav == 'Most common words in various sentiment classes (cleaned data)':

				clean_common_words_list = ['All tweets', 'Negative tweets', 'Positive tweets', 
				'News-related tweets', 'Neutral tweets']
				clean_common_words = st.radio('Cleaned Data Sentiment Classes:', clean_common_words_list)

				if clean_common_words == 'All tweets':
					st.subheader('Common Words in all Tweets')
					st.image('https://i.imgur.com/1aOr9DD.png', width=700)

				if clean_common_words == 'Negative tweets':
					st.subheader('Common Words in Negative Tweets')
					st.image('https://i.imgur.com/7Mp2NRX.png', width=700)

				if clean_common_words == 'Positive tweets':
					st.subheader('Common Words in Positive Tweets')
					st.image('https://i.imgur.com/3SDTh6c.png', width=700)

				if clean_common_words == 'News-related tweets':
					st.subheader('Common Words in News-related Tweets')
					st.image('https://i.imgur.com/sDFn7PF.png', width=700)

				if clean_common_words == 'Neutral tweets':
					st.subheader('Common Words in Neutral Tweets')
					st.image('https://i.imgur.com/0ur3J3C.png', width=700)

		if visual_options_selection == "Word Clouds":
			st.image('https://i.imgur.com/QDhrJTR.png')

			wc_nav_list = ['Most Common Words (Raw Data)', 
			'Most Common Words (Cleaned Data)']

			wc_nav = st.selectbox('I would like to view the...', wc_nav_list)


			if wc_nav == 'Most Common Words (Raw Data)':
				st.subheader('Most Common Words for all Tweets (raw data)')
				st.image('https://i.imgur.com/MsGuWFv.png')

			if wc_nav == 'Most Common Words (Cleaned Data)':

				wc_clean_list = ['All tweets', 'Negative tweets', 'Positive tweets', 
				'News-related tweets', 'Neutral tweets']
				wc_clean = st.radio('Cleaned Data Sentiment Classes:', wc_clean_list)

				if wc_clean == 'All tweets':
					st.subheader('Common Words in all Tweets')
					st.image('https://i.imgur.com/0MeELLk.png')

				if wc_clean == 'Negative tweets':
					st.subheader('Common Words in Negative Tweets')
					st.image('https://i.imgur.com/fc0Aa9l.png')

				if wc_clean == 'Positive tweets':
					st.subheader('Common Words in Positive Tweets')
					st.image('https://i.imgur.com/4LSqpBm.png')

				if wc_clean == 'News-related tweets':
					st.subheader('Common Words in News-related Tweets')
					st.image('https://i.imgur.com/cmrDvhk.png')

				if wc_clean == 'Neutral tweets':
					st.subheader('Common Words in Neutral Tweets')
					st.image('https://i.imgur.com/AJMYNOu.png')


	#Build the "Classify Tweets" Page

	if selection == "Classify Tweets":
		st.image("https://imgur.com/HL0NhVQ.png")

		st.info("Interact with our model by classifying some 'Single Tweets' or upload a '.csv' file with tweets to classify")
		data_source = ['Select option', 'Single Tweet','Dataset'] #Defines the type of input to classify
		source_selection = st.selectbox('Select your preferred data input option:', data_source)
		st.info('Make Predictions of your Tweet(s) using our ML Model')

		all_models = ["Logistic_Regression (base)", "MultinomialNB" ,"Linear_SVC", "SGDClassifier", "SVC"]


		if source_selection == "Single Tweet":
			st.subheader('Single tweet classification')
			tweet_text = st.text_area("Enter Tweet (max. 120 characters):")
			tweet_text = clean(tweet_text)
			selected_model = st.selectbox("Select preferred Model to use:", all_models)

			
			if selected_model == "Logistic_Regression (base)":
				model = "resources/Lrmodel.pkl"
			elif selected_model == "Linear_SVC":
				model = "resources/Linsvcmodel.pkl"
			elif selected_model == "MultinomialNB":
				model = "resources/multimodel.pkl"
			elif selected_model == "SGDClassifier":
				model = "resources/SGDmodel.pkl"
			else:
				model = "resources/SVCmodel.pkl"

			if st.button ("Classify"):
				st.text("Your cleaned input tweet: \n{}".format(tweet_text))
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				predictor = joblib.load(open(os.path.join(model), "rb"))
				prediction = predictor.predict(vect_text)

				result = ""
				if prediction == 0:
					result = '"**Neutral**"; it neither supports nor negates the belief of man-made climate change'
				elif prediction == 1:
					result = '"**Pro**"; it  supports the belief of man-made climate change'
				elif prediction == 2:
					result = '"**News**"; it contains factual links to climate change'
				else:
					result = '"**Negative**"; it negates the belief of man-made climate change'
				
				st.success("Categorized as {}".format(result))

		if source_selection == "Dataset":
			st.subheader("Classification for Dataset")
			selected_model = st.selectbox("Select preferred Model to use:", all_models)
			loaded_dataset = st.file_uploader("Upload your .csv file here", type = 'csv')

			if loaded_dataset is not None:
				df = pd.read_csv(loaded_dataset)

				if st.checkbox("Preview Uploaded Dataset"):
					st.dataframe(df.head(5))
				
				columns = df.columns.tolist()
				column = st.selectbox("Select the Column with the Tweets to Classify", columns)
				if selected_model == "Logistic_Regression (base)":
					model = "resources/Lrmodel.pkl"
				elif selected_model == "Linear_SVC":
					model = "resources/Linsvcmodel.pkl"
				elif selected_model == "MultinomialNB":
					model = "resources/multimodel.pkl"
				elif selected_model == "SGDClassifier":
					model = "resources/SGDmodel.pkl"
				else:
					model = "resources/SVCmodel.pkl"
     
     
				if st.button ("Classify"):
					if column == 'message':
						df[column] = df[column].apply(clean)
						df1 = np.array(df[column])
						vect_text = tweet_cv.transform(df1)
						predictor = joblib.load(open(os.path.join(model), "rb"))
						prediction = predictor.predict(vect_text) 
						submission = pd.DataFrame(
    								{'tweetid': df['tweetid'],
     										'sentiment': prediction
    								})
						submission["sentiment"] = submission["sentiment"].replace({0:'Neutral',1:'Pro',2:'News',-1:'Anti'}, inplace=False)
						st.write("YOUR CLEANED TWEET")
						st.dataframe(pd.DataFrame(df1))
						st.success("Sentiments has been analyzed and compiled into a CSV format, Thank you for your patronage")
						result = submission.to_csv(index=False)
						filename = 'result.csv'
						title = "Download CSV file"
						b64 = base64.b64encode(result.encode()) # some strings <-> bytes conversions necessary here
						payload = b64.decode()
						href = f'<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
						st.markdown(href, unsafe_allow_html=True)

		
	if selection == "Contact Us":
		st.title("The Management Team")
		st.info("Here is the awesome Team behind this robust model 👇🏿")

		elvis = Image.open("resources/imgs/Elvis.jpg")
		title = "Team Lead: Elvis"
		st.image(elvis, caption = title, width = 200)
		link='[Linkedin](https://www.linkedin.com/in/elvis-esharegharan-822a6633/)'
		st.markdown(link,unsafe_allow_html=True)
		st.write(" ")

		elizabeth = Image.open("resources/imgs/Elizabeth.jpg")
		st.image(elizabeth, caption = "Administrative Head: Elizabeth", width = 200)
		link='[Linkedin](https://www.linkedin.com/in/elizabeth-ajabor-42234422b)'
		st.markdown(link,unsafe_allow_html=True)
		st.write(" ")

		mac = Image.open("resources/imgs/MacMatthew.jpg")
		st.image(mac, caption = "Technical Lead: MacMatthew", width = 200)
		link='[Linkedin](https://www.linkedin.com/in/macmatthew-ahaotu-388a90123/)'
		st.markdown(link,unsafe_allow_html=True)
		st.write(" ")

		bongani = Image.open("resources/imgs/Bongani.jpg")
		st.image(bongani, caption = "Deputy Tech Lead: Bongani", width = 200)
		st.write(" ")

		josh = Image.open("resources/imgs/Josh.jpg")
		st.image(josh, caption = "Communications Lead: Josh", width = 200)
		link='[Linkedin](https://www.linkedin.com/in/joshua-olalemi)'
		st.markdown(link,unsafe_allow_html=True)
		st.write(" ")

		izu = Image.open("resources/imgs/Izunna.jpg")
		st.image(izu, caption = "Dep. Communications Lead: Izunna", width = 200)
		link='[Linkedin](https://ng.linkedin.com/in/izunna-eneude-77743492)'
		st.markdown(link,unsafe_allow_html=True)
		






		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
