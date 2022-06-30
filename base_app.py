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
import emoji
from nltk.corpus import stopwords
from nltk import pos_tag
import seaborn as sns
import re
from nlppreprocess import NLP
nlp = NLP()
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy import sparse

st.set_page_config(page_icon="resources/imgs/globe.png", page_title="Greener Cloud Global Warming Model")


blank, title_col, blank = st.columns([2,3.5,2])
title_col.title("Greener Cloud")

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
	options = ["General Information", "Classify Tweets", "Contact Us"]
	selection = st.sidebar.selectbox("Choose an Option Here:", options)

	# Building out the "Information" page
	if selection == "General Information":
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

	#Build the "Classify Tweets" Page

	if selection == "Classify Tweets":
		st.info("Interact with our model by classifying some 'Single Tweets' or upload a '.csv' file with tweets to classify")
		data_source = ['Select option', 'Single Tweet', 'Dataset'] #Defines the type of input to classify
		source_selection = st.selectbox('Select your preferred data input option:', data_source)
		st.info('Make Predictions of your Tweet(s) using our ML Model')

		all_models = ["Logistic_Regression", "Linear_Regression" ,"Linear_SVC"]



		if source_selection == "Single Tweet":
			st.subheader('Single tweet classification')
			tweet_text = st.text_area("Enter Tweet (max. 120 characters):")
			
			selected_model = st.selectbox("Select preferred Model to use:", all_models)

			
			if selected_model == "Logistic_Regression":
				model = "resources/Logistic_regression.pkl"
			elif selected_model == "Linear_SVC":
				model = "resources/LIN_SVC_model.pkl"
			else:
				model = "resources/Lin_Reg_model.pkl"

			if st.button ("Classify"):
				st.text("Your inputted tweet: \n{}".format(tweet_text))
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
				
				columns = [df.columns]
				selected_column = st.selectbox("Select the Column with the Tweets to Classify", columns)
				selected_model_dataset = st.selectbox("Select preferred Model to use:", all_models)
				if selected_model_dataset == "Logistic_Regression":
					model = "resources/Logistic_regression.pkl"
				elif selected_model_dataset == "Linear_SVC":
					model = "resources/LIN_SVC.pkl"
				else:
					model = "resources/Lin_Reg_model.pkl"
				
				for col in selected_column:
					vect_text = tweet_cv.transform([df[col]]).toarray()
					predictor = joblib.load(open(os.path.join(model), "rb"))
					prediction = predictor.predict(vect_text)
					
					result = ""
					if prediction == 0:
						result = 'Neutral'
					elif prediction == 1:
						result = 'Pro'
					elif prediction == 2:
						result = 'News'
					else:
						result = 'Negative'
					
					result_df["Sentiment"] = result

					st.success("Tweet Categorized as: {}".format(result_df))
					df_result = result_df.to_csv(iindex=False)
					b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
					href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
					st.markdown(href, unsafe_allow_html=True)

	if selection == "Contact Us":
		st.title("The Management Team")
		st.info("Here is the awesome Team behind this robust model 👇🏿")

		elvis = Image.open("resources/imgs/Elvis.jpg")
		title = "Team Lead: Elvis"
		st.image(elvis, caption = title, width = 500)

		elizabeth = Image.open("resources/imgs/Elizabeth.jpg")
		st.image(elizabeth, caption = "Administrative Head: Elizabeth", width = 300)


		mac = Image.open("resources/imgs/MacMatthew.jpg")
		st.image(mac, caption = "Technical Lead: MacMatthew", width = 300)

		bongani = Image.open("resources/imgs/Bongani.jpg")
		st.image(bongani, caption = "Deputy Tech Lead: Bongani", width = 300)

		josh = Image.open("resources/imgs/Josh.jpg")
		st.image(josh, caption = "Communications Lead: Josh", width = 300)

		izu = Image.open("resources/imgs/Izunna.jpg")
		st.image(izu, caption = "Technical Lead: Izunna", width = 300)







		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
