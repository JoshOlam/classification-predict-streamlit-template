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
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd

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
	
	image = Image.open('resources/imgs/greener_cloud.jpg')
	st.image(image, caption = 'Greener Cloud', use_column_width=True)

	st.title("Greener Cloud")
	st.subheader("""Project Title: Climate Change Tweet Classification
							Date: June, 2022""")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["General Information", "Classify Tweets", "Contact Us"]
	selection = st.sidebar.selectbox("Choose an Option Here:", options)

	# Building out the "Information" page
	if selection == "General Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown('''This is an accurate and robust solution that gives companies (or individuals) access to a broad base of consumer sentiment,
			spanning multiple demographic and geographic categories - thus increasing their insights and informing future 
			marketing strategies.''')
		st.markdown('''The sentiment is divided into 4-categories:''')
		st.write(" ==> -1 (anti)- those who don't believe global warming is real;")
		st.write(' ==>  0 (neutral) those who neither believe nor disbelieve global warming;')
		st.write(' ==>  1 (pro) those who believe global warming is real;')
		st.write(' ==>  2 (news) contains some information or facts related to global warming.')

		st.subheader("Raw Twitter data and label")
		data_display = ['Select option', 'Header', 'Random_row', 'Full_data']
		source_selection = st.selectbox('Select desired display:', data_display)

		if source_selection == 'Header':
			st.write(raw.columns)

		if source_selection == 'Random_row':
			st.write(raw.sample())
			st.write('You can re-select this same option from the dropdown to view another random row.')

		if source_selection == 'Full_data':
			st.write(raw)


	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
