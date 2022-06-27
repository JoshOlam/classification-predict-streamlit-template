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
			
	if selection == 'Classify Tweets':
		st.info("Interact with our model by classifying some tweets or upload a '.csv' file with tweets to classify")
		data_source = ['Select option', 'Single text', 'Dataset'] #Defines the type of input to classify
		source_selection = st.selectbox('Select your preferred data input option:', data_source)

        # Load Our Models
		def load_prediction_models(model_file):
			loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
			return loaded_models

        # Getting the predictions
		def get_keys(val,my_dict):
			for key,value in my_dict.items():
				if val == value:
					return key
		
		#Single Tweet input
		if source_selection == 'Single text': 
			st.subheader('Single tweet classification')
			
			input_text = st.text_area('Enter Text (max. 140 characters):')
			all_ml_models = ["LR","NB","RFOREST","DECISION_TREE"]
			model_choice = st.selectbox("Choose ML Model",all_ml_models)
			
			st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')
			
			prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
			if st.button('Classify'):
				st.text("Original test:\n{}".format(input_text))
				text1 = cleaner(input_text) ###passing the text through the 'cleaner' function
				vect_text = tweet_cv.transform([text1]).toarray()
				if model_choice == 'LR':
					predictor = load_prediction_models("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
					# st.write(prediction)
				elif model_choice == 'RFOREST':
					predictor = load_prediction_models("resources/RFOREST_model.pkl")
					prediction = predictor.predict(vect_text)
					# st.write(prediction)
				elif model_choice == 'NB':
					predictor = load_prediction_models("resources/NB_model.pkl")
					prediction = predictor.predict(vect_text)
					# st.write(prediction)
				elif model_choice == 'DECISION_TREE':
					predictor = load_prediction_models("resources/DTrees_model.pkl")
					prediction = predictor.predict(vect_text)
					# st.write(prediction)

				
			word = ''
			if prediction == 0:
				word = '"**Neutral**". It neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				word = '"**Pro**". The tweet supports the belief of man-made climate change'
			elif prediction == 2:
				word = '**News**. The tweet links to factual news about climate change'
			else:
				word = 'The tweet do not belief in man-made climate change'
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as {}".format(word))
			final_result = get_keys(prediction,prediction_labels)
			st.success("Tweet Categorized as:: {}".format(final_result))
				
		if source_selection == 'Dataset':
            ### DATASET CLASSIFICATION ###
			st.subheader('Dataset tweet classification')
			all_ml_models = ["LR","NB","RFOREST","SupportVectorMachine", "MLR", "LDA"]
			model_choice = st.selectbox("Choose ML Model",all_ml_models)
			
			st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')
			prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
			text_input = st.file_uploader("Choose a CSV file", type="csv")
			if text_input is not None:
				text_input = pd.read_csv(text_input)

            #X = text_input.drop(columns='tweetid', axis = 1, inplace = True)
			uploaded_dataset = st.checkbox('See uploaded dataset')
			if uploaded_dataset:
				st.dataframe(text_input.head(25))
			
			col = st.text_area('Enter column to classify')

            #col_list = list(text_input[col])

            #low_col[item.lower() for item in tweet]
            #X = text_input[col]

            #col_class = text_input[col]
			if st.button('Classify'):
				st.text("Original test ::\n{}".format(text_input))
				X1 = text_input[col].apply(cleaner) ###passing the text through the 'cleaner' function
				vect_text = tweet_cv.transform([X1]).toarray()
				if model_choice == 'LR':
					predictor = load_prediction_models("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif model_choice == 'RFOREST':
					predictor = load_prediction_models("resources/Random_model.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif model_choice == 'NB':
					predictor = load_prediction_models("resources/NB_model.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif model_choice == 'SupportVectorMachine':
					predictor = load_prediction_models("resources/svm_model.pkl")
					prediction = predictor.predict(vect_text)
				elif model_choice == 'MLR':
					predictor = load_prediction_models("resources/mlr_model.pkl")
					prediction = predictor.predict(vect_text)
				elif model_choice == 'SupportVectorMachine':
					predictor = load_prediction_models("resources/simple_lda_model.pkl")
					prediction = predictor.predict(vect_text)

                
				# st.write(prediction)
				text_input['sentiment'] = prediction
				final_result = get_keys(prediction,prediction_labels)
				st.success("Tweets Categorized as:: {}".format(final_result))

                
				csv = text_input.to_csv(index=False)
				b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
				href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
				
				st.markdown(href, unsafe_allow_html=True)



	# Building out the predication page
	if selection == "Contact Us":
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
