"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

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
# dependencies
import os
import streamlit as st
import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
import nltk
import numpy as np
from nltk.corpus import stopwords

nltk.download('stopwords')
# Load test data (assuming test.csv has a column 'category' containing labels)
test_data = pd.read_csv('test.csv')
y_true = test_data['category'].tolist()

base_path = os.getcwd()


# Function to preprocess text
def preprocess_text(text, stopwords):
    cleaned_text = text.lower()  # Lowercase the text
    cleaned_text = cleaned_text.split()  # Tokenization (splitting into words)
    cleaned_text = [word for word in cleaned_text if word not in stopwords]  # Remove stopwords
    cleaned_text = ' '.join(cleaned_text)  # Join the cleaned tokens back into a string
    return cleaned_text

# Function to process data for prediction
def process_data_for_prediction(input_text_series, tfidf_vectorizer, selector, scaler):
    input_text = input_text_series.astype(str)
    input_text_cleaned = input_text.apply(lambda x: preprocess_text(x, stopwords.words('english')))
    vectorized_data = tfidf_vectorizer.transform(input_text_cleaned)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    vectorized_data = selector.transform(vectorized_data)
    selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
    vectorized_df = pd.DataFrame(vectorized_data.toarray(), columns=selected_feature_names)
    vectorized_data_scaled = scaler.transform(vectorized_df)
    return vectorized_data_scaled

# Function to predict category
def predict_category(input_paragraph, label_encoder, tfidf_vectorizer, selector, scaler, model):
    vectorized_data_scaled = process_data_for_prediction(pd.Series([input_paragraph]), tfidf_vectorizer, selector, scaler)
    prediction = model.predict(vectorized_data_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label, prediction

# Function to load model, classification report, and metadata from meta.yaml
def load_model_and_metadata(model_folder):
    meta_path = os.path.join(model_folder, 'meta.yaml')
    with open(meta_path, 'r') as file:
        meta_data = yaml.safe_load(file)
    
    artifact_uri = os.path.join(model_folder, 'artifacts')
    model_path = os.path.join(artifact_uri, 'model', 'model.pkl')
    model = joblib.load(model_path)
    
    
    
    Newshub_app = os.path.join(base_path, 'data_pre')
    label_encoder_path = os.path.join(Newshub_app, 'label_encoder.pkl')
    tfidf_vectorizer_path = os.path.join(Newshub_app, 'tfidf_vectorizer.pkl')
    selector_path = os.path.join(Newshub_app, 'selectkbest.pkl')
    scaler_path = os.path.join(Newshub_app, 'scaler.pkl')

    label_encoder = joblib.load(label_encoder_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    selector = joblib.load(selector_path)
    scaler = joblib.load(scaler_path)
    
    return label_encoder, tfidf_vectorizer, selector, scaler, model, meta_data


def display_model_card(selected_model):
    st.sidebar.markdown(
        f"""
        <div style='padding: 8px; border: 1px solid #ccc; border-radius: 5px; background-color: #f0f0f0;'>
            <h4 style='margin-bottom: 5px; font-size: 14px;'>Selected Model</h4>
            <p style='font-size: 14px; margin-top: 5px;'>Model Name: <b>{selected_model}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Vectorizer
#news_vectorizer = open("streamlit/tfidfvect.pkl","rb")
#test_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
#raw = pd.read_csv("streamlit/train.csv")

# The main function where we will build the actual app
def main():
    # Set the page title and icon
    st.set_page_config(page_title="NewsHub Inc. Article Classifier", page_icon=":newspaper", layout="wide")

    # Create a sidebar with navigation options
    st.sidebar.title("Content")
    col1, col2 = st.sidebar.columns([2, 1])  # Divide the sidebar into two columns

    with col1:
        image_path = os.path.join(base_path, 'pictures', 'home.jpg')
        st.image(image_path, width=300)  # Place the logo image in the first column

    with col2:
       options = ["Home", "About", "EDA & Overview", "Models", "Limitations", "Calendar"]
       selected_option = st.sidebar.selectbox("Select a page:", options)  # Place the navigation options in the second column


  # Build out the home page
    if selected_option == "Home":

        image_path2 = os.path.join(base_path, 'pictures', 'home.jpg')
        st.image(image_path2, width=300)
        st.title("NewsHub Classifier App")
        st.image("https://media.istockphoto.com/id/1283277781/photo/read-online-news-media-on-laptop.jpg?s=612x612&w=0&k=20&c=D2EcPDQ3YbPPxGEWkB8iWvpt7LqNso9Sa-bX-9Q3RL4=", width=1500)
        st.header("Our Mission")
        st.write("""
            **Our Vision:** To be the most trusted and widely-read news source, providing high-quality, unbiased, and engaging content to a diverse audience. We aim to be the go-to platform for people seeking accurate and timely information on local, national, and global news, as well as in-depth analysis and commentary.
            """)
        st.header("Our Values")
        st.write("""
            **Truth:** We are committed to factual reporting and accuracy, striving to provide our audience with the most reliable and trustworthy information.
            
            **Inclusivity:** We believe that everyone deserves access to quality news, regardless of their background, demographics, or geographic location. Our platform will cater to diverse perspectives and voices.
            
            **Innovative Storytelling:** We will push the boundaries of traditional journalism by incorporating multimedia content, interactive features, and immersive experiences to engage our audience.
           
            **Transparency:** We will be open and transparent in our reporting, revealing our sources, methodology, and biases to maintain trust with our readers.
            """)
        # video section (for advertisement)
        st.header("News Update:")
        st.header("The president announces plans to eradicate poverty")
        video_url = "https://youtu.be/NbGTe-d0QVY?si=xp4wq3I51pedwoDc"
        st.video(video_url)

    elif selected_option == "EDA & Overview":
        st.title("EDA and Overview")
        st.markdown("""
<div class="summary-container">

<h1 class="header-text">Welcome!</h1>

<p class="summary-text">
We've been exploring a vast dataset of news articles from various sources. 
Exploratory Data Analysis (EDA) is like taking a first look at a new place – it helped us understand the data's landscape and uncover interesting patterns. This exploration provided us with valuable insights that will be crucial in building effective models for categorizing news articles.
</p>

<h2>What's next?</h2>

<p class="summary-text">
On this page, we'll provide a high-level overview of the key takeaways from our analysis.
</p>

</div>
""", unsafe_allow_html=True)

        st.header("Entity Word Clouds Gallery")
        st.write("A visual representation of the most frequent words in the dataset.")

        
        wordcloud_paths = [
    		os.path.join(base_path, 'pictures', 'output_eda1.png'),
    		os.path.join(base_path, 'pictures', 'output_eda2.png'),
    		os.path.join(base_path, 'pictures', 'output_eda3.png'),
    		os.path.join(base_path, 'pictures', 'output_eda4.png'),
    		os.path.join(base_path, 'pictures', 'output_eda5.png')
		]

        
        cols = st.columns(len(wordcloud_paths))
        for i, col in enumerate(cols):
            col.image(wordcloud_paths[i], caption=f'Entity Word Cloud {i+1}', use_column_width=True)
        
        st.markdown("""
<div class="summary-container">
 
  <p class="summary-text">The word clouds above provide insights into the most frequent words in our news article dataset. Here's what they tell us::</p>
  <ul class="summary-text">
    <li><strong>Key Themes:</strong> "Said", "film" and "company" stand out, indicating focus.</li>
    <li><strong>Content:</strong> They reveal the most covered subjects, shaping our understanding.</li>
    <li><strong></strong> These insights help categorize articles effectively, reaching the right audience.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
        
        st.header("UMap Projection")
        st.write("UMap Projection of Document Clusters")

        # Display UMap projection image
        st.image(os.path.join(base_path, 'pictures', 'output.png'), caption='UMap Projection of Document Clusters', use_column_width=True, width=300)

        # Display using st.markdown with Markdown syntax
        st.markdown("""
<div class="summary-container">
  <h2 class="header-text">Document Clusters (UMAP)</h2>
  <ul class="summary-text">
    <li>UMAP reveals clusters suggesting thematic groupings (e.g., education).</li>
    <li>Highlights: isolated themes (education) and overlapping themes (business-technology).</li>
  </ul>
</div>
""", unsafe_allow_html=True)

        st.header("Overview Summary")
        st.markdown("""
<style>
  .summary-container {
    background-color: #2c3e50; 
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
  }
  .header-text {
    color: #0066cc; /* Dark blue color */
    margin-bottom: 10px;
  }
  .summary-text {
    color: #ffffff; /* White color */
    font-size: 16px;
    line-height: 1.6;
  }
</style>

<div class="summary-container">
  <h3 class="header-text">Key Findings</h3>
  <ul class="summary-text">
    <li>Identified hidden insights across categories and within articles.</li>
    <li>Explored connections between topics and categories, refining models for better predictions and insights.</li>
    <li>Highlights include isolated themes, cluster shapes reflecting topic distribution, and overlapping.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
    
    elif selected_option == "Models":
        st.title("Model Page")
        st.write("""
            Our app supports three different classification models: SVM, Random Forest, and Gradient Boosting.
            Select a model from the sidebar and input text for classification.
        """)

        st.subheader("Enter Text for Classification")
        input_paragraph = st.text_area("Input Text:", value="Enter your text here.")
        model_choice = st.sidebar.selectbox("Choose a Model", ["SVM", "Random Forest", "Gradient Boosting"])

        

        if model_choice == "SVM":
            
            model_folder = os.path.join(base_path, 'mlruns', '0', 'fa12835b82dd42c3bb8400d3b6982f49')

        elif model_choice == "Random Forest":
            model_folder = os.path.join(base_path, 'mlruns', '0', '24e6911e68314c02a57e6c8f64748f49')

        elif model_choice == "Gradient Boosting":
            model_folder = os.path.join(base_path, 'mlruns', '0', '8a4d5b797c2c4a289ac2694bfd7ba77d')

        
        label_encoder, tfidf_vectorizer, selector, scaler, model, meta_data = load_model_and_metadata(model_folder)

        display_model_card(model_choice)

        

        if st.button("Classify"):
            if input_paragraph.strip() == "Enter your text here." or input_paragraph.strip() == "":
                st.error("Please enter valid text for classification.")
            else:
                with st.spinner("Predicting..."):
                    # Load model and metadata
                    label_encoder, tfidf_vectorizer, selector, scaler, model, meta_data = load_model_and_metadata(model_folder)
                    # Predict category
                    predicted_label, prediction = predict_category(input_paragraph, label_encoder, tfidf_vectorizer, selector, scaler, model)

                st.success(f"Prediction complete!: {predicted_label}")
                
                

        
        
        


    
    
# Build out the About page
    elif selected_option == "About":
        st.title("About Us")
        st.write("Newshub is a media company that's pushing the boundaries of innovation in journalism.It has been around for over ten years, with a strong reputation for delivering high-quality content to its audience.The company has developed an AI-powered article classifier app for the editorial department, which uses machine learning and natural language processing to help editors classify articles quickly and accurately. This allows editors to focus on higher-level tasks like writing, editing, and storytelling.At Newshub, we're committed to using technology to enhance the journalism experience for both consumers and professionals. Join us as we continue to shape the future of media.")
        st.header( "Meet the Minds Behind Our Technology")
        st.write("Our data science team is comprised of experienced professionals who worked closely together to develop our AI-powered article classifier app. Their expertise in machine learning and natural language processing was crucial in ensuring the app's accuracy, efficiency, and effectiveness.")
        st.write("- **Kamogelo Nkwana:** Lead Data Scientist")
        st.write("- **Malusi Ngubane:** Project Manager")
        st.write("- **Michael Thema:** Data Scientist")
        st.write("- **Nduvho Nesengani Davhana:** Data Scientist")
        st.write("- **Johannes Malefetsane Makgetha:** Data Scientist")
    

        
       
    #Add collaborations
        st.header("Our partnerships include:")
        st.image(os.path.join(base_path, 'pictures', 'avout.jpg'))
        st.header("Frequently Asked Questions")
        faq={"How do I use the app to classify articles?": "Simply input the text of the article and press the \"predict\" button. The app will automatically classify which category the article belongs to.",
    "What do I do if the app incorrectly classifies an article?": "If the app makes an incorrect classification, please provide feedback through the designated feedback option to help us improve the accuracy of the model.",
    "How accurate is the article classifier?": "While our models are highly accurate. Continuous improvements are being made based on user feedback and advancements in technology.",
    "Can I use the app on my mobile device?": "Yes, the app is designed to be mobile-friendly and can be used on various devices, including smartphones and tablets.",
    "Is the classification instant or does it take time?": "The classification is typically instant. However, processing time may vary based on server load and the length of the article."}
        for q,a in faq.items():
            expander=st.expander(q)
            expander.markdown(a)
        
        

    

    elif selected_option =="Calendar":
        st.header("Upcoming Events")
        st.image(os.path.join(base_path, 'pictures', 'Calendar.jpeg'))
        notification_text = "Tech Case Study due at 10:30" 
        if selected_option == "Calendar" and st.button("Show Notification"):
            with st.expander("Notification"):
                st.info(notification_text)
     

    
    elif selected_option == "Limitations":
	    
        st.markdown("# Strengths and Limitations")
        st.markdown("Our article classification app has its strengths and limitations. Here's a balanced view:")
	
	
        st.markdown("## Strengths")
        st.markdown("- **User-Friendly Interface**: The app provides a straightforward and intuitive user interface, making it easy for users to input text and receive classification results.")
        st.markdown("- **Versatile Models**: The app utilizes three powerful machine learning models—Support Vector Machine (SVM), Random Forest, and Gradient Boosting—which enhance the accuracy and robustness of classifications.")
        st.markdown("- **Flexible Text Input**: The app can classify articles of varying lengths. Whether you have short snippets or lengthy articles, the app can handle text of various sizes.")
	
        st.markdown("## Limitations")
        st.markdown("- **Limited Dataset**: Our model is trained on a limited dataset, which may not cover all possible topics or styles.")
        st.markdown("- **Noise and Ambiguity**: Articles can be noisy or ambiguous, making it challenging for our model to accurately classify them.")
        st.markdown("- **Inadequate Handling of Meta-Data**: The app relies heavily on meta-data to make classification decisions. It may not perform well if this information is missing or inaccurate.")
        st.markdown("- **Domain-Specific Knowledge**: The app may struggle with articles from domains or topics that are outside its training data or not well-represented in its categories.")
        
        st.header("Rating")
        rating = st.slider("Rate our App", 1, 3, 5)
        if st.button("Submit"):
            st.write("You rated us:", rating)
        
   
         

        

       
     


  
 
   
            
            
     



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

    
 

    

    
 
