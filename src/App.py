import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from PIL import Image, ImageDraw, ImageOps

nltk.download('stopwords')

def clean(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    return ' '.join(word for word in text.split() if word not in stop_words)

def create_contact_section():
    st.markdown("---")
    st.header("Contact Information📩")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            # Open the image
            image = Image.open(r"F:\Data Science Projects\Text Document Classification\src\profile.jpg")
            
            # Create a mask to make the image rounded
            width, height = image.size
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, width, height), fill=255)
            
            # Apply the mask to the image
            image = ImageOps.fit(image, (width, height), method=0, bleed=0.0)
            image.putalpha(mask)
            
            # Display the image in Streamlit with the rounded shape
            st.image(image, width=150, caption="Profile Picture")
            
        except Exception as e:
            st.error("Error loading profile picture")
    
    with col2:
        st.markdown("""
        ### Connect with me 
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahermuhamed/)
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sahermuhamed1)
        #### email: sahermuhamed176@gmail.com
        """)

def main():
    st.title("Text Category Classifier🤖")
   
    # Load data
    df = pd.read_csv(r'F:\Data Science Projects\Text Document Classification\data\df_file.csv')
    text_column = df.columns[0]
    label_column = df.columns[1]
   
    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('nb', MultinomialNB())
    ])
   
    processed_text = df[text_column].apply(clean).apply(remove_stopwords)
    pipeline.fit(processed_text, df[label_column])
   
    input_text = st.text_area("Enter your text:", height=150)
   
    if st.button("Classify"):
        if input_text:
            processed_input = remove_stopwords(clean(input_text))
            prediction = pipeline.predict([processed_input])[0]
            # 0: 'Politics', 1: 'Sport', 2: 'Technology', 3: 'Entertainment', 4: 'Business'
            if prediction == 0:
                prediction = "Politics"
            elif prediction == 1:
                prediction = "Sport"
            elif prediction == 2:
                prediction = "Technology"
            elif prediction == 3:
                prediction = "Entertainment"
            else:
                prediction = "Business"
            st.success(f"Category: {prediction}")
        else:
            st.warning("Please enter text to classify.")
            
    # Add contact section
    create_contact_section()

if __name__ == "__main__":
    main()