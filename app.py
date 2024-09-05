# Import necessary libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model for response generation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load pre-trained sentiment analysis and emotion detection models
# Ensure these paths point to your actual model files
sentiment_model = tf.keras.models.load_model('model1.h5')
emotion_model = tf.keras.models.load_model('model2.h5')

# Streamlit UI
st.title("Emotion-Aware Neural Chatbot")
st.write("Type your message below to chat with the Emotion-Aware Chatbot.")

# Text input for the user
user_input = st.text_input("You:", "")

# Preprocessing function (modify according to your model)
def preprocess_text(text):
    # This is a placeholder function. Update it according to your preprocessing steps.
    return text

# Function to predict sentiment (positive, negative, neutral)
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    # Replace with your own model's prediction logic
    prediction = sentiment_model.predict([preprocessed_text])
    sentiment_class = np.argmax(prediction)  # Assuming the model outputs one-hot encoding
    if sentiment_class == 0:
        return "negative"
    elif sentiment_class == 2:
        return "neutral"
    else:
        return "positive"

# Function to predict emotions (valence, arousal, dominance)
def predict_emotion(text):
    preprocessed_text = preprocess_text(text)
    # Replace with your own model's prediction logic
    prediction = emotion_model.predict([preprocessed_text])
    emotions = {
        'valence': prediction[0][0],  # Assuming first column is valence
        'arousal': prediction[0][1],  # Assuming second column is arousal
        'dominance': prediction[0][2] # Assuming third column is dominance
    }
    return emotions

# Function to generate responses using GPT-2 based on detected sentiment and emotion
def generate_gpt_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt_model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to generate chatbot responses
def generate_response(sentiment, emotion):
    # Customize prompts based on detected sentiment and emotion
    if sentiment == 'negative':
        prompt = "User is feeling sad. Respond in a supportive and comforting manner."
    elif sentiment == 'positive':
        prompt = "User is feeling happy. Respond in a cheerful and positive manner."
    else:
        prompt = "User is feeling neutral. Respond accordingly."

    # Add emotional state to the prompt
    prompt += f" Valence: {emotion['valence']:.2f}, Arousal: {emotion['arousal']:.2f}, Dominance: {emotion['dominance']:.2f}."
    
    # Generate a GPT-2-based response
    response = generate_gpt_response(prompt)
    return response

# Main chatbot logic
if user_input:
    # Predict the sentiment and emotion from the user input
    sentiment = predict_sentiment(user_input)
    emotion = predict_emotion(user_input)

    # Generate a chatbot response based on the sentiment and emotion
    response = generate_response(sentiment, emotion)

    # Display the chatbot's response
    st.write(f"Chatbot: {response}")
