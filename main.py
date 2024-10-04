from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 

app = FastAPI()

model = joblib.load('fake_news_detector.pkl')
vectorizer = joblib.load('vectorizer.pkl')

class DetectRequest(BaseModel):
    text: str

class DetectResponse(BaseModel):
    is_fake: bool
    confidence: float
    message: str

class TrainRequest(BaseModel):
    title: str  
    content: str  
    is_fake: bool  

class TrainResponse(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    text: str
    is_fake: bool

class FeedbackResponse(BaseModel):
    message: str

def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\d+', ' ', text)  # Remove digits
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stopwords
    return text

@app.post("/detect", response_model=DetectResponse)
async def detect_news(request: DetectRequest):
    try:
        input_data = vectorizer.transform([preprocess(request.text)])
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        is_fake = prediction[0] == 1  
        confidence = max(probabilities[0]) 

        return DetectResponse(
            is_fake=is_fake,
            confidence=confidence,
            message="Prediction completed successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    try:
        title = preprocess(request.title)
        content = preprocess(request.content)
        combined_text = title + " " + content  
        new_label = int(request.is_fake)

        X_train = vectorizer.transform([combined_text])
        Y_train = [new_label]

        if not hasattr(model, 'classes_'):  
            model.partial_fit(X_train, Y_train, classes=[0, 1])
        else:
            model.partial_fit(X_train, Y_train)

        joblib.dump(model, 'fake_news_detector.pkl')

        return TrainResponse(message="Model successfully updated with the new data.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    try:
        feedback_text = preprocess(request.text)
        feedback_label = int(request.is_fake) 

        feedback_vector = vectorizer.transform([feedback_text])

        if not hasattr(model, 'classes_'):
            model.partial_fit(feedback_vector, [feedback_label], classes=[0, 1])
        else:
            model.partial_fit(feedback_vector, [feedback_label])

        joblib.dump(model, 'fake_news_detector.pkl')

        return FeedbackResponse(message="Feedback successfully submitted and model updated.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")