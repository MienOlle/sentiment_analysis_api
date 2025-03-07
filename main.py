from huggingface_hub import hf_hub_download
import torch
from transformers import AutoModelForSequenceClassification as modelSC, AutoTokenizer as token
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_path = hf_hub_download(repo_id="MienOlle/sentiment_analysis_api",
                             filename="sentimentAnalysis.pth"
                             )
modelToken = token.from_pretrained("mdhugol/indonesia-bert-sentiment-classification")
model = modelSC.from_pretrained("mdhugol/indonesia-bert-sentiment-classification", num_labels=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

class TextInput(BaseModel):
    text: str

def predict(input):
    inputs = modelToken(input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    ret = logits.argmax().item()

    labels = ["positive", "neutral", "negative"]
    return labels[ret]

@app.post("/predict")
def get_sentiment(data: TextInput):
    sentiment = predict(data.text)
    return {sentiment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)