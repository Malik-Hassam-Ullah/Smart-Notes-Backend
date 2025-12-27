from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()

# Pipeline banana
summarizer = pipeline("summarization", model="t5-small")

class TextData(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "T5 Summarizer API is Running!"}

@app.post("/summarize")
async def summarize_text(data: TextData):
    try:
        input_text = data.text
        # Summary generate karna
        summary = summarizer(input_text, max_length=150, min_length=40, do_sample=False)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)