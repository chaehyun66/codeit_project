from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ReviewInput(BaseModel):
    review_text: str

@app.post("/api/v1/reviews/analyze")
def analyze_review(data: ReviewInput):

    user_review = data.review_text
    
    #임시코드 (테스트용 가짜 답장)
    dummy_result = "긍정"
    dummy_score = 0.95

    return {
        "succeed": True,
        "result": dummy_result,
        "score": dummy_score
        }