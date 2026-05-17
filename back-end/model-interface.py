from transformers import pipeline
from soynlp.normalizer import repeat_normalize
import re

# 1. 모델 로드 (서버가 켜질 때 한 번만 실행됨)
# 주의: ./my_review_model 폴더가 서버 코드와 같은 위치에 있어야 함.
classifier = pipeline("text-classification", model="./my_review_model")

# 2. 텍스트 정제 함수
def clean_text(review_text):
    review_text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', str(review_text))
    review_text = repeat_normalize(review_text, num_repeats=2)
    return review_text.strip()

# 3. 예측 함수 (백엔드 API에서 불러다 쓸 함수)
def predict_review(review_text):
    clean_txt = clean_text(review_text)
    result = classifier(clean_txt)
    
    label = result[0]['label']
    score = result[0]['score']
    
    # 모델 출력 결과에 따른 긍부정 매핑 (LABEL_1이 긍정인지 꼭 학습 때 확인해보기!)
    sentiment = "긍정" if label == "LABEL_1" else "부정"
    
    # 백엔드가 프론트엔드에 JSON으로 넘겨줄 수 있도록 딕셔너리로 반환.
    return {
        "sentiment": sentiment,
        "score": round(score, 4)
    }

# ----- [참고용 테스트 코드] -----
if __name__ == "__main__":
    print(predict_review("이건 정말 야르한 맛이에요 스껄스껄;"))
    print(predict_review("음식에서 머리카락 나오고 맛도 별로네요."))