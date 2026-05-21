from transformers import pipeline
from soynlp.normalizer import repeat_normalize
import re

# 1. 모델 로드 (서버가 켜질 때 한 번만 실행됨)
classifier = pipeline("text-classification", model="./my_review_model8")

# 2. 텍스트 정제 함수
def clean_text(review_text):
    review_text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', str(review_text))
    review_text = repeat_normalize(review_text, num_repeats=2)
    return review_text.strip()

# 3. 예측 함수 (백엔드 API에서 불러다 쓸 함수)
def predict_review(review_text):
    clean_txt = clean_text(review_text)
    
    # 🛑 만약 전처리했더니 아무 글자도 안 남은 빈 문자열이라면? 예외 처리
    if not clean_txt:
        return {
            "sentiment": "분류 불가",
            "score": 0.0
        }
    
    # ─── 🛑 여기서부터 단답형 예외처리 필터 시작 ───
    # 글자 수가 5글자 이하로 극단적으로 짧은 경우 검사
    if len(clean_txt) <= 5:
        # 명확한 부정 단어가 포함되어 있다면 AI한테 묻지도 않고 바로 부정(0) 컷!
        if any(bad_word in clean_txt for bad_word in ['나빠요', '실망', '최악', '별로', '노답', '비추', '안좋아', '별루','붐따']):
            return {
                "sentiment": "부정",
                "score": 1.0  # 사람이 확신하는 100%이므로 1.0 부여
            }
        # 명확한 긍정 단어가 포함되어 있다면 바로 긍정(1) 컷!
        elif any(good_word in clean_txt for good_word in ['조아', '좋아', '최고', '추천', '스껄', '야르', '존맛', '대박','붐업']):
            return {
                "sentiment": "긍정",
                "score": 1.0
            }
    # ─── 🛑 단답형 예외처리 필터 끝 ───

    # 긴 문장이거나 필터에 걸리지 않은 애매한 단어들은 원래대로 똑똑한 AI 모델에게 토스!
    result = classifier(clean_txt)
    
    label = result[0]['label']
    score = result[0]['score']
    
    # 모델 출력 결과에 따른 긍부정 매핑 (LABEL_1이 긍정)
    sentiment = "긍정" if label == "LABEL_1" else "부정"
    
    return {
        "sentiment": sentiment,
        "score": round(score, 4)
    }

# ----- [참고용 테스트 코드] -----
if __name__ == "__main__":
   
    print("테스트 1:", predict_review("이건 정말 야르한 맛이에요 스껄스껄;"))
    
    # 2. 일반적인 부정 문장 (AI 모델이 판별)
    print("테스트 2:", predict_review("음식에서 머리카락 나오고 맛도 별로네요."))
    
    # 3. 🚨 AI가 헷갈려하던 단답형 문장 (하드코딩 필터가 하드캐리하여 '부정'으로 완벽 방어)
    print("테스트 3:", predict_review("나빠요"))
    print("테스트 4:", predict_review("별로임"))
