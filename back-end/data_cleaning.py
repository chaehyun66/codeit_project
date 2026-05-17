import pandas as pd
from transformers import AutoTokenizer
import torch

# 1. 1단계에서 저장한 파일 불러오기 (파일명이 맞는지 꼭 확인!)
try:
    df = pd.read_csv('cleaned_review_data.csv')
    print("파일 로드 성공!")
except FileNotFoundError:
    print("에러: 'cleaned_review_data.csv' 파일이 없습니다. 1단계를 먼저 실행해주세요.")

# 2. 토크나이저 준비
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

# 3. 데이터 확인 및 토큰화
if 'cleaned_review' in df.columns:
    sample_review = df['cleaned_review'].iloc[0]
    tokenized_output = tokenizer(sample_review, padding='max_length', max_length=64, truncation=True)

    print(f"원본 리뷰: {sample_review}")
    print(f"숫자로 변환된 결과 (input_ids): \n{tokenized_output['input_ids']}")
else:
    print(f"에러: CSV에 'cleaned_review' 컬럼이 없습니다. 현재 컬럼명: {df.columns}")


# 전체 텍스트를 한꺼번에 토큰화 (텐서 형태)
all_inputs = tokenizer(
    df['cleaned_review'].tolist(),
    padding='max_length',
    max_length=64,
    truncation=True,
    return_tensors="pt"
)

# 파일로 저장 (input_ids만 저장하거나 딕셔너리 통째로 저장)
torch.save(all_inputs, 'tokenized_inputs.pt')
print("학습용 텐서 파일 'tokenized_inputs.pt' 저장 완료!")