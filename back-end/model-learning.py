import pandas as pd
import torch
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from soynlp.normalizer import repeat_normalize

# --- [1] 데이터 로드 및 간단 정제 ---
# 본인의 파일명에 맞게 수정하세요 (예: my_data.csv)
df = pd.read_csv('cleaned_review_data.csv') 

def clean_text(text):
    text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', str(text))
    text = repeat_normalize(text, num_repeats=2)
    return text.strip()

df['cleaned_review'] = df['review'].apply(clean_text)

# --- [2] 토크나이저 및 모델 설정 ---
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- [3] 데이터셋 클래스 정의 ---
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer):
        self.encodings = tokenizer(reviews, padding=True, truncation=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 데이터셋 생성 (100개 데이터 활용)
train_dataset = ReviewDataset(df['cleaned_review'].tolist(), df['label'].tolist(), tokenizer)

# --- [4] 학습 설정 (CPU 최적화) ---
training_args = TrainingArguments(
    output_dir='./results',          # 결과물 저장 폴더
    num_train_epochs=3,              # 3회 반복 학습
    per_device_train_batch_size=8,   # 한 번에 공부할 양
    logging_steps=10,                # 10단계마다 로그 출력
    save_steps=100,                  # 체크포인트 저장 주기
    use_cpu=True,                    # CPU 사용 강제
    dataloader_num_workers=0         # 윈도우 에러 방지
)

# --- [5] 학습 시작 (Trainer) ---
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

print("🚀 모델 학습을 시작합니다!")
trainer.train()

# --- [6] 모델 저장 ---
model.save_pretrained("./my_review_model")
tokenizer.save_pretrained("./my_review_model")
print("✅ 모델이 './my_review_model' 폴더에 저장되었습니다.")