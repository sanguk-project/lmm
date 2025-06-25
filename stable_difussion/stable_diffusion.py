import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import tqdm as tqdm_module
import warnings
import time
import numpy as np
from tabulate import tabulate
import os

# 경고 메시지 억제
warnings.filterwarnings("ignore")

# 터미널 출력 유틸리티 함수
def clear_line():
    """현재 라인을 지우는 함수"""
    print("\033[K", end="\r")

def print_table(data, headers, title=None):
    """표 형식으로 데이터 문자열 생성"""
    output_str = ""
    if title:
        output_str += f"{title}\n"
    table_str = tabulate(data, headers=headers, tablefmt="grid", floatfmt=".6f")
    output_str += table_str
    return output_str

# 설정
model_id = "stabilityai/stable-diffusion-2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 장치: {device}")

# 모델 구성 요소 로드
print("🔄 모델 구성 요소 로드 중...")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
print("✅ 모델 로드 완료")

# 환경변수에서 데이터셋 경로 가져오기
dataset_file_path = os.getenv('DATASET_FILE_PATH', "./datasets/captions/test_captions_ko.jsonl")

if not os.path.exists(dataset_file_path):
    raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_file_path}")

dataset = load_dataset("json", data_files=dataset_file_path, split="train")

def load_image(example):
    example["image"] = Image.open(example["file_path"])
    example["text"] = example["caption"]
    return example

dataset = dataset.map(load_image)

# 전처리
preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    texts = examples["text"]
    return {"images": images, "texts": texts}

dataset.set_transform(transform)
print(f"✅ 데이터셋 준비 완료 ({len(dataset)} 이미지)")

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 옵티마이저 설정
optimizer = AdamW(unet.parameters(), lr=1e-5)
print("🚀 학습 시작...")

# 학습 루프
num_epochs = 1  # 필요에 따라 조정
total_steps = len(train_dataloader) * num_epochs
losses = []

# 손실 표시를 위한 테이블 구성
loss_headers = ["Step", "현재 손실", "이동 평균 (10)", "평균 손실"]
loss_history = []
loss_display_interval = 10  # 손실 정보 업데이트 주기

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_losses = []
    previous_total_lines = 0  # 이전 테이블의 라인 수 초기화
    
    # 고정된 진행바 생성
    progress_bar = tqdm(
        total=len(train_dataloader),
        desc=f"Epoch {epoch+1}/{num_epochs}",
        bar_format='{l_bar}{bar:30}{r_bar}',
        ncols=80,
        position=0,
        leave=True
    )
    
    # 테이블 초기화
    loss_table_rows = []
    
    for step, batch in enumerate(train_dataloader):
        images = batch["images"].to(device)
        texts = batch["texts"]
        
        # 텍스트 토큰화
        inputs = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        
        # 텍스트 임베딩 생성
        with torch.no_grad():
            text_embeddings = text_encoder(input_ids)[0]
        
        # 이미지를 잠재 공간으로 인코딩
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # 스케일링 계수
        
        # 노이즈 샘플링
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],)).long().to(device)
        
        # 노이즈 추가
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # UNet으로 노이즈 예측
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # 손실 계산
        loss = F.mse_loss(noise_pred, noise)
        
        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 손실 기록
        loss_value = loss.item()
        epoch_losses.append(loss_value)
        losses.append(loss_value)
        
        # 진행 상황 업데이트
        progress_bar.update(1)
        progress_bar.set_postfix({
            "손실": f"{loss_value:.4f}"
        })
        
        # 손실 테이블에 정보 추가
        moving_avg_10 = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)
        avg_loss = np.mean(epoch_losses)
        
        # 단계마다 데이터 수집
        loss_table_rows.append([step + 1, loss_value, moving_avg_10, avg_loss])
        
        # 지정된 간격마다 테이블 출력
        if (step + 1) % loss_display_interval == 0 or step == len(train_dataloader) - 1:
            # 마지막 10개 행만 표시
            display_rows = loss_table_rows[-10:]
            
            # 테이블 문자열 생성
            output_str = print_table(
                display_rows, 
                loss_headers, 
                title=f"📉 손실 현황 (Epoch {epoch+1}, 단계 {step+1}/{len(train_dataloader)})"
            )
            lines = output_str.split('\n')
            total_lines = len(lines)
            
            # 이전 테이블이 있으면 커서를 위로 이동
            if previous_total_lines > 0:
                move_up_str = "\033[F" * previous_total_lines
                tqdm_module.write(move_up_str, end='')
            
            # 새 테이블 출력
            tqdm_module.write(output_str)
            previous_total_lines = total_lines
    
    progress_bar.close()
    
    # 에폭 요약
    epoch_time = time.time() - epoch_start_time
    avg_loss = np.mean(epoch_losses)
    
    # 에폭 요약 테이블
    epoch_summary = [
        ["시간 (초)", epoch_time],
        ["평균 손실", avg_loss],
        ["최소 손실", min(epoch_losses)],
        ["최대 손실", max(epoch_losses)]
    ]
    
    print(print_table(
        epoch_summary, 
        headers=["지표", "값"], 
        title=f"📊 Epoch {epoch+1} 요약"
    ))

# 최종 요약 테이블
final_summary = [
    ["전체 단계", total_steps],
    ["최종 평균 손실", np.mean(losses)],
    ["최소 손실", min(losses)],
    ["최대 손실", max(losses)]
]

print(print_table(
    final_summary,
    headers=["지표", "값"],
    title="🏁 학습 완료!"
))

# 모델 저장
print("💾 모델 저장 중...")
pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,  # 안전성 검사 비활성화
    feature_extractor=None  # 특성 추출기 비활성화
)
pipe.save_pretrained("fine-tuned-model/nia/test1")
print("✅ 모델 저장 완료: 'fine-tuned-model'")