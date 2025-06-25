import json
import time
import os
import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
import pathlib
from tqdm import tqdm

# 환경변수에서 경로를 가져오거나 기본값 사용
dataset_path = pathlib.Path(os.getenv('DATASET_PATH', './datasets/nia/test/images'))
captions_path = pathlib.Path(os.getenv('CAPTIONS_PATH', './datasets/captions'))
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

# 디렉토리가 존재하지 않으면 생성
dataset_path.mkdir(parents=True, exist_ok=True)
captions_path.mkdir(parents=True, exist_ok=True)

image_files = [f for f in dataset_path.glob("*") if f.suffix.lower() in image_extensions]

# GPU 메모리 상태 확인
if torch.cuda.is_available():
    print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU 메모리 총량: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# CLIP Interrogator 설정
config = Config(
    clip_model_name="ViT-L-14-quickgelu/openai",
    device="cuda" if torch.cuda.is_available() else "cpu",
    clip_model_path=None,
)

# Interrogator 초기화
ci = Interrogator(config)

# 캡션 저장 파일 정의
captions_file = captions_path / f'{dataset_path.name}_captions.jsonl'

# JSONL 파일 초기화 (기존 파일 삭제)
if captions_file.exists():
    captions_file.unlink()

print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

# tqdm으로 진행 바 설정
with tqdm(total=len(image_files), desc="이미지 처리", unit="image") as pbar:
    for image_file in image_files:
        try:
            start_time = time.time()
            
            # 이미지 로드 및 리사이즈
            image = Image.open(image_file).convert('RGB')
            max_size = 512
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            # 영어 캡션 생성
            caption = ci.interrogate(image)
            print(f"생성된 캡션: {caption}")
            
            # 캡션이 비어 있는지 확인
            if caption:
                item = {
                    'file_path': str(image_file),
                    'file_name': image_file.name,
                    'caption': caption  # 영어 캡션 저장
                }
                with open(captions_file, 'a', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            else:
                print(f"{image_file.name}: 캡션 생성 실패")
            
            # 처리 시간 출력
            elapsed_time = time.time() - start_time
            pbar.set_postfix(file=image_file.name, time=f"{elapsed_time:.2f}s")
            pbar.update(1)
        except Exception as e:
            print(f"{image_file.name} 처리 오류: {str(e)}")
            pbar.update(1)

# 최종 결과 확인
if captions_file.exists() and captions_file.stat().st_size > 0:
    print(f"캡션 JSONL 저장 완료: {captions_file}")
    with open(captions_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(line.strip())
            else:
                break
else:
    print("유효한 이미지-캡션 쌍이 없습니다.")