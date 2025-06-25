# LMM (Large Multimodal Model) Repository

이 저장소는 Large Multimodal Model을 위한 다양한 도구들을 포함하고 있습니다.

## 📁 프로젝트 구조

- `clip_interrogator/` - CLIP 기반 이미지 캡셔닝
- `stable_difussion/` - Stable Diffusion 관련 코드
- `Grounding-Dino-FineTuning/` - Grounding DINO 파인튜닝

## 🛠️ 사용법

### CLIP Interrogator

```bash
cd clip_interrogator
python image_captioning.py
```

### Stable Diffusion

```bash
cd stable_difussion
python stable_diffusion.py
```

### Grounding DINO Demo

```bash
cd Grounding-Dino-FineTuning/demo
python gradio_app.py
```

## ⚠️ 주의사항

1. 모든 데이터셋은 `./datasets/` 폴더에 위치해야 합니다
2. GPU 메모리가 부족한 경우 CPU 모드로 자동 전환됩니다
3. 외부 접근이 필요한 경우에만 `GRADIO_SERVER_NAME`을 변경하세요