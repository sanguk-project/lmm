import json
from tqdm import tqdm

def replace_captions(input_jsonl_path, output_jsonl_path, fixed_caption):
    """
    JSONL 파일의 모든 캡션을 고정된 한국어 캡션으로 대체하여 새로운 파일에 저장합니다.
    
    Parameters:
        input_jsonl_path (str): 원본 JSONL 파일 경로
        output_jsonl_path (str): 새로운 JSONL 파일 경로
        fixed_caption (str): 대체할 고정된 한국어 캡션
    """
    # 기존 파일의 라인 수를 세어 진행 바에 사용
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # 새로운 JSONL 파일을 작성
    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile, total=total_lines, desc="캡션 대체 중"):
                try:
                    data = json.loads(line)
                    data['caption'] = fixed_caption
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {e}")
                    continue

if __name__ == "__main__":
    # 경로 및 고정된 캡션 설정
    input_jsonl_path = "/mnt/ssd/1/sanguk/lmm/clip_interrogator/datasets/captions/images_captions.jsonl"
    output_jsonl_path = "/mnt/ssd/1/sanguk/lmm/clip_interrogator/datasets/captions/images_captions_ko.jsonl"
    fixed_caption = (
        "가운데 나무가 있는 풍경화, 한국 아르누보 애니메이션 스타일, 티베트 문자 스크립트, "
        "밀밭, Kanō Hōgai에서 영감을 받은, 게임 내 스크린샷, Mark Schultz, "
        "21:9 비율, 사파리 배경, 드문드문한 식물, 16:9 비율, 로딩 화면."
    )
    
    # 캡션 대체 실행
    replace_captions(input_jsonl_path, output_jsonl_path, fixed_caption)