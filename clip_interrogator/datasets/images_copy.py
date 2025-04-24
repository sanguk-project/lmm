import shutil
import os
from tqdm import tqdm

def copy_images(source_path, num_copies=3000):
    """
    지정된 경로의 이미지를 num_copies만큼 복사합니다.
    
    Parameters:
        source_path (str): 원본 이미지 파일 경로
        num_copies (int): 생성할 복사본 수 (기본값: 3000)
    """
    # 원본 파일이 존재하는지 확인
    if not os.path.exists(source_path):
        print(f"Error: Source file '{source_path}' does not exist.")
        return
    
    # 원본 파일의 디렉토리와 파일명 추출
    directory, filename = os.path.split(source_path)
    name, ext = os.path.splitext(filename)
    
    # 복사본 생성
    for i in tqdm(range(1, num_copies + 1), desc="Copying images"):
        dest_filename = f"{name}_{i}{ext}"
        dest_path = os.path.join(directory, dest_filename)
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            print(f"Error copying file: {e}")
            continue

if __name__ == "__main__":
    # 사용자가 지정한 원본 이미지 경로
    source_path = "/mnt/ssd/1/sanguk/lmm/clip_interrogator/datasets/nia/test/images/nia_stable_diffusion_test.jpg"
    copy_images(source_path)