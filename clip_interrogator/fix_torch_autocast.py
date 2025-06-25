#!/usr/bin/env python3
import os
import re
import site
import sys

def find_clip_interrogator_path():
    """clip_interrogator 패키지 경로를 자동으로 찾습니다."""
    try:
        import clip_interrogator
        return os.path.join(os.path.dirname(clip_interrogator.__file__), 'clip_interrogator.py')
    except ImportError:
        # 환경변수에서 가져오기
        return os.getenv('CLIP_INTERROGATOR_PATH', '')

clip_interrogator_path = find_clip_interrogator_path()

if not clip_interrogator_path or not os.path.exists(clip_interrogator_path):
    print("경고: clip_interrogator.py 파일을 찾을 수 없습니다.")
    print("CLIP_INTERROGATOR_PATH 환경변수를 설정하거나 패키지를 올바르게 설치하세요.")
    sys.exit(1)

def fix_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # torch.cuda.amp.autocast()를 torch.amp.autocast('cuda')로 교체
    pattern = r'torch\.cuda\.amp\.autocast\(\)'
    replacement = r"torch.amp.autocast('cuda')"
    new_content = re.sub(pattern, replacement, content)
    
    # 변경 사항이 있는 경우에만 파일 업데이트
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"파일 수정 완료: {file_path}")
        return True
    else:
        print(f"수정 사항 없음: {file_path}")
        return False

if __name__ == "__main__":
    # 백업 파일 생성
    backup_path = clip_interrogator_path + ".bak"
    if not os.path.exists(backup_path):
        os.system(f"cp {clip_interrogator_path} {backup_path}")
        print(f"백업 파일 생성: {backup_path}")
    
    # 파일 수정
    fixed = fix_file(clip_interrogator_path)
    
    if fixed:
        print("autocast 경고 수정 완료! 이제 프로그램을 다시 실행하세요.")
    else:
        print("수정할 내용이 없습니다. 다른 문제가 있을 수 있습니다.") 