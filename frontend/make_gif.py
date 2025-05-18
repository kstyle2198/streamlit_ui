# from PIL import Image
# import os

# def convert_jpgs_to_gif(input_folder, output_gif, duration=200, loop=0):
#     """
#     여러 JPG 이미지를 하나의 GIF로 변환하는 함수
    
#     :param input_folder: JPG 이미지들이 있는 폴더 경로
#     :param output_gif: 출력 GIF 파일 경로
#     :param duration: 각 프레임 간 지연 시간(밀리초)
#     :param loop: GIF 반복 횟수 (0은 무한 반복)
#     """
#     # 폴더에서 JPG 파일 목록 가져오기
#     jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    
#     # 파일명으로 정렬 (옵션)
#     jpg_files.sort()
    
#     # 이미지 열기
#     images = []
#     for jpg_file in jpg_files:
#         file_path = os.path.join(input_folder, jpg_file)
#         try:
#             img = Image.open(file_path)
#             images.append(img)
#         except Exception as e:
#             print(f"Error opening {file_path}: {e}")
    
#     if not images:
#         print("No valid JPG images found!")
#         return
    
#     # 첫 번째 이미지로 GIF 생성 (나머지는 append)
#     images[0].save(
#         output_gif,
#         save_all=True,
#         append_images=images[1:],
#         duration=duration,
#         loop=loop,
#         optimize=True
#     )
    
#     print(f"Successfully created GIF: {output_gif}")

# # 사용 예제
# if __name__ == "__main__":
#     input_folder = "system_image"  # JPG 파일들이 있는 폴더
#     output_gif = "output.gif"    # 출력 GIF 파일 경로
    
#     convert_jpgs_to_gif(input_folder, output_gif, duration=1500, loop=0)

import glob
from PIL import Image, ImageChops

def add_crossfade(images, fade_frames=5):
    faded_images = []
    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]
        
        # 현재 이미지 추가
        faded_images.append(img1)
        
        # Crossfade 프레임 생성
        for alpha in [x / (fade_frames + 1) for x in range(1, fade_frames + 1)]:
            blended = Image.blend(img1, img2, alpha)
            faded_images.append(blended)
    
    faded_images.append(images[-1])  # 마지막 이미지 추가
    return faded_images

# 사용 예시
images = [Image.open(f) for f in sorted(glob.glob("system_image/*.jpg"))]
faded_images = add_crossfade(images, fade_frames=3)
faded_images[0].save("smooth.gif", save_all=True, append_images=faded_images[1:], duration=100, loop=0)