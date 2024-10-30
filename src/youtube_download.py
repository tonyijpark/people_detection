import yt_dlp

# 유튜브 동영상 URL
url = "https://youtu.be/8UzSJBUt1dE"

# 다운로드 옵션 설정
ydl_opts = {
    'format': 'best',  # 가장 좋은 화질로 다운로드
    'outtmpl': '%(title)s.%(ext)s'  # 파일명: 동영상 제목
}

# yt-dlp를 사용한 다운로드
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    print(f"Downloading video from {url}...")
    ydl.download([url])
    print("Download completed!")