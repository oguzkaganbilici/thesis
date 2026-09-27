import yt_dlp

url = "https://www.youtube.com/watch?v=71iH9yLmvhw"

ydl_opts = {
    # vcodec^=avc filtresi ile sadece H.264 video kodlamasını zorlar
    'format': 'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc]/best',
    
    'outtmpl': '%(title)s.%(ext)s',
}

print("YouTube videosu H.264 (AVC) codec ile indiriliyor...")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("\nİndirme işlemi başarıyla tamamlandı!")