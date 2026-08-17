import yt_dlp

downloads = [
    {
        "url": "https://www.youtube.com/watch?v=oic1W5ZriQE",
        "outputname": "full_match"
    },
    {
        "url": "https://www.youtube.com/watch?v=LnKrnoMjqVw",
        "outputname": "highlights"
    }
]

def download_videos(match_links: list):
    for item in match_links:
        options = {
            # en iyi ses akışını seçer
            "format": "bestaudio/best",

            # indirilen dosya adını .. otomatik wav yapar
            "outtmpl": f"{item['outputname']}.%(ext)s",

            # 
            "postprocessors": [{
                "key": "FFmpegExtractAudio", # sesi videodan ayırır
                "preferredcodec": "wav", # sıkıştırılmamış, ham ses. En temizi
            }]
        }

        print(f"Downloading: {item['outputname']}")

        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([item["url"]])


