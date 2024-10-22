import os
import requests

class YouTubeVideo:
    def __init__(self, video_url):
        self.video_url = video_url
        self.video_id = self._extract_video_id(video_url)
        self.api_key = "AIzaSyBV9j-2DheWwvUkvBgu80SwizV6UEjr274"

    def _extract_video_id(self, video_url):
        video_id = None
        if 'youtube.com' in video_url:
            video_id = video_url.split('v=')[1]
            if '&' in video_id:
                video_id = video_id.split('&')[0]
        elif 'youtu.be' in video_url:
            video_id = video_url.split('/').pop().split("?")[0]
        else:
            raise ValueError('Invalid YouTube URL')
        
        print(video_id)
        return video_id

    def get_info(self):
        api_url = f"https://www.googleapis.com/youtube/v3/videos?id={self.video_id}&key={self.api_key}&part=snippet"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            
            if data.get('items') and len(data['items']) > 0:
                video_info = data['items'][0]['snippet']
                title = video_info['title']
                author = video_info['channelTitle']
                thumbnail = video_info['thumbnails']['default']['url']
                return {
                    'title': title,
                    'author': author,
                    'thumbnail': thumbnail
                }
            else:
                raise ValueError('Video not found or API request failed')

        except Exception as error:
            print(error)
            raise Exception('Failed to fetch video info') from error

# Example usage
# api_key = "your_api_key_here"
# link = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# youtube_video = YouTubeVideo(link)
# try:
#     video_info = youtube_video.get_info()
#     print('Title:', video_info['title'])
#     print('Author:', video_info['author'])
#     print('Thumbnail URL:', video_info['thumbnail'])
# except Exception as e:
#     print(e)

#https://www.youtube.com/watch?v=dQw4w9WgXcQ