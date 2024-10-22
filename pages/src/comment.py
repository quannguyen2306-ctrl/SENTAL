import os
import requests

class CommentInteractor:
    def __init__(self, video_url):
        self.video_id = self._extract_video_id(video_url)
        self.api_key = 'AIzaSyBV9j-2DheWwvUkvBgu80SwizV6UEjr274'
        print(self.api_key)
        self.url = 'https://www.googleapis.com/youtube/v3/commentThreads'
        self.all_comments = []

    def _extract_video_id(self, video_url):
        if 'youtube.com' in video_url:
            video_id = video_url.split('v=')[1]
            if '&' in video_id:
                video_id = video_id.split('&')[0]
        elif 'youtu.be' in video_url:
            video_id = video_url.split('/').pop().split('?')[0]
        else:
            raise ValueError('Invalid YouTube URL')
        
        print(video_id)
        return video_id

    def get_comments(self, maxResults, page_token=None, ):
        self.all_comments = []
        params = {
            'part': 'snippet',
            'videoId': self.video_id,
            'key': self.api_key,
            'maxResults': maxResults
        }
        if page_token:
            params['pageToken'] = page_token
        
        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            data = response.json()
            
            while 'nextPageToken' in data :
                # Process comments here...
                comments = data.get('items', [])
                for comment in comments:
                    comment_text_original = comment['snippet']['topLevelComment']['snippet']['textOriginal']
                    result = {'text': comment_text_original}
                    self.all_comments.append(result)  # Push comment to the array
                
                yield self.all_comments
                params = {
                        'part': 'snippet',
                        'videoId': self.video_id,
                        'key': self.api_key,
                        'pageToken': data['nextPageToken'], 
                        'maxResults': maxResults
                    }
                response = requests.get(self.url, params=params)
                response.raise_for_status()
                data = response.json()
                
            # Recursively fetch comments if there's a nextPageToken
            # if 'nextPageToken' in data:
            #     print("Testing")
            #     self.get_comments(data['nextPageToken'])
        except Exception as e:
            print("Error in fetching comments", e)
        
  # Return the array after recursion is complete

# # Example usage:
# link = "https://www.youtube.com/watch?v=d-Eq6x1yssU"
# new_comment = CommentInteractor(link)
# comments = new_comment.get_comments()
# for i in comments: 
#     print(i)
