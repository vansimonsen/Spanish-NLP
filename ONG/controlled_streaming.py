

#import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import signal
import time

def get_tweets_stream(filename, f):
    #Variables that contains the user credentials to access Twitter API 
    access_token = "268580142-cYIw9WbhU3OrkmX39P919C0c7ja10kop1S8z7VkD"
    access_token_secret = "16FAKFThtGgJe7vrivFPYHjJVo27AyZgTRCpbcjUiZZ1L"
    consumer_key = "2V4HCvd2M82NzsmAhaP2DNp1m"
    consumer_secret = "pcoPfUXQ8Hl4SSBzvKjy4ZIMRt25CFE0eK7l8NZXWtx8Pv9qGG"
    
	#This handles Twitter authetification and the connection to Twitter Streaming API
	l = StdOutListener(filename)
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	stream = Stream(auth, l)

	#This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
	stream.filter(follow= f)

class StdOutListener(StreamListener):
	def __init__(self,filename):
		self.tweets = open(filename, 'a')

	def on_data(self, data):
		self.tweets.write(data)
		return True

	def on_error(self, status):
		print status

class TimeoutException(Exception):   # Custom exception class
	pass

def timeout_handler(signum, frame):   # Custom signal handler
	raise TimeoutException

# Change the behavior of SIGALRM

def calculate_t():
	h = time.strftime("%H:%M:%S")
	hour = int(h[:2])
	t = ((24-hour)*3600)+1
	return t

def start_and_check(t, filename, accounts):
	date = (time.strftime("%d/%m/%Y"))[:2]
	file = 'Tweets/'+filename+date+'.json'
	signal.signal(signal.SIGALRM, timeout_handler)

	# Start the timer. Once 5 seconds are over, a SIGALRM signal is sent.
	signal.alarm(t)    
	# This try/except loop ensures that 
	#   you'll catch TimeoutException when it's sent.
	try:
		get_tweets_stream(file, accounts)
	except TimeoutException:
		print t
		new_date = (time.strftime("%d/%m/%Y"))[:2]
		if date != new_date:
			date = new_date
			t = calculate_t()
			print t
			#signal.alarm(0)
			start_and_check(t, filename, accounts)
			#foo(filename, var=0)
	
			# continue the for loop if function A takes more than 5 second
		#else:
		# Reset the alarm




quinta = np.load("ValpoList.npy")
metro = np.load("MetroList.npy")
biobio = np.load("BiobioList.npy")

accounts = np.concatenate((quinta,metro,biobio))


t = calculate_t()
print t
start_and_check(t, 'ONGTweets', accounts)

