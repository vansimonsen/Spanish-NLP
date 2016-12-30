from SVM import *
import glob



files_twitter = glob.glob('/home/vansimonsen/projects/Spanish-NLP/Datasets/ONG/Valparaiso_Data_Twitter/*.csv')
files_facebook = glob.glob('/home/vansimonsen/projects/Spanish-NLP/Datasets/ONG/Valparaiso_Json_Facebook/csv/*.csv')


tr = tr_data('TASS/csv/general-tweets-train-tagged.csv')

for f in files_twitter:
	ts = ts_data(f)
	tweets_classification(tr,ts, csv=True, json= False, result_path='Results/ONG/Twitter/'+f.split('/')[-1])
	print "ready", f.split('/')[-1]

for f in files_facebook:
	ts = ts_data(f)
	tweets_classification(tr,ts, csv=True, json= False, result_path='Results/ONG/Facebook/'+f.split('/')[-1])
	print "ready", f.split('/')[-1]