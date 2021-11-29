import warnings

from AudioModeller import train_model, test_model
from AudioRecorder import record_audio_train, record_audio_test

warnings.filterwarnings("ignore")

while True:
	choice=int(input("\n 1.Record audio for training \n 2.Train Model \n 3.Record audio for testing \n 4.Test Model\n"))
	if(choice==1):
		record_audio_train()
	elif(choice==2):
		train_model()
	elif(choice==3):
		record_audio_test()
	elif(choice==4):
		test_model()
	if(choice>4):
		exit()
