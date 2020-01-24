from read_frames import readframes
import schedule
import time

def playwakeupsample():
    directory = "/home/pi/Desktop/startup_script/data_collection/commands"
    directory += "/wakeupsignal_11000_12000_comb_tooth.wav"
    readframes(directory)

def playpiecewisesample():
    directory = "/home/pi/Desktop/startup_script/data_collection/commands"
    directory += "/piecewise_10600_to_12600_comb_tooth.wav"
    readframes(directory)

hour = 21
minute = 30
second = 0

for i in range(0, 1001, 20):
    hour = 21 + i/3600
    minute = 30 + i/60
    second = i%60
    time = "%d:%d:%d" % (hour, minute, second)
    schedule.every().day.at(time).do(playwakeupsample)

hour = 21
minute = 30
second = 10

for i in range(0, 1001, 20):
    hour = 21 + i/3600
    minute = 30 + i/60
    second = i%60
    time = "%d:%d:%d" % (hour, minute, second)
    schedule.every().day.at(time).do(playpiecewisesample)

while True:
    schedule.run_pending()
    time.sleep(1)