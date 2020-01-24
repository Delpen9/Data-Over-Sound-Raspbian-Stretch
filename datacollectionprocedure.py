from read_frames import readframes
import schedule
import time

## $crontab -r to stop this in worst case scenario

def playwakeupsample():
    directory = "/home/pi/Desktop/startup_script/data_collection/commands"
    directory += "/wakeupsignal_11000_12000_comb_tooth.wav"
    readframes(directory)

def playpiecewisesample():
    directory = "/home/pi/Desktop/startup_script/data_collection/commands"
    directory += "/piecewise_10600_to_12600_comb_tooth.wav"
    readframes(directory)

try:
    hour_start = 20
    minute_start = 32
    second_start = 0

    for i in range(0, 1001, 20):
        hour = "{:0>2d}".format(hour_start + i//3600)
        minute = "{:0>2d}".format(minute_start + i//60)
        second = "{:0>2d}".format((second_start + i)%60)
        datetime = "%s:%s:%s" % (hour, minute, second)
        print(datetime)
        schedule.every().day.at(datetime).do(playwakeupsample)

    hour_start = 20
    minute_start = 32
    second_start = 10

    for i in range(0, 1001, 20):
        hour = "{:0>2d}".format(hour_start + i//3600)
        minute = "{:0>2d}".format(minute_start + i//60)
        second = "{:0>2d}".format((second_start + i)%60)
        datetime = "%s:%s:%s" % (hour, minute, second)
        print(datetime)
        schedule.every().day.at(datetime).do(playpiecewisesample).tag('data-collection')

    while True:
        schedule.run_pending()
        time.sleep(1)

except KeyboardInterrupt:
    # Cleanup/exiting code
    schedule.clear('data-collection')