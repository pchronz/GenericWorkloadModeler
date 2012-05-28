'''
Created on Sep 5, 2011

@author: work
'''
import csv
import string
import datetime
import time
from thesis.scripts.dataset.german_utc import germany_utc
__microsecond = 200000
__months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dic':12}

#filesource = "/media/4AC0AB31C0AB21E5/Documents and Settings/Claudio/Documenti/Thesis/Workloads/MSClaudio/ews/modified/access_log-20110729.csv"
def toepoch(date):
#    file = csv.reader(open(filesource, "rb"), delimiter= ';', quotechar = '"')
    
#    for row in file:
#        data = row[1]
    words = string.split(date, ":")
    date = string.split(words[0], "/")
    gutc = germany_utc()
    epochtime = None
    try:
        dtime = datetime.datetime(int(date[2]), __months[date[1]], int(date[0]), int(words[1]), int(words[2]), int(words[3]), __microsecond, tzinfo = gutc)
        epochtime = time.mktime(dtime.timetuple())
    except:
        print date
#    print dtime
#    print epochtime
    return epochtime
        
    
