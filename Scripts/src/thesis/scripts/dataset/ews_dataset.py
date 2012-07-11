'''
Created on Jul 2, 2012

@author: work
'''
import sys
import csv
import time
import datetime

def main(filepath):
    
    logfile = csv.reader(open(filepath, 'rb'), delimiter=';', quotechar='"')
    
    streamout = csv.writer(open('ews_article2.csv', 'wb'), dialect=csv.excel, delimiter=';')
    streamerror = csv.writer(open('missed_lines', 'wb'), delimiter=';')
    
    count = 0
    
    for item in logfile:
        try:
            dt = item[1]
            datet = datetime.datetime.strptime(dt,'%d/%b/%Y:%H:%M:%S')
            new_item = [time.mktime(datet.timetuple()), item[0]]
            streamout.writerow(new_item)
            count = count + 1
        except:
            streamerror.writerow([count, item])
            count = count + 1
     
     
    
    
if __name__ == '__main__':
    main("/media/4AC0AB31C0AB21E5/Documents and Settings/Claudio/Documenti/Thesis/Workloads/MSClaudio/ews/ews2weeks.csv")