'''
Created on Apr 24, 2012

@author: work
'''
import sys
import csv
import time
import datetime

def main(filepath, year):
    
    logfile = csv.reader(open(filepath, 'rb'), delimiter=' ')
    
    streamout = csv.writer(open('log_mod.csv', 'wb'), dialect=csv.excel, delimiter=';')
    streamerror = csv.writer(open('missed_lines', 'wb'), delimiter=';')
    
    count = 0
    
    for item in logfile:
        try:
            dt = item[0]+ "," + item[1] + year + "," + item[2]
            datet = datetime.datetime.strptime(dt,'%d,%b,%Y,%H:%M:%S')
            new_item = [time.mktime(datet.timetuple()), item[3][1:len(item[3])-1], item[5], item[7]]
            streamout.writerow(new_item)
            count = count + 1
        except:
            streamerror.writerow([count, item])
            count = count + 1
     
     
    
    
if __name__ == '__main__':
    main(sys.argv[1:])