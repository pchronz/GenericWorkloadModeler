'''
Created on Mar 20, 2012

@author: work
'''
import csv, numpy
import datetime
import time

def create_dataset():
    logfile = logfile = csv.reader(open('/media/OS/Users/Claudio/Documents/Thesis/Workloads/Calgary HTTP/access_log', 'rb'), delimiter=' ')
    
    count = 0
    
    streamout = csv.writer(open('/media/OS/Users/Claudio/Documents/Thesis/Workloads/Calgary HTTP/log_mod.csv', 'wb'), dialect=csv.excel, delimiter=';')
    streamerror = csv.writer(open('/media/OS/Users/Claudio/Documents/Thesis/Workloads/Calgary HTTP/missed_lines', 'wb'), delimiter=';')
    
    for item in logfile:
        if len(item) == 8:
            dt = item[3][1:len(item[3])-1]
            datet = datetime.datetime.strptime(dt,'%d/%b/%Y:%H:%M:%S')
            new_item = [item[0], time.mktime(datet.timetuple()), item[6], item[7]] + item[5].split()
            streamout.writerow(new_item)
            count = count + 1
        else:
            streamerror.writerow([count, len(item)])
            count = count + 1
    
if __name__ == '__main__':
    create_dataset()