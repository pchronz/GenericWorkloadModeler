'''
Created on Jul 11, 2011

@author: work
'''
import sys
from dataset import weeklydataset

def main(argv):
    [train, test] = weeklydataset(argv[0], [])
    print len(train)
    print len(test)
        
if __name__ == '__main__':
    main(sys.argv[1:])