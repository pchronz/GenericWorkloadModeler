'''
Created on Sep 8, 2011

@author: work
'''

from datetime import tzinfo, timedelta, datetime

ZERO = timedelta(0)
HOUR = timedelta(hours=2)

# A UTC class.

class germany_utc(tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return HOUR

    def tzname(self, dt):
        return "UTC +2"

    def dst(self, dt):
        return ZERO