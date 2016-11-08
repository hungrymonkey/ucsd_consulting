#!/usr/bin/env python

import re
import csv
import sys,os,stat

outputCSV = 'AGPS_PERCENT_DATA_LIST.csv'
dataMICFiles = '^agps.+data_percent\.csv'

if len(sys.argv) == 1:
    files=os.listdir(".")
    #ignore files starting with '.' using list comprehension
    files=[filename for filename in files if filename[0] != '.']
else:
    files=sys.argv[1:]
if files is None:
   sys.exit(1)
   
dataReg = re.compile(dataMICFiles)

def filterPick(files, regex):
    return [x for x in files if regex.match(x.lower())]
mic = filterPick(files,dataReg)
mic.sort()

out = open(outputCSV,'w')

for fi in mic:
   reader = csv.reader( open(fi,'r') )
   out.write(("%s\n")%(fi))
   for row in reader:
      out.write(("%s\n")%(','.join(row)))
   out.write("\n")
out.close()
    
