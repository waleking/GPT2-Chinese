'''
turn the following file into a line in train.txt
title:月
date:

炊烟上下
月亮是掘井的白猿
月亮是惨笑的河流上的白猿

多少回天上的伤口淌血
白猿流过钟楼
流过南方老人的头顶

掘井的白猿
村庄喂养的白猿
月亮是惨笑的白猿
月亮自己心碎
月亮早已心碎

==>
月[MASK][MASK]炊烟上下[MASK]月亮是掘井的白猿[MASK]月亮是惨笑的河流上的白猿[MASK][MASK]多少回天上的伤口淌血...
Rule 1: replace \n with [MASK]
Rule 2: replace the whitespace " " with [unused1] 
'''

import os
import sys
import pdb

def format(foldername):
    files = [f for f in os.listdir(foldername) if ".pt" in f]
    # pdb.set_trace()
    with open(foldername+"/"+"train.txt", "w") as fWriter:
        for filename in files:
            result = ""
            with open(foldername+"/"+filename, "r") as fIn:
                for line in fIn:
                    if("title:" in line):
                        result += line.strip("title:").replace("\n", "[MASK]")
                    elif("date:" in line):
                        result += "[MASK]"
                    else:
                        result += line.replace("\n", "[MASK]").replace(" ", "[unused1]")
            fWriter.write("%s\n" % result)

def test():
    format("rawdata")


if __name__=="__main__":
    if(len(sys.argv)!=2):
        print("usage: python format_raw_txt.py target_folder")
        sys.exit(0)
    else:
        target_folder = sys.argv[1]
        format(target_folder)
