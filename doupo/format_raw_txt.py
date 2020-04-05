'''
This script is used to remove empty lines in the input file 
'''
import sys

def format(in_filename, out_filename):
    with open(out_filename, "w") as fOut:
        with open(in_filename, "r") as fIn:
            for line in fIn:
                if(line!=""):
                    fOut.write(line.lstrip())


def test():
    in_filename="./rawdata/train_raw.txt"
    out_filename="./rawdata/train.txt"
    format(in_filename, out_filename)


if __name__=="__main__":
    if(len(sys.argv)<3):
        print("usage: python format_raw_txt.py in_filename out_filename")
    else:
        in_filename = sys.argv[1]
        out_filename = sys.argv[2]
        format(in_filename, out_filename)
