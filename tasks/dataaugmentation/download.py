'''
download haizi's poems
'''
import os
import subprocess
import sys

def download(target_folder):
    url = "https://raw.githubusercontent.com/sheepzh/poetry/master/data/origin/海子_haizi/"
    with open("%s/poem_list.txt" % target_folder, "r") as f:
        for line in f:
            try:
                title = line.strip()
                # turn off wget's output log, including error information
                subprocess.Popen("wget -c -q -O %s %s%s.pt" % (target_folder+"/"+title+".pt", url, title), shell=True) 
            except KeyboardInterrupt:
                break


def test():
    download("rawdata/")


if __name__=="__main__":
    # test()
    if(len(sys.argv)!=2):
        print("usage: download target_folder")
        sys.exit(0)
    else:
        target_folder = sys.argv[1]
        download(target_folder)
