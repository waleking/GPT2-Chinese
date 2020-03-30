import re
#p_url = re.compile(r'<url>(.*)<\/url>')

p = re.compile(r'^<doc>$|^<\/doc>$|^<url>.*<\/url>$|^<docno>.*<\/docno>$|^<contenttitle>.*<\/contenttitle>$|^<content>.*<\/content>$')
p_start = re.compile(r'^<doc>$')
p_end = re.compile(r'^<\/doc>$')
p_title = re.compile(r'^<contenttitle>(.*)<\/contenttitle>$')
p_content = re.compile(r'^<content>(.*)<\/content>$')

i = 0
title_content = ""

with open("sogou_utf8_only_content.txt", "w") as writer_onlycontent:
    with open("sogou_utf8_title_content.txt", "w") as writer:
        with open("sogou_utf8.txt", "r") as f:
            for line in f:
                i += 1
                line = line.strip()
                if(p.match(line) is None):
                    print(line)
                else:
                    matchobj_title = p_title.match(line)
                    if(matchobj_title):
                        title_content = matchobj_title.group(1)
                    else:
                        matchobj_content = p_content.match(line)
                        if(matchobj_content and title_content==""):
                            writer_onlycontent.write("%s\n" % matchobj_content.group(1))
                        if(matchobj_content and title_content!="" and matchobj_content.group(1)!=""):
                            title_content = title_content + "\t" + matchobj_content.group(1)
                            writer.write("%s\n" % title_content)
                        else:
                            if(p_end.match(line)):
                                title_content=""
                if(i%1000000==0):
                    print(i)
