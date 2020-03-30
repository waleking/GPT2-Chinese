#!/bin/bash

echo "decompressing files"
tar xf SogouCS.tar.gz
tar xf SogouCA.tar.gz
tar xf news_sohusite_xml.full.tar.gz
tar xf news_tensite_xml.full.tar.gz

echo "cat-ing txt and dat into a xml file"
cat *.txt *.dat > sogou.xml
rm *.txt *.dat

echo "turning gbk to utf-8"
luit -encoding gbk -c <sogou.xml >sogou_utf8.txt
rm sogou.xml

python extract_title_content.py
rm sogou_utf8.txt
