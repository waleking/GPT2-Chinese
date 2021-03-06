# 如何利用SogouCA数据集和SogouCS数据集进行预训练

## 下载数据集
运行如下命令，下载`SogouCS.tar.gz`，`SogouCA.tar.gz`，`news_sohusite_xml.full.tar.gz`，`news_tensite_xml.full.tar.gz`四个文件。依照[全网新闻数据(SogouCA)](https://www.sogou.com/labs/resource/ca.php)和[搜狐新闻数据(SogouCS)](https://www.sogou.com/labs/resource/cs.php)的描述，这四个文件包含2008年，2012年抓取到的新闻或者网页。
```
bash download_sogou_datasets.sh
```

## 数据预处理
由于这四个压缩包所包含的文件的编码都是gbk的，因此需要使用luit工具将其编码转换为UTF-8。并且，每个新闻包含在xml格式之中，我们需要利用正则表达式将标题和正文抽取。运行如下的命令。
最终得到的文件为`sogou_utf8_title_content.txt`，其每一行对应了一篇新闻，格式为`标题\t内容\n`。文件大小24GB，包含新闻或网页8,800,897篇。
```
bash preprocess.sh
```

