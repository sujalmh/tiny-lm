import bz2

path = "wiki-dump\\enwiki-20260101-pages-articles.xml.bz2"

with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i > 50:
            break
        print(line.strip())