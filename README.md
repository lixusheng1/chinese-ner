# chinese-ner
# chinese-ner

基于BGRU-CRF或BILSTM-crf的中文命名实体识别

首先新建文件夹data，再训练后下载一个词向量文件存在在data文件夹下，修改model/config.py的文件路径

基于python3，tensorflow>=1.2

1.python3  build_data.py

2. python3 train.py

3. python3 evaluate.py
