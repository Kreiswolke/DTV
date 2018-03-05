cd models

wget -c --no-check-certificate nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
python -m gensim.scripts.glove2word2vec --input glove.6B.300d.txt --output glove2word2vec300d.txt 

cd ..

