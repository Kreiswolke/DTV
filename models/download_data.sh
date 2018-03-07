cd models

# Download GloVe model
wget -c --no-check-certificate nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# Process to make compatible with gensim 
python -m gensim.scripts.glove2word2vec --input glove.6B.300d.txt --output glove2word2vec300d.txt 
# Remove not needed files
rm glove.6B.*

# Download Financal-News dataset embeddings 
wget https://ndownloader.figshare.com/files/10667560 -O FinancialNews.tar.gz
mkdir ../data/FinancialNews
tar -zxvf FinancialNews.tar.gz -C ../data/FinancialNews

cd ..

