# DTV
Source code for the Dynamic Topic Vector model.
Code to reproduce the results from experiments on the Financial-News dataset from https://github.com/philipperemy/financial-news-dataset.

# Requirements
Code requires a gensim-compatible version of GloVe embeddings which can be obtained by calling:

`./models/download_data.sh`

Pre-processed document embeddings are included in /data/FinancialNews/ and the gensim-usable GloVe model is stored at /models/glove2word2vec300d.txt.

# Experiments

To run one experiment, specify a SAVEDIR and optionally a list of number of topics X in

`./run_experiments_FN.sh`,

  then run from a shell.
  
 After successful training, for every year results are stored in the specified SAVEDIR.
 This includes a pickled results dictionary res_n_X.p and a topics_n_X.txt file. The latter contains the top-10 words 
 descriptive for one generated topic.
 
 
 

