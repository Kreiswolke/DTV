#!/bin/bash

# To specify:
# Save directory
SAVEDIR="" 
# List of number of topics, default is 50 topics
ntopics=(50)  

# Run experiment
years=(2006 2007 2008 2009 2010 2011 2012 2013)

for year in "${years[@]}"
do
    for i in "${ntopics[@]}"
    do
	if [ "$year" == "2006" ]
	  then 
              python /home/oliver/Desktop/RandomTopicsScripts/run_STV_dynamic.py --savedir "$SAVEDIR/$year" --t $i --lr 0.01 --gamma 0.99 --maxiter 10000 --miniter 50 --bs 500 --case "FN_$year" 
	  else
	      python /home/oliver/Desktop/RandomTopicsScripts/run_STV_dynamic.py --savedir "$SAVEDIR/$year" --t $i --lr 0.01 --gamma 0.99 --maxiter 10000  --miniter 50 --bs 500 --case "FN_$year"  --weight "$SAVEDIR/$last_year/t2v/res_n_$i.p"
	fi
        last_year=$year

    done
done
