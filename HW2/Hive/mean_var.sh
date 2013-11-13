#!/bin/sh
# mean_var.sh
# Author: Nick Ulle
# Description:
#   This script, along with its companion script mean_var.sql, loads the
#   groups.txt data into Hadoop and uses Hive to compute the within-group means
#   and variances. The results are stored to the local file system's output/
#   directory as CSV files.

hadoop fs -mkdir data/
hadoop distcp s3://sta250bucket/groups.txt data/

hive -f mean_var.sql

mkdir -p output/
hadoop fs -cat output_means/* > ~/output/means.csv
hadoop fs -cat output_variances/* > ~/output/variances.csv

