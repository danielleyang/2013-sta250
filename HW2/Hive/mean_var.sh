#!/bin/sh

hadoop fs -mkdir data/
hadoop distcp s3://sta250bucket/groups.txt data/

hive -f mean_var.sql

mkdir -p output/
hadoop fs -cat output_means/* > ~/output/means.csv
hadoop fs -cat output_variances/* > ~/output/variances.csv

