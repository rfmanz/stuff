library(data.table)
library(DataExplorer)
library(readr)
library(easycsv)
library(scales)


unzip('~/Downloads/tabular-playground-series-apr-2021.zip',list = TRUE)

train  = fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  train.csv')
test = fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  test.csv')
sample_submission = fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  sample_submission.csv')

dim(train)
dim(test)
dim(sample_submission)

DataExplorer::plot_missing(train)
train[,.N,Sex]
train[,percent(.N/nrow(train)),.(Survived,Sex)][order(Survived)]
na.omit(train[Sex=='male',.N,.(Hmisc::cut2(Age,c(18,25,50,75)),Survived)])[order(-N), c(.SD,.(N = percent(N/nrow(na.omit(train)))))]




