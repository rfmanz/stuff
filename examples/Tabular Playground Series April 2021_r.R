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
#Distribution of survived vs not survived of men by age ranges 
na.omit(train[Sex=='male',.N,.(Age=Hmisc::cut2(Age,c(18,25,50,75)),Survived)])[order(-N), c(.SD,.(age_group_pct = percent(N/nrow(na.omit(train)))))][order(Age)]

na.omit(train[Sex=='male',.N,.(Age=Hmisc::cut2(Age,c(18,25,50,75)),Survived)])[order(Age),.GRP,Age]

na.omit(train[Sex=='male',.(.N,.GRP),.(Age=Hmisc::cut2(Age,c(18,25,50,75)))])




na.omit(train[Sex=='male', .N,.(Hmisc::cut2(Age,c(18,25,50,75)),Sex)])[order(-N), c(.SD, .(Pct =percent(N/na.omit(train[,.N]))))][,c(2,1,3,4)]

#25 to 50 age range survival rate

na.omit(train[Sex=='male',.N,.(Hmisc::cut2(Age,c(25,50),minmax = FALSE),Survived)])[order(-N), c(.SD,.(N = percent(N/na.omit(train[Sex=='male' & between(Age,25,50),.N]))))]

na.omit(train[Sex=='male',.N,.(Hmisc::cut2(Age,c(25,50),minmax = FALSE),Survived)])[order(-N), c(.SD,.(N = percent(N/na.omit(train[,.N]))))]

#Survival by age men 
na.omit(train[Sex=='male' & Survived==1,.N,.(Hmisc::cut2(Age,c(18,25,50,75)))])[order(-N), c(.SD,.(N = percent(N/nrow(na.omit(train)))))]

na.omit(train[Sex=='male', .(.N,percent(.N/na.omit(train[,.N]))),Survived])
na.omit(train[,.N,Sex])
na.omit(train[Sex=='male' & Survived ==1 , .N, Hmisc::cut2(Age,c(18,25,50,75))])[order(-N)]





Base_pol[,prop.table(table(is.na(FECANUL)))]