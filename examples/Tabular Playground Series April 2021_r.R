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

#age range men 
na.omit(train[Sex=='male', .N,.(Hmisc::cut2(Age,c(18,25,50,75)),Sex)])[order(-N), c(.SD, .(Pct =percent(N/na.omit(train[,.N]))))][,c(2,1,3,4)]


#Distribution of survived vs not survived of men by age ranges | total percent 
na.omit(train[Sex=='male',.N,.(Age=Hmisc::cut2(Age,c(18,25,50,75)),Survived)])[order(-N), c(.SD,.(age_group_pct = percent(N/nrow(na.omit(train)))))][order(Age)]

#Distribution of survived vs not survived of men by age ranges | age group percent  
na.omit(train[Sex=='male',
              {GRP = .GRP-1
              .SD[,.(.N,GRP),Survived]},.(Age=Hmisc::cut2(Age,c(18,25,50,75)))][,c(.SD, .(pct = percent(N/sum(N)), Sex = "Male")),GRP][,-"GRP"][,c(5,seq(1,4))][order(Age,Survived)])

hist(train[Sex=='male',.N])

na.omit(train[Sex=='female',
              {GRP = .GRP-1
              .SD[,.(.N,GRP),Survived]},.(Age=Hmisc::cut2(Age,c(18,25,50,75)))][,c(.SD, .(pct = percent(N/sum(N)), Sex = "female")),GRP][,-"GRP"][,c(5,seq(1,4))][order(Age,Survived)])



#25 to 50 age range survival rate

na.omit(train[Sex=='male',.N,.(Hmisc::cut2(Age,c(25,50),minmax = FALSE),Survived)])[order(-N), c(.SD,.(N = percent(N/na.omit(train[Sex=='male' & between(Age,25,50),.N]))))]

na.omit(train[Sex=='male',.N,.(Hmisc::cut2(Age,c(25,50),minmax = FALSE),Survived)])[order(-N), c(.SD,.(N = percent(N/na.omit(train[,.N]))))]

#Survival by age men 
na.omit(train[Sex=='male' & Survived==1,.N,.(Hmisc::cut2(Age,c(18,25,50,75)))])[order(-N), c(.SD,.(N = percent(N/nrow(na.omit(train)))))]

na.omit(train[Sex=='male', .(.N,percent(.N/na.omit(train[,.N]))),Survived])
na.omit(train[,.N,Sex])
na.omit(train[Sex=='male' & Survived ==1 , .N, Hmisc::cut2(Age,c(18,25,50,75))])[order(-N)]



na.omit(train[,.N,.(Sex, Survived)])

Base_pol[,prop.table(table(is.na(FECANUL)))]

print(t[,.(names = names(t), p = lapply(.SD, function(x) sum(x==0)), t= lapply(.SD, function(x) scales::percent(sum(x==0)/nrow(t))))][order(as.numeric(p)),],topn=150) 
