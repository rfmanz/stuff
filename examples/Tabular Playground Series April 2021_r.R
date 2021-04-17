library(data.table)
library(DataExplorer)
library(readr)
library(easycsv)
library(scales)
library(ggplot2)
library(Hmisc)
library(xray)
anomalies(train)
# load ---- 
unzip('~/Downloads/tabular-playground-series-apr-2021.zip',list = TRUE)

train  = fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  train.csv')
test = fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  test.csv')
sample_submission = fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  sample_submission.csv')

# dim(train)
# dim(test)
# dim(sample_submission)
 


# survived by sex ---- 
# use this for proportions of any category
prop.table(table(train[Sex=='male',.(Survived,Sex)]))*100
prop.table(table(train[Sex=='female',.(Survived,Sex)]))*100
# ok so the survival rates for the sexes are opposite. 
# so lets start with the smallest group. men who survived by age 

ggplot(train[Sex=='male'], aes(Age, fill=factor(Survived))) + geom_histogram(color='black') + scale_x_continuous(breaks =  seq(0,85,5))

ggplot(train[Sex=='male' & Survived==0], aes(Age, fill=factor(Pclass))) + geom_histogram(color='black') + scale_x_continuous(breaks =  seq(0,85,5))
x11()


# then women 

ggplot(train[Sex=='female'], aes(Age, fill=factor(Survived))) + geom_histogram(color='black') + scale_x_continuous(breaks =  seq(0,85,5))


# Distribution of survived vs not survived of men by age ranges | age group percent  ----
na.omit(train[Sex=='male',
              {GRP = .GRP-1
              .SD[,.(.N,GRP),Survived]},.(Age=Hmisc::cut2(Age,c(18,25,50,75)))][,c(.SD, .(pct = percent(N/sum(N)), Sex = "Male")),GRP][,-"GRP"][,c(5,seq(1,4))][order(Age,Survived)])



# age range men  ---- 
na.omit(train[Sex=='male', .N,.(Hmisc::cut2(Age,c(18,25,50,75)),Sex)])[order(-N), c(.SD, .(Pct =percent(N/na.omit(train[,.N]))))][,c(2,1,3,4)]


# Distribution of survived vs not survived of men by age ranges | ----
na.omit(train[Sex=='male',.N,.(Age=Hmisc::cut2(Age,c(18,25,50,75)),Survived)])[order(-N), c(.SD,.(age_group_pct = percent(N/nrow(na.omit(train)))))][order(Age)]




na.omit(train[Sex=='female',
              {GRP = .GRP-1
              .SD[,.(.N,GRP),Survived]},.(Age=Hmisc::cut2(Age,c(18,25,50,75)))][,c(.SD, .(pct = percent(N/sum(N)), Sex = "female")),GRP][,-"GRP"][,c(5,seq(1,4))][order(Age,Survived)])

plot_density(train[Sex=='male',Age])
hist(train[Sex=='male',Age])
DataExplorer::plot_histogram(train[Sex=='male',Age],geom_histogram_args = list('fill'= 'blue',"color" = 'black' ))

#25 to 50 age range survival rate ----

na.omit(train[Sex=='male',.N,.(Hmisc::cut2(Age,c(25,50),minmax = FALSE),Survived)])[order(-N), c(.SD,.(N = percent(N/na.omit(train[Sex=='male' & between(Age,25,50),.N]))))]

na.omit(train[Sex=='male',.N,.(Hmisc::cut2(Age,c(25,50),minmax = FALSE),Survived)])[order(-N), c(.SD,.(N = percent(N/na.omit(train[,.N]))))]


#Survival by age men ----
na.omit(train[Sex=='male' & Survived==1,.N,.(Hmisc::cut2(Age,c(18,25,50,75)))])[order(-N), c(.SD,.(N = percent(N/nrow(na.omit(train)))))]

na.omit(train[Sex=='male', .(.N,percent(.N/na.omit(train[,.N]))),Survived])
na.omit(train[,.N,Sex])
na.omit(train[Sex=='male' & Survived ==1 , .N, Hmisc::cut2(Age,c(18,25,50,75))])[order(-N)]




# pct distribution of nas  ---- 
train[,prop.table(table(is.na(Age)))]

# Find zeroes and nas by column----
print(train[,.(names = names(train), p = lapply(.SD, function(x) sum(x==0)), t= lapply(.SD, function(x) scales::percent(sum(x==0)/nrow(t))))][order(as.numeric(p)),],topn=150) 

train[,.(names = names(train), p = lapply(.SD, function(x) sum(is.na(x))), t= lapply(.SD, function(x) scales::percent(sum(is.na(x))/nrow(train))))][order(as.numeric(p)),]












