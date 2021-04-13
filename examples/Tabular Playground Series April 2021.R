library(data.table)
library(DataExplorer)
library(readr)
library(easycsv)

path  = '~/Downloads/tabular-playground-series-apr-2021.zip'
paste0('unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  train.csv'))

fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  train.csv')[1,'Name']
fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  train.csv')[2,'Name']
fread(cmd = 'unzip -p /home/r/Downloads/tabular-playground-series-apr-2021.zip  train.csv')[3,'Name']

asdsd
eval(parse(text=paste(set[[i]],"$",var1,sep="")))


DT = do.call(rbind, lapply(files, fread))
# The same using `rbindlist`
DT = rbindlist(lapply(files, fread))

DataExplorer::profile_missing(na.omit(train))

ggplot(na.omit(train), aes(x=train$Embarked, fill = train$Survived)) + 
  geom_bar(stat="count", position=position_dodge())


ls()

train[,.(),Survived]

