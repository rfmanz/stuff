### Estimador de ingresos

Dates 
```python
from datetime import datetime, timedelta, date
cierre = '2019-01-31'
datetime.strptime(cierre,'%Y-%m-%d').strftime('%Y-%m-%d')
```
Get today's day of the week as number 
```python
date.weekday(datetime.strptime(datetime.today().strftime('%Y-%m-%d'),'%Y-%m-%d'))
```
