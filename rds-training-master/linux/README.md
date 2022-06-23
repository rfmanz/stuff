## Linux Tutorial
---

Author: Pei-Ju (PJ) Sung

### Resources
---
* useful website
https://www.tutorialspoint.com/unix/index.htm
* example.sh in the current directory

### Commands
---
##### check server usage 
```bash
nvdia-smi
```

##### check storage
```bash
du -s $folder
```

##### list file
```bash
ls
ls -lrt
```

##### to folder
```bash
cd
```

##### copy folder
```bash
cp -r
```

##### rename folder
```bash
mv
```

##### remove
```bash
rm
```

##### read content
```bash
morels
cat
grep
cut
sort
```

##### grep last 10 lines
```bash
tail -f log.out
```

##### modified file
```bash
vi
```

##### read error msg from log
```bash
grep Error < test.log
```

##### Check process
```bash
ps
top |grep $username
```

##### Shell Scripting 
```bash
#!/bin/sh <-----bourne shell compatible script
#!/usr/bin/perl <---perl script
#!/usr/bin/php <----php script
#!/bin/false <------do-nothing script, because false retruns immediately anyways
```

##### run python
```bash
nohup python $pyfile.py > logs.out 2>&1 &
```

##### run hive
```bash
hive -f test.hql
```

##### Connect to server
```bash
ssh -l $username $ server
```