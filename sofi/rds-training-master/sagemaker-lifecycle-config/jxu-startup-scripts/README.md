## How To
---

Pro:

SageMaker Lifecycle Config installation script times out after 5 mins, so it is a bad idea to install a lot of things in your config script as your instance will never wake up. To circumvent this issue, we simply start the instance using a minimal setup, then complete configuration using a script `pre-saved` on S3.

Con:

This is a version that is inherited and in someways overly-complicated...maybe just use this as a reference. 


If you still wanna use this script...here is how it works. Please read and modify before borrowing it.

* put your `startup.sh` on S3. e.g. `startup.sh`, `startup-autostop.sh`
* include the following scipt (`jxu-lifecycle-config.sh`) in the place where we usually put lifecycle configurations.

```bash
# copy the startup.sh script from s3
aws s3 cp '<s3 path to your start script>' /home/ec2-user/SageMaker/

# change the access permission of the copied script
chmod +x /home/ec2-user/SageMaker/startup.sh

# run the script in the background
nohup /home/ec2-user/SageMaker/startup.sh &
```

The script will be saved in the SageMaker directory on your instance and will be excuting in the background. Depending on how much stuff `startup.sh` contains, probably need to wait for some time before using your instance.