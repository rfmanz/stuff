#!/bin/bash

cp ~/SageMaker/keys/* ~/.ssh/
git config --global user.name <gitlab id>  # jxu-sofi
git config --global user.email <sofi-email>  # jxu@sofi.org
chmod 700 ~/.ssh/*