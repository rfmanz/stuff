# Training - Risk Data Science
---

First stop for any onboarding process. In this repo we will try to cover all day-to-day technology uses for our modeling endeavors. 

## Table of Content
---
- [x] useful-urls: useful links to important stuff
- [x] git: git tutorials
- [x] sagemaker-lifecycle-config: sample AWS sagemaker-lifecycle-configurations
- [x] project-setup: sample project repository
- [x] model2pmml: convert model/pickle files to PMML format for Credit Karma Lightbox

## Notes on some quick fixes
---

Common issues we run into from work and their potential fixes!

1. when  SDM is dead and not reponding in sagmaker, try the comand below and reconnect sdm:
    bash -c 'while true; do rm -rf /tmp/update*; sleep 120; done' &