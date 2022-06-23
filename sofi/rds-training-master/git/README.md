### How to learn Git
---

##### Please learn Git! -- one of the minimum requirement for being an engineer.

In order to collaborate to on projects, learning Git is a must since it is effectively the language programmers speaks that maintains the social structure between engineers. 

#### Plan
---
0. Git setup: link GitLab with SageMaker (should we do this at the end?)
1. Basic idea: visualization in GitLab cheatsheet
2. How to learn Git
3. Basics: clone, add, commit, push


#### Resources
---

** All resources can be found in this directory (updated 2/7/2021) **

* GitLab Cheatsheet: [link](https://about.gitlab.com/images/press/git-cheat-sheet.pdf)
* GitHub Cheatsheet: [link](https://education.github.com/git-cheat-sheet-education.pdf)
* Git For Humans: the book that got me started. A very quick read that gives a mental model of the tool, and effectively making further studying significantly easier. 
* a more thorough Cheatsheet: [link](https://github.com/cmugpi/learn-git/tree/master/src)

How to Learn Git:
* Read the basics
* YouTube/Google Images for visual explainations
* Stackoverflow for text-based explainations/discussions

How to connect to gitlab
* Generate SSH Key [tutorial](https://docs.gitlab.com/ee/ssh/#rsa-ssh-keys) or use an existing one.
* Follow the [guide](https://docs.gitlab.com/ee/ssh/#adding-an-ssh-key-to-your-gitlab-account)

#### How to operate when things go smoothly.
---
Most can be found in the book and in the cheatsheets, but here are the topics that will get us going far enough -- solving 95% of the issue with 20% effort. 
* Topics to learn:
    * See GitLab Cheatsheet. The organization is amazing that the ideas are clusted by funcationalities.
* Topics for day-to-day work at SoFi: From the GitLab Cheatsheet
    * `01 Git configuration`: since we start SageMaker notebooks everyday
    * `02 Starting A Project`: clone
    * `03 Day-To-Day Work`: status, add, checkout, commit -m "message", rm, stash/reset
    * `04 Git branching model`: branch, checkout, merge
    * `08 Synchronizing repositories`: fetch, pull, push 


#### How to operate when something is wrong.

Since Git created for version control, the value of the tool appears mostly when something goes **WRONG**. So far the only time I got frustrated with Git was when resolving merge conflicts created when merging the branches. 
* Solve merge conflicts: [THE BEST video found](https://www.youtube.com/watch?v=xNVM5UxlFSA&ab_channel=Ihatetomatoes)
    * Merge Conflicts are represented in the following format:
        ```
           <<<<<<< HEAD 
           your code
           =======
           other's code
           >>>>>>>
        ```
    * But (IMO) it might be benefitial to forget about the order and resolve the conflict as if you are a 3rd party. **Please do not simply overwrite other people's code. Penalty applies.**
* troubleshooting: `git --help`
* Need to set up to track a specific branch
    * track upstream git repos: [link](https://www.git-tower.com/learn/git/faq/track-remote-upstream-branch/#:~:text=You%20can%20tell%20Git%20to,flag%20with%20%22git%20push%22.)

