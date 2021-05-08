# Machine Learning

---

## Definitions: 

- A machine learning algorithm is an algorithm that is able to learn from data. But what do we mean by learning? Mitchell (1997) provides a succinct deﬁnition: “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.” <br> <br>
- Machine learning enables us to tackle tasks that are too difficult to solve with ﬁxed programs written and designed by human beings.  For example, if we want a robot to be able to walk, then walking is the task. We could program the robot to learn to walk, or we could attempt to directly write a program that speciﬁes how to walk manually. <br><br>
- Machine learning tasks are usually described in terms of how the machine learning system should process an example. An example is a collection off features that have been quantitatively measured from some object or event that we want the machine learning system to process.
  <br>*5.1 Learning Algorithms* https://www.deeplearningbook.org/contents/ml.html
  

## Stats
### [Odds vs Probability](https://www.youtube.com/watch?v=ARfXDSkQf1Y)
![Odds vs Probability](1.png)<br>

Above odds, below probability.<br>

![Odds vs Probability](2.png)
<br>
From probability to odds: <br>
0.625/0.375
<br>
---
### Gradient Boosting

Additive modelling is the foundation of boosting. <br>
Additive modelling basically means we start from simple functions and add more on top. <br>
Example:<br>
We want to find the function which best fits this set of points(y values): 
![[Pasted image 20210508102757.png]]


![Additive Modelling](3.png)



Boosting is a loosely-defined strategy that combines multiple simple models into a single composite model. The idea is that, as we introduce more simple models, the overall model becomes a stronger and stronger predictor. In boosting terminology, the simple models are called weak models or weak learners.

Boosting constructs and adds weak models in a stage-wise fashion, one after the other, each one chosen to improve the overall model performance. 

> Resources:
  - https://explained.ai/gradient-boosting/L2-loss.html


