# Decorators in Python

## Introduction
* a higher-order function that takes a function as argument and returns a extended/decorated function without modifying the input/bare function.
* Adding features to groups of functions and classes without modifying them

## Creating and using Decorators

* Simple decorator by wrapping a function
  def my_decorator(func):
      def wrapper(*args, **kwargs):
        // do somthing before the function is called
          func(*args, **kwargs)
        // do somthing after the function is called
      return wrapper

  @my_decorator
  def some_function(*args, **kwargs):
      // some behavior

## Examples

### using a decorator to keep track and get the models

* best_model.py

models = []

def register(func):
  models.append(func)
  return func

@register
def LSTM(data):
  model =
  f1_score =
  return f1_score, model

@register
def SVM(data):



def best_model(data):
  return max(model(data) for model in models)



