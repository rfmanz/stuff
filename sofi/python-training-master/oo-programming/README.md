# Object-Oriented-Programming with Python

* [What is Object-oriented Programming](#What-is-Object-oriented-Programming)
  - [Concept](#Concept)
  - [Key Features](#Key-Features)
  - [Benefits](#Benefits)
* [Class, Object and State](#Class,-Object-and-State)
  - [Class](#Class)
  - [Object](#Object)
  - [State](#State)
* [Creating Classes](#Creating-Classes)
  - [Class Definition](#Class-Definition)
  - [Type vs. Class](#Type-vs.-Class)
  - [Add attributes to a class](#Add-attributes-to-a-class)
  - [Add methods to a class](#Add-methods-to-a-class)
  - [The Python data model](#The-Python-data-model)
* [Create an object from a class](#Create-an-object-from-a-class)

* [Organizing Classes and Objects](#Organizing-Classes-and-Objects)

* [Case Study](#Case-Study)


## What is Object-oriented Programming

* OOP is a programming paradigm based on the concept of “objects”,
  which can contain data, in the form of attributes, and code, in the form of methods.
  - Objects are a representation of real-world and communicate with each other via messages.

* OOP vs. procedural programming (PP):
  - PP solves a ML problem by working on data and calling procedures (known as functions).
  - OOP takes a step back and tries to understand the problem from an "actor" perspective.

    - step 1 identify actors and look at what they do (actions), which are their behaviors.
    - step 2 look at any data that's needed to carry out these actions.
    - step 3 change state of an object via its data by calling methods of the object.
    - step 4 interact with other objects via _message passing_ by calling a method on the object.


* OOP vs. object-oriented Design (OOD)
  - OOP is as much a design exercise as it is about writing the code to implement its logic.
  - OOD is about learning to identify the actors, the data needed, their behaviors, and
    how they interact with other actors.

### Key Features

* abstraction: the process of taking something concrete and stripping it of specifics (e.g. black-box abstraction)
* encapsulation: put something inside a black-box (e.g. Function, Class, Module, Package)
* inheritance: is the mechanism of deriving new classes from existing ones to support code reusability
* polymorphism: refers to the use of a single type entity (method, operator or object) to represent different types in different scenarios.

### Benefits

* Access Cotrol: hiding data away from the rest of the system and control their access.
* Managing Complexity: Building large ML systems is a complex task, with many problems to solve.
* Modifying and maintaining ML pipleines are easy.
* Support team development including code reuse and sharing

## Class, Object and State

### Class - is a blueprint or a template for creating objects

* Classes form the basis for all data types.


### Object - is an abstraction for data. 

* All data in a Python program is represented by objects.
* It is the instance of a class and represents an actor who does something within a system.
* An identifier (also known as a name) is implicitely associated with the _memory address_ of the object to which it refers.

  Example:

```
x = 10
x is an identifier
the object expressed on the right side of the assignment is an integer object with the value of 5.
```
>_Note_:
> Although there is no advance declaration associating an identifier with a particular type (_dynamic typed_), the object to which it refers has a definite type.


### State 

* like PP, OOP uses variables to hold information and keep track of state, but unlike PP, variables in OOP are attached to objects instead of being defined on their own.
* OOP refers to variables on an object as attributes that can be used to hold its state.


## Creating Classes

### Class Definition

- A class in Python is created by using the keyword _class_ and giving it a name
- All classes are inherited from a built-in base class called _object_ (the superclass)

Example:

   
```python
  >>> object
```

```
  <class 'object'>
```

```python
  >>> class FileReader:
        pass
  >>> FileReader
```

```
  <class '__main__.FileReader'>
```

```python
  >>> class FileReader(object):
        pass
  >>> FileReader
```

```
  <class '__main__.FileReader'>
```

  - Check the parent class

```python
  >>> object.__bases__
```

```
 ()
```
```python
 >>> FileReader.__bases__
```

```
 (<class 'object'>,)
```


- use dir(cls) to list all the special methods - the default behaviors

```python
  >>> dir(object)
```

```
  ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
  '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__',
  '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
  '__sizeof__', '__str__', '__subclasshook__']
```


### Type vs. Class

- _Type_ is a metaclass that inherits from the object class, of which classes are instances.
    However, type is also an instance of the type metaclass, so it is an instance of itself.

>_Note_:
> Prior to v2.2, Python had both types and classes. Python 3 completed the unification of types and classes, so it is reasonable to refer to an object’s type and its class interchangeably.
> The python hierarchy is type (the Metaclass) -> class -> instance.
> Think of the function type() as going one level up and returns the class of the instance.
> Advanced: the type function is used to create class object in meta programming.
> Remember that, in Python, everything is an object. Classes are objects as well. As a result, a class must have a type.


  Example:

```python
  >>> type
```

```
  <class 'type'>
```

```python
  >>> type.__base__
```

```
  <class 'object'>
```

  Apply to a class returns the class of the class (the Metaclass).

  All classes are in fact types.

```python
  >>> type(object)
```

```
  <class 'type'>
```

```python
  >>> type(type)
```

```
  <class 'type'>
```

  The type of the built-in classes is also type:

```python
  >>> for t in int, float, dict, list, tuple:
  ...     print(type(t))
  ...
```

```
  <class 'type'>
  <class 'type'>
  <class 'type'>
  <class 'type'>
  <class 'type'>
```

  user-defined class

```python
  >>> type(FileReader)
```

```
  <class 'type'>
```

  Apply to a class instance returns the class of the instance.

```python
  >>> ty=type
  >>> type(ty)

```

```
  <class 'type'>
```

```python
  >>> obj=object
  >>> type(obj)
```

```
  <class 'type'>
```

```python
  >>> type(2)
```

```
  <class 'int'>
```

```python
  >>> type(fr)
```

```
  <class '__main__.FileReader'>
```



### Add attributes to a class

  - Knowing what attributes (variables) that should be added to a class is part of OOD.
  - At the object's creation time using a constructor -
    the special function __init()__ with a special keyword, self, as the first parameter and is called implicitly.
    The keyword self refers to the object's instance.
  - In this method, you create the attributes the object should have and assign any initial values.

  Example:

```python
  >>> class FileReader:
        def __init__(self, path):
          self.path =
```


### Add methods to a class
  
- There are 4 types: instance, class, static and abstract methods


  Example:

```python
  class FileReader(ABC):

    def instance_method(self):
        print("This is an instance method ", self)

    @classmethod
    def class_method(cls):
        print("This is a class method ", cls)

    @staticmethod
    def static_method():
        print("This is a static method")

    @abstractmethod
    def abstract_method():
        raise NotImplementedError
```


- _Instance method_ is a method that is bound to the instance of the class i.e. object.
  It takes self parameter as an argument which holds the reference to an instance of the class.
  It can access both object and class attributes.
  Using the self.__class__ attribute, any class variable can be accessed inside the method.
  It can only be called using an object reference.

  Example:

```python
    >>> fr = FileReader()
    >>> fr.instance_method()
```

```
    This is an instance method <__main__.FileReader object at 0x109c464a8>
```

```python
    >>> FileReader.instance_method()
```

```
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: instance_method() missing 1 required positional argument: 'self'
```

  - _Class method_ is a method with ‘@classmethod‘ decorator in the function declaration.
    It is bounded by the class not by an instance and accepts cls parameter which holds the reference of the class.
    It can only access class attributes, but not object attributes.
    It can be called using both object and class reference.

    Example:

```python
    >>> fr.class_method()
```

```
    This is a class method <class '__main__.FileReader'>
```

```python
    >>> FileReader.class_method()
```

```
    This is a class method <class '__main__.FileReader'>
```

>_Note_:    
> Use class methods when you want to access the class state or variables inside the function.

  - _Static method_ is a method with ‘@staticmethod‘ decorator in the function declaration.
    Neither self or cls are passed as an argument. Hence it cannot access or modify any class or instance attributes.
    It can be called using both class and object reference.

    Example:

```python
    >>> fr.static_method()
```

```
    This is a static method
```

```python
    >>> FileReader.static_method()
```

```
    This is a static method
```

>_Note_:    
> Static methods are best suited for the task that is independent of every aspect of the class.

  - Abstract method is a method which is declared but does not have implementation.
    A Child class is responsible for providing an implementation for the parent class abstract method.


>_Note_:    
> Abstract methods ....


### The Python data model

- It can be viewed as the ‘_Python Framework_’ which formalizes the API interfaces of the basic building blocks of the language itself, such as sequences, iterators, functions, classes, and so on.

- Its API consits a set of special methods that are written with leading and trailing double underscores.

- By implementing special methods, your custom data types (objects) can behave like the built-in types.

- Below are special methods we almost always need to override their default implementations:

  -  __init__
  -  __repr__ for debugging and logging
  -  __str__ for presentation to end users
  -  __format__

>_Note_:
> Special methods are meant to be called by the Python interpreter, and not by your program unless you are doing a lot of metagrogramming. 
> If you need to invoke a special method (e.g. __len__), it is better to call the related built-in function (e.g. len()).

## Create an object from a class

- Instantiation 
  Creating an object from a class is called instantiate.
  This basically means "asking the operating system, with this template (the class) and these starter values, to give the program job enough memory and create an object.

- Characteristics of an object 
  Every object has an identity, a type and a value.

  - An object’s identity never changes once it has been created;
    we may think of it as the object’s address in memory.
    - Using the id() function to get an integer representing its identity.
    - Using the ‘is’ operator to compare the identity of two objects;

  Example:

```python
  >>> fr1 = FileReader()
  >>> id(fr1)
```

```
  4459146992
```
```python
  >>> fr2 = FileReader()
  >>> id(fr2)
```

```
  4459147216
```
```python
  >>> print(fr1 is fr2)
```

```
  False
```
```python
  >>> fr2=fr1
  >>> id(fr2)
```

```
  4459146992
```

```python
  >>> print(fr1 is fr2)
```

```
  True
```


  - An object’s type specifies two things:
    – what operations are allowed
    – the set of values the object can hold

    - Using the type() function to check its type

  Example:

```python
  >>> type(fr)
```

```
  <class '__main__.FileReader'>
```
  However, fr is an instance of FileReader, so its type is FileReader.


  - An object’s value is the data it holds and can be mutable or immutable.
   The mutability is determined by its type (e.g. numbers are immutable, while lists are mutable).
   - Using == to compare the equality of values



### Object's lifecycle

* instantiation

  - is the process of creating a new instance of a class.
  - ways for instantiating an object
    - by invoking the constructor of a class
    - many Python built-in classes support a literal form designating new instances (e.g. x=5).
    - some Python built-in functions return a new instance of a class (e.g. sorted)
    
>_Note_:
> When we create an object, our program asks the operating system for resources, namely, memory,
  to be able to construct the object based on its class definition.


- initialization

- destruction




## Organizing Classes and Objects





## Case Study

- Machine learning workflow
  No matter which paradigm is used, programs use the same series of steps to solve ML problems:

  - Pre-processing: Input data is ingested, validated, and prepared for modeling
  - In-processing: Data has been transformed, features are selected, and parameters have been optimized
  - Post-processing: Model has been created, explained, and evaluated.


- Identify objects, data, and behavior

Phase     Actor         Behavior   Data
Input     FileReader    read       path


