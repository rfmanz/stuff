# Object-oriented Programming
## Introduction
* **What's Object-oriented Programming**
    * Object-oriented programming is a method of structuring a program by bundling related properties and behaviors into individual objects
    * Procedural-oriented Programming vs. Object-oriented Programming
    <br /> <img src="https://gitlab.com/sofiinc/data-science-risk/mlops/rds-training/-/raw/5b443b2791acb003f3b24be9d05a4eec9348a61b/object-oriented-programming/Pic/oop_vs_pp.png" width='650' height='350'>
    * Example:
    <br /> Object-oriented Programming can be compared to a 'closed access' library
    >  books and publications in a library -> data in an object-oriented programming
    <br /> access to the books is restricted -> access to the data is ristricted in object-oriented programming
    <br /> getting/returning a book only possible via the staff -> access to the data is controled through methods

* **Advantages of Object-oriented Programming**
    * Security
    * Reusability
    * Easy partition of work
    * Maintenance


* **Object-oriented Programming Vocabulary**
    * OOP: a commonly used abbreviation for object-oriented programming
    * class: a blueprint consisting of methods and attributes
    * object: an instance of a class
    * attribute: a descriptor or characteristic
    * method: an action that a class or object could take
    >  function vs. method: 
    <br />1. both use the `def` keyword, both have inputs and return outputs
    <br />2. a method is inside of a class; a function is outside of a class
    <br />3. the first parameter in the definition of a method has to be a reference to the instance; the parameter is usually called `self`. The `self` tells Python where to look in the computer's memory for the specific object, it's a name convention.
* **Example**
```python
class Rectangle:
    """ A Python object that describes the properties of a rectangle """
    def __init__(self, width, height, center=(0.0, 0.0)):
        """ Sets the attributes of a particular instance of `Rectangle`.

            Parameters
            ----------
            width : float
                The x-extent of this rectangle instance.

            height : float
                The y-extent of this rectangle instance.

            center : Tuple[float, float], optional (default=(0, 0))
                The (x, y) position of this rectangle's center"""
        self.width = width
        self.height = height
        self.center = center

    def __repr__(self):
        """ Returns a string to be used as a printable representation
            of a given rectangle."""
        return "Rectangle(width={w}, height={h}, center={c})".format(h=self.height,
                                                                     w=self.width,
                                                                     c=self.center)
    def compute_area(self):
        """ Returns the area of this rectangle

            Returns
            -------
            float"""
        return self.width * self.height

    def compute_corners(self):
        """ Computes the (x, y) corner-locations of this rectangle, starting with the
            'top-right' corner, and proceeding clockwise.

            Returns
            -------
            List[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]"""
        cx, cy = self.center
        dx = self.width / 2.0
        dy = self.height / 2.0
        return [(cx + x, cy + y) for x,y in ((dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy))]
```
An instance of this `Rectangle` class is an individual rectangle whose attributes include its width, height, and center-location. Additionally, we can use the rectangle’s methods (its attributes that are functions) to compute its area and the locations of its corners.
<br />Output
```python
# create a rectangle of width 4, height 10, centered at (0, 0)
# here __init__ is executed and the width/height/center attributes are set
>>> rect1 = Rectangle(4, 10)

# the __repr__ method defines how a rectangle instance will be displayed here
# in the console
>>> rect1
Rectangle(width=4, height=10, center=(0, 0))

# compute the area for this particular rectangle
>>> rect1.compute_area()
40

# compute the corner-locations of this rectangle
>>> rect1.compute_corners()
[(2.0, 5.0), (2.0, -5.0), (-2.0, -5.0), (-2.0, 5.0)]
```
`dir(object_name)` can return list of the attributes and methods of any object.
## 4 Major Principles of OOP
* **Encapsulation**
<br />Encapsulation allows you to hide implementation details and combine functions and data all into a single entity
* **Abstraction**
<br /> a process of handling complexity by hiding unnecessary information from the user. That enables the user to implement even more complex logic on top of the provided abstraction without understanding or even thinking about all the hidden background/back-end complexity.
<br /> <img src="https://gitlab.com/sofiinc/data-science-risk/mlops/rds-training/-/raw/58ae26a9ac15719c2d71116fbf1a0e404699950c/object-oriented-programming/Pic/abstraction_vs_encapsulation.png" width='650' height='350'>

* **Polymorphism**
<br />Polymorphism is an ability to use a common interface for multiple forms. We can use the concept of polymorphism while creating class methods as Python allows different classes to have methods with the same name. We can then later generalize calling these methods by disregarding the object we are working with.
<br />Example:
```python
class Parrot:
    
    def fly(self):
        print("Parrot can fly")
        
    def swim(self):
        print("Parrot can't swim")
        
class Penguin:
    
    def fly(self):
        print("Parrot can't fly")
        
    def swim(self):
        print("Parrot can swim")
        
# common interface
def flying_test(bird): 
    bird.fly()
    
# instantiate objects
>>> blu = Parrot()
>>> peggy = Penguin()

# passing the objects
>>> flying_test(blu)
Parrot can fly

>>> flying_test(peggy)
Parrot can't fly
```
* **Inheritance**
<br />It refers to defining a new class with little or no modification to an existing class. The new class is called derived (or child) class and the one from which it inherits is called the base (or parent) class. Derived class inherits features from the base class where new features can be added to it. This results in re-usability of code.
<br />Example:
<br />Recall that we have wrote `Rectangle` class in the introduction section. Now suppose that we want to write a `Square` class. Recognize that a square is a special type of rectangle. We can leverage the code for `Rectangle`.
```python
# Creating Square, a subclass of Rectangle
class Square(Rectangle):
    def __init__(self, side, center=(0, 0)):
        # equivalent to `Rectangle.__init__(self, side, side, center)`
        super().__init__(side, side, center)
```
<br />The built-in `issubclass` function allows us to verify the relation ship between `Square` and `Rectangle`
```python
# `Square` and `Rectangle` are distinct classes
>>> Square is not Rectangle
True

# `Square` is a subclass of `Rectangle`
>>> issubclass(Square, Rectangle)
True

# `my_square is an both an instance of `Square` and `Rectangle`
>>> isinstance(my_square, Square)
True

>>> isinstance(my_square, Rectangle)
True
```
## Special (Magic/Dunder) Method
In Python, special methods are a set of predefined methods you can use to enrich your classes. They start and end with double underscores.
* **`__init__`**
<br />a method which is immediately and automatically called after an instance has been created, used to initialize an instance
* **`__len__`**
<br />this method returns the length of the object. It is executed when we perform len() method
```python
>>> my_list = ["a", "b", "c"]
>>> len(my_list)  
3

>>> my_list.__len__() 
3
```
> note: len() is the public interface we use to get the length of an object. The `__len__` method is the implementation that an object that supports the concept of the length is expected to implement
* **`__add__`**
<br /> this method is called when we attempt to add two numbers, it's one of the operator magic methods
```python
>>> a=1
>>> b=2
>>> a+b   
3
>>> a.__add__(b)
3
```
* **`__repr__`**
<br />the `__repr__` method is executed when we want to create a developer-friendly representation of an object
* **`__str__`**
<br />the `__str__()` method is executed when we want to print an object in a printable format
* **`__del__`**
<br />destructor, it is called when the instance is about to be destroyed and if there is no other reference to this instance
* etc
