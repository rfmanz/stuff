grades = []
grades.append((95, 0.45))
grades.append((85, 0.55))
total = sum(score * weight for score, weight in grades)
total_weight = sum(weight for _, weight in grades)
average_grade = total / total_weight

from dataclasses import dataclass


@dataclass
class Position:
    name: str
    lon: float
    lat: float


Position(name, lon, lat)

Position.__delattr__(self, name)


from dataclasses import dataclass
from typing import List


@dataclass
class Dogs:
    names: List[str]
    ages: List[int]

    def __post_init__(self):
        self.info = [Dog(name, age) for name, age in zip(self.names, self.ages)]


a = Dogs()
a.names = "Fido"
a.age = 3
a.info


class Building(object):
    def __init__(self, floors):
        self._floors = [None] * floors

    def occupy(self, floor_number, data):
        self._floors[floor_number] = data

    def get_floor_data(self, floor_number):
        return self._floors[floor_number]


building1 = Building(4)  # Construct a building with 4 floors
building1.occupy(0, "Reception")
building1.occupy(1, "ABC Corp")
building1.occupy(2, "DEF Inc")
print(building1.get_floor_data(2))

# iterrows

# https://github.com/twolodzko/getter/blob/main/getter/decorators.py


from dataclasses import dataclass


@dataclass()
class Student:
    name: str
    clss: int
    stu_id: int
    marks: []
    avg_marks: float

    def average_marks(self):
        return sum(self.marks) / len(self.marks)


student1 = Student("HTD", 10, 17, [11, 12, 14], 50.0)

print(student)

Student(name="HTD", clss=10, stu_id=17, marks=[11, 12, 14], avg_marks=50.0)

student.average_marks()


from dataclasses import dataclass, field


@dataclass()
class Student:
    name: str
    clss: int
    stu_id: int
    # marks: []
    # avg_marks: float = field(init=False)

    # def (self):
    #     self.avg_marks = sum(self.marks) / len(self.marks)


student.average_marks

student = Student("HTD", 10, 17, [98, 85, 90])
student = Student("HTD", 10, 17)

print(student)

Student(name="HTD", clss=10, stu_id=17, marks=[98, 85, 90], avg_marks=91.0)

print(student.__dataclass_fields__)

np.random.rand

x1 = np.random.randint(10, size=6)
x1
x2 = np.random.randint(10, size=(3, 4))
x2
x3 = np.random.randint(10, size=(3, 4, 5))
x3


# TODO:
# READ about PROTOBUF
# READ unittests
# proto utils - download the folder
# start to think about things that come up a lot and ways to automate stuff
