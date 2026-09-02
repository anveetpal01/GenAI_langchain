from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'John'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10)

new_student = {'age':'20', 'email':'dlsfd.com'}

student = Student(**new_student)

# print(type(student))
print(student)