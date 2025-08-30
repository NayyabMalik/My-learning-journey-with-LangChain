#pydantic is library responsible to check data type is valid or not
from pydantic import BaseModel,Field
from typing import Optional
class person(BaseModel):
    name:str 
    age:Optional[int]=None
    cgpa:float=Field(gt=0,lt=4,default=0) ## constrain cgpaa should be within range of 0-4 otherwise 0


new_person={'name':'nayyab','age':21,'cgpa':3.92} ## if someone pass integer type data then validation error show 
per1=person(**new_person)
print(per1)