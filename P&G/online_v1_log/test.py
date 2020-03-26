# -*- coding = utf-8 -*-
'''
@Time: 2019/5/21 20:18
@Author: Ye Jinyu
'''
import sys
import os
from datetime import datetime,date
import pandas as pd
import numpy as np
from random import randint
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



for i in range(0,68):
    print(str('150401'+str(randint(1970,2000))+'0'+str(randint(1,9))+
              '0'+str(randint(1,9))+str(randint(1000,9999))))
    print(str('150401' + str(randint(1970, 2000)) + str(randint(10, 12)) +
               str(randint(10, 30)) + str(randint(1000, 9999))))