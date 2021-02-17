# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:36:13 2020

@author: asus
"""

def asal_mi(sayi):
    for i in range(2,sayi):
        if(sayi%i == 0):
            return False
        return True
        
while True:
    a = input("sayi giriniz:")
    if a =="q":
        break
    else:
        print(asal_mi(int(a)))