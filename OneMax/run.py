import subprocess as sub

for i in range(0, 101):
    turn = 0.001*i
    sub.call(f"./OneMax_Fixed.exe {turn}")