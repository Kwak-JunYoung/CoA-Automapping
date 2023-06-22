from test import *

while(1):
    x = input("What to predict?")
    
    answer = predict(x.strip())
    print()
    print(answer)
    print()