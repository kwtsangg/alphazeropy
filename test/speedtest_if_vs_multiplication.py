import timeit 
  
example = ''' 
a = 1
def example(a): 
  return -1 if a == 1 else 1
'''
  
# timeit statement 
print timeit.timeit(stmt = example, number = int(1e8)) 

example = ''' 
a = 1
def example(a): 
  return -a
'''
  
# timeit statement 
print timeit.timeit(stmt = example, number = int(1e8)) 
