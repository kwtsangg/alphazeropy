import timeit 
  
example = ''' 
a = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
def example(a): 
  return a.copy()
'''
  
# timeit statement 
print timeit.timeit(stmt = example, number = int(1e6)) 

example = ''' 
a = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
def example(a): 
  return [b[:] for b in a]
'''
 
# timeit statement 
print timeit.timeit(stmt = example, number = int(1e6)) 
