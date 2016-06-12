import urllib2 
from multiprocessing.dummy import Pool as ThreadPool 

urls = [
	[1,1,1],
	[4,5,3],
	[1,1,1]
  ]

# Make the Pool of workers
pool = ThreadPool(4) 

# Open the urls in their own threads
# and return the results
results = pool.map(sum, urls)

print sum(results)
#close the pool and wait for the work to finish 
pool.close() 
pool.join() 