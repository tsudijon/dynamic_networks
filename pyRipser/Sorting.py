import random

def mergesort(x):
	if len(x) == 1: return x
	else:
		return lmerge(mergesort(x[0:len(x)//2]),mergesort(x[len(x)//2:len(x)]))

def lmerge(x,y):
	if len(x) == 0: return y
	if len(y) == 0: return x
	i,j = 0,0
	sorted_array = []
	while not (i == len(x)  and j == len(y) ):
		if j == len(y):
			sorted_array.append(x[i])
			i = i+1
		elif i == len(x):
			sorted_array.append(y[j])
			j = j+1
		else:
			if x[i] >= y[j]:
				sorted_array.append(y[j])
				j = j+1
			else:
				sorted_array.append(x[i])
				i = i+1
	return sorted_array
'''
start = time.time()
mergesort(y)
end = time.time()
'''

		'''
		if x[i] >= y[j]:
			if j == len(y):
				sorted_array.append(x[i])
				print(sorted_array)
				i = i+1
			else:
				while(x[i] >= y[j]):
					sorted_array.append(y[j])
					print(sorted_array)
					j = j+1
		else:
			if i == len(x):
				sorted_array.append(y[j])
				print(sorted_array)
				j = j+1
			else:
				while(x[i] < y[j]):
					sorted_array.append(x[i])
					print(sorted_array)
					i = i+1
		'''
def quicksort(x):
	if len(x) == 0: return []
	#randomly pick sort point
	split = random.choice(x)
	smaller = []
	equal = []
	larger = []
	for i in x:
		if i > split: larger.append(i)
		elif i < split: smaller.append(i)
		else: equal.append(i)

	return quicksort(smaller) + equal + quicksort(larger)
