def formMergedClusters(IDList, clusterMap, validCompCluster):
	newCList = []
	index = 0
	for i in range(len(IDList)):
		if validCompCluster[i] == 1:
			newCluster = clusters[IDList[i][0]]
			n = newCluster.tcv.n
			w = [x * n for x in newCluster.tcv.wsum_v]
			for j in range(len(newCluster.tweets)):
				newCluster.tweets[j].cluster = index
			for j in range(len(newCluster.tcv.ft_set)):
				newCluster.tcv.ft_set[j].cluster = index
			for j in range(1, len(IDList[i])):
				newCluster.tcv.sum_v = [x + y for x, y in zip(newCluster.tcv.sum_v, clusters[IDList[i][j]].tcv.sum_v)]
				newCluster.tcv.wsum_v = [x + y for x, y in zip(newCluster.tcv.wsum_v, clusters[IDList[i][j]].tcv.wsum_v)]
				newCluster.tcv.ts1 = newCluster.tcv.ts1 + clusters[IDList[i][j]].tcv.ts1
				newCluster.tcv.ts2 = newCluster.tcv.ts2 + clusters[IDList[i][j]].tcv.ts2
				newCluster.tcv.n = newCluster.tcv.n + clusters[IDList[i][j]].tcv.n
				newCluster.tcv.ft_set = newCluster.tcv.ft_set + clusters[IDList[i][j]].tcv.ft_set
				n = clusters[IDList[i][j]].tcv.n
				w1 = [x * n for x in clusters[IDList[i][j]].tcv.wsum_v]
				w = [x + y for x, y in zip(w, w1)]
				newCluster.tweets = newCluster.tweets + clusters[IDList[i][j]].tweets
			newCluster.centroid = [x / newCluster.tcv.n for x in newCluster.tcv.wsum_v]
			ftSort = []
			for j in range(len(newCluster.tcv.ft_set)):
				sim = calc_sim(newCluster.tcv.ft_set[j].tv,newCluster.centroid)
				ftSort.append(ftsetSorter(newCluster.tcv.ft_set[j],sim))
			ftSort = sorted(ftSort, key=operator.attrgetter('sim'), reverse = True)
			newCluster.tcv.ft_set = ftSort[:size_ftset]
			newCList.append(newCluster)
			index = index + 1

	for i in range(len(clusterMap)):
		if clusterMap[i] == -1:
			k = len(newCList) - 1
			for j in range(len(clusters[i].tweets)):
				clusters[i].tweets[j].cluster = k
			newCList.append(clusters[i])
	return newCList

def merge():
	arr = []
	IDList = []
	clusterMap = []
	validCompCluster = []
	for i in range(len(clusters)):
		IDList[i].append([])
	for i in range(len(validCompCluster)):
		validCompCluster.append(0)
	for i in range(len(clusters)):
		clusterMap.append(-1)
	for i in range(len(clusters)):
		for j in range(len(clusters)):
			if i != j:
				sim = calc_sim(clusters[i].centroid,clusters[j].centroid)
				arr.append(ClusterSorter(i,j,sim))
	arr = sorted(arr, key=operator.attrgetter('sim'), reverse = True)
	n = len(clusters)
	lim = mc * n
	k = 0
	for i in range(len(arr)):
		if n == lim:
			break
		i1 = arr[i].i
		i2 = arr[i].j 
		if clusterMap[i1] == -1 and clusterMap[i2] == -1:
			clusterMap[i1] = k
			clusterMap[i2] = k
			IDList[k].append(i1)
			IDList[k].append(i2)
			n = n - 1
			validCompCluster[k] = 1
		elif clusterMap[i1] != -1 and clusterMap[i2] != -1:
			if clusterMap[i1] != clusterMap[i2]:
				if len(IDList[clusterMap[i1]]) <= len(IDList[clusterMap[i2]]):
					validCompCluster[clusterMap[i1]] = 0
					prevC = clusterMap[i1]
					for j in range(len(IDList[prevC])):
						clusterMap[IDList[j]] = clusterMap[i2]
					IDList[clusterMap[i2]] = IDList[clusterMap[i2]] + IDList[prevC] 
				if len(IDList[clusterMap[i1]]) > len(IDList[clusterMap[i2]]):
					validCompCluster[clusterMap[i2]] = 0
					prevC = clusterMap[i2]
					for j in range(len(IDList[prevC])):
						clusterMap[IDList[j]] = clusterMap[i1]
					IDList[clusterMap[i1]] = IDList[clusterMap[i1]] + IDList[prevC] 
		elif clusterMap[i1] != -1 and clusterMap[i2] == -1:
			clusterMap[i2] = clusterMap[i1]
			IDList[clusterMap[i1]].append(i2)
		elif clusterMap[i1] == -1 and clusterMap[i2] != -1:
			clusterMap[i1] = clusterMap[i2]
			IDList[clusterMap[i2]].append(i1) 

		
