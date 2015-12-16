class Tweet:
	dim = 0
	def __init__(self,text,tv,w,normtv,dist=-1,cluster=-1,flag=0):
		self.text = text
		self.tv = tv
		self.w = w
		self.normtv = normtv
		self.dist = dist
		self.cluster = cluster
		self.flag = flag