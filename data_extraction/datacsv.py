import pandas as pd
files=open("temp.txt","r")

df=pd.DataFrame(columns=["vector","max_pid","total_pro","cpu_user","cpu_sys","memory","swap","tx_packets","rx_packets","tx_bytes","rx_bytes","malware"])

temp=[]
while True:
	line=files.readline()
	if not line:
		break
	try:	
		if line=="\n":			
			t=temp[-1][0]
			temp=temp[:-1]
			z=0
			for x in temp:
				x.append(t)
				x.insert(0,z)
				z+=1
				df.loc[len(df.index)]=x
			temp=[]
			continue

		line=line.replace("[","")
		line=line.replace("]","")	
		line=line.strip()
		line=line.replace(",","").split()
		line1=list(map(lambda x:float(x),line))
		temp.append(line1)
	except:
		print(line)
		pass
		
df.to_csv("test_dataset.csv",index=False)
print(df.head())
