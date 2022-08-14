import glob
import json
import time

time.sleep(30)

print("start")
path=r".cuckoo/storage/analyses/*"
folders=[]
indix=[]
off=0

for x in glob.glob(path):
	if "latest" not in x:
		z=int(x.split("/")[3])
		indix.append(z-1+off)
		folders.append(x+"/analysis.log")
print(indix)
print("next")

data=open("temp.txt","w")

for i,fil in enumerate(folders):
	try:
		temp=open(fil,"r")
		fname=""
		data.write("[")
		while True:
			line=temp.readline()
			if not line:
				data.write(f"{indix[i]}]\n")
				break
					
			if "result_log_info_hit" in line:
				id1=line.find("result_log_info_hit")+len("result_log_info_hit")+1
				text=line[id1:]
				text=text.replace("L","")
				data.write(text+",")
	except:
		print(fil)
		continue
	data.write("\n")
data.close()

