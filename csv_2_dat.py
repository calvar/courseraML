
In = open("Call.csv","r")
Out = open("Call.dat","w")
for line in In:
    v = line.split(',')
    for e in v:
        Out.write("{0} ".format(e))
In.close()
Out.close()
