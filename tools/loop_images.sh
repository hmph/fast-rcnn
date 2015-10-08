 #!/bin/bash 
for (( c=1; c<=5; c++ ))
do
   echo "Welcome $c times"
done

for (( i = 0 ; i < 2914; i=i+1 )) 
do
	python output_detections.py --i=${i}
done
