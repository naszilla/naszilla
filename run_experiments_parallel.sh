
param_str=fifty_epochs
experiment_name=bananas

# set all instance names and zones
instances=(bananas-t4-1-vm bananas-t4-2-vm bananas-t4-3-vm bananas-t4-4-vm \
	bananas-t4-5-vm bananas-t4-6-vm bananas-t4-7-vm bananas-t4-8-vm \
	bananas-t4-9-vm bananas-t4-10-vm)

zones=(us-west1-b us-west1-b us-west1-b us-west1-b us-west1-b us-west1-b \
	us-west1-b us-west1-b us-west1-b us-west1-b)

# set parameters based on the param string
if [ $param_str = test ]; then
	start_iteration=0
	end_iteration=1
	k=10
	untrained_filename=untrained_spec
	trained_filename=trained_spec
	epochs=1
fi
if [ $param_str = fifty_epochs ]; then
	start_iteration=0
	end_iteration=9
	k=10
	untrained_filename=untrained_spec
	trained_filename=trained_spec
	epochs=50
fi

# start bananas
for i in $(seq $start_iteration $end_iteration)
do 
	let start=$i*$k
	let end=($i+1)*$k-1

	# train the neural net
	# input: all pickle files with index from 0 to i*k-1
	# output: k pickle files for the architectures to train next (indices i*k to (i+1)*k-1)
	echo about to run meta neural network in iteration $i
	python3 metann_runner.py --experiment_name $experiment_name --params $nas_params --k $k \
		--untrained_filename $untrained_filename --trained_filename $trained_filename --query $start
	echo outputted architectures to train in iteration $i

	# train the k architectures
	let max_j=$k-1
	for j in $(seq 0 $max_j )
	do
		let query=$i*$k+$j
		instance=${instances[$j]}
		zone=${zones[$j]}
		untrained_filepath=$experiment_name/$untrained_filename\_$query.pkl
		trained_filepath=$experiment_name/$trained_filename\_$query.pkl

		echo about to copy file $untrained_filepath to instance $instance
		gcloud compute scp $untrained_filepath $instance:~/naszilla/$experiment_name/ --zone $zone

		echo about to ssh into instance $instance
		gcloud compute ssh $instance --zone $zone --command="cd naszilla; \ 
		python3 train_arch_runner.py --untrained_filepath $untrained_filepath \
		--trained_filepath $trained_filepath --epochs $epochs" &
	done
	wait
	echo all architectures trained in iteration $i

	# copy results of trained architectures to the master CPU
	let max_j=$k-1
	for j in $(seq 0 $max_j )
	do
		let query=$i*$k+$j
		instance=${instances[$j]}
		zone=${zones[$j]}
		trained_filepath=$experiment_name/$trained_filename\_$query.pkl
		gcloud compute scp $instance:~/naszilla/$trained_filepath $experiment_name --zone $zone
	done
	echo finished iteration $i
done

