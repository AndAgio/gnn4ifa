# List of logs and who should be notified of issues
topos=("small" "dfn")
convs=("gcn" "cheb" "gin" "tag" "sg")
pools=("mean" "sum" "max" "s2s" "att")
mus=(1 2 3 4 5 6 7 8)

# Look for signs of trouble in each log
for i in ${!topos[@]};
do
  for j in ${!convs[@]};
  do
    for k in ${!mus[@]};
    do
      for p in ${!pools[@]};
      do
        python train.py --model="class_${convs[$j]}_${mus[$k]}x100_${pools[$p]}" --train_scenario="existing" --train_topology="${topos[$i]}" --epochs=10 --diferential=0 --masking=0
      done
    done
  done
done