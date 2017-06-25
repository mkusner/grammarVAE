#!/bin/bash

touch /tmp/testRMSEgrammar.txt
rm /tmp/testRMSEgrammar.txt
touch /tmp/testRMSEcharacter.txt
rm /tmp/testRMSEcharacter.txt
simulations=1
for i in `seq $simulations`
do
    echo $i
    cat simulation$i/grammar/nohup.out | grep RMSE | head -1 | cut -d " " -f 4 >> /tmp/testRMSEgrammar.txt
    cat simulation$i/character/nohup.out | grep RMSE | head -1 | cut -d " " -f 4 >> /tmp/testRMSEcharacter.txt
done

mean_RMSE_character=`python -c "import numpy as np; print(np.mean(np.loadtxt('/tmp/testRMSEcharacter.txt')))"`
mean_RMSE_grammar=`python -c "import numpy as np; print(np.mean(np.loadtxt('/tmp/testRMSEgrammar.txt')))"`

std_RMSE_character=`python -c "import numpy as np; print(np.std(np.loadtxt('/tmp/testRMSEcharacter.txt')) / np.sqrt(10))"`
std_RMSE_grammar=`python -c "import numpy as np; print(np.std(np.loadtxt('/tmp/testRMSEgrammar.txt')) / np.sqrt(10))"`

echo RMSE grammar  : $mean_RMSE_grammar $std_RMSE_grammar
echo RMSE character: $mean_RMSE_character $std_RMSE_character

touch /tmp/testRMSEgrammar.txt
rm /tmp/testRMSEgrammar.txt
touch /tmp/testRMSEcharacter.txt
rm /tmp/testRMSEcharacter.txt
for i in `seq $simulations`
do
    echo $i
    cat simulation$i/grammar/nohup.out |  grep "Test ll" | grep -v erro | head -1 | cut -d " " -f 4 >> /tmp/testRMSEgrammar.txt
    cat simulation$i/character/nohup.out |grep "Test ll" | grep -v erro | head -1 | cut -d " " -f 4 >> /tmp/testRMSEcharacter.txt
done

mean_RMSE_character=`python -c "import numpy as np; print(np.mean(np.loadtxt('/tmp/testRMSEcharacter.txt')))"`
mean_RMSE_grammar=`python -c "import numpy as np; print(np.mean(np.loadtxt('/tmp/testRMSEgrammar.txt')))"`

std_RMSE_character=`python -c "import numpy as np; print(np.std(np.loadtxt('/tmp/testRMSEcharacter.txt')) / np.sqrt(10))"`
std_RMSE_grammar=`python -c "import numpy as np; print(np.std(np.loadtxt('/tmp/testRMSEgrammar.txt')) / np.sqrt(10))"`

echo LL grammar:   $mean_RMSE_grammar $std_RMSE_grammar
echo LL character: $mean_RMSE_character $std_RMSE_character

rm /tmp/testRMSEgrammar.txt
rm /tmp/testRMSEcharacter.txt
