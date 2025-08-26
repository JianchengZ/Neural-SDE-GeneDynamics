python train_nSDE.py --batch_hyper 32  --repeat 1   >> results/repeat1/en_de_temp32/log_time_sde.txt
python train_nSDE_3dim.py  --batch_hyper 32  --repeat 1 >> three_dim/repeat1/en_de_temp32/log_time_sde.txt


python learn_noise.py --batch_hyper  1 --repeat 1 >> results/repeat1/en_de_temp1/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 2  --repeat 1 >> results/repeat1/en_de_temp2/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 4  --repeat 1 >> results/repeat1/en_de_temp4/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 8  --repeat 1 >> results/repeat1/en_de_temp8/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 16  --repeat 1 >> results/repeat1/en_de_temp16/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 32 --repeat 1 >> results/repeat1/en_de_temp32/log_time_predictnoise.txt

python Test_on_experimental_data.py --batch_hyper 1 --repeat 1 >> results/repeat1/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 1 --repeat 1 >> three_dim/repeat1/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 2 --repeat 1 >> results/repeat1/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 2 --repeat 1 >> three_dim/repeat1/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 4 --repeat 1 >> results/repeat1/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 4 --repeat 1 >> three_dim/repeat1/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 8 --repeat 1 >> results/repeat1/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 8 --repeat 1 >> three_dim/repeat1/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 16 --repeat 1 >> results/repeat1/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 16 --repeat 1 >> three_dim/repeat1/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 32 --repeat 1 >> results/repeat1/en_de_temp32/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 32 --repeat 1 >> three_dim/repeat1/en_de_temp32/log_time_predictexp.txt


python train_nSDE.py --batch_hyper 32  --repeat 2   >> results/repeat2/en_de_temp32/log_time_sde.txt
python train_nSDE_3dim.py  --batch_hyper 32  --repeat 2 >> three_dim/repeat2/en_de_temp32/log_time_sde.txt


python learn_noise.py --batch_hyper  1 --repeat 2 >> results/repeat2/en_de_temp1/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 2  --repeat 2 >> results/repeat2/en_de_temp2/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 4  --repeat 2 >> results/repeat2/en_de_temp4/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 8  --repeat 2 >> results/repeat2/en_de_temp8/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 16  --repeat 2 >> results/repeat2/en_de_temp16/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 32 --repeat 2 >> results/repeat2/en_de_temp32/log_time_predictnoise.txt

python Test_on_experimental_data.py --batch_hyper 1 --repeat 2 >> results/repeat2/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 1 --repeat 2 >> three_dim/repeat2/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 2 --repeat 2 >> results/repeat2/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 2 --repeat 2 >> three_dim/repeat2/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 4 --repeat 2 >> results/repeat2/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 4 --repeat 2 >> three_dim/repeat2/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 8 --repeat 2 >> results/repeat2/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 8 --repeat 2 >> three_dim/repeat2/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 16 --repeat 2 >> results/repeat2/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 16 --repeat 2 >> three_dim/repeat2/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 32 --repeat 2 >> results/repeat2/en_de_temp32/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 32 --repeat 2 >> three_dim/repeat2/en_de_temp32/log_time_predictexp.txt


python train_nSDE.py --batch_hyper 32  --repeat 3   >> results/repeat3/en_de_temp32/log_time_sde.txt
python train_nSDE_3dim.py  --batch_hyper 32  --repeat 3 >> three_dim/repeat3/en_de_temp32/log_time_sde.txt


python learn_noise.py --batch_hyper  1 --repeat 3 >> results/repeat3/en_de_temp1/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 2  --repeat 3 >> results/repeat3/en_de_temp2/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 4  --repeat 3 >> results/repeat3/en_de_temp4/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 8  --repeat 3 >> results/repeat3/en_de_temp8/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 16  --repeat 3 >> results/repeat3/en_de_temp16/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 32 --repeat 3 >> results/repeat3/en_de_temp32/log_time_predictnoise.txt

python Test_on_experimental_data.py --batch_hyper 1 --repeat 3 >> results/repeat3/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 1 --repeat 3 >> three_dim/repeat3/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 2 --repeat 3 >> results/repeat3/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 2 --repeat 3 >> three_dim/repeat3/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 4 --repeat 3 >> results/repeat3/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 4 --repeat 3 >> three_dim/repeat3/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 8 --repeat 3 >> results/repeat3/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 8 --repeat 3 >> three_dim/repeat3/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 16 --repeat 3 >> results/repeat3/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 16 --repeat 3 >> three_dim/repeat3/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 32 --repeat 3 >> results/repeat3/en_de_temp32/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 32 --repeat 3 >> three_dim/repeat3/en_de_temp32/log_time_predictexp.txt

python train_nSDE.py --batch_hyper 32  --repeat 4   >> results/repeat4/en_de_temp32/log_time_sde.txt
python train_nSDE_3dim.py  --batch_hyper 32  --repeat 4 >> three_dim/repeat4/en_de_temp32/log_time_sde.txt


python learn_noise.py --batch_hyper  1 --repeat 4 >> results/repeat4/en_de_temp1/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 2  --repeat 4 >> results/repeat4/en_de_temp2/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 4  --repeat 4 >> results/repeat4/en_de_temp4/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 8  --repeat 4 >> results/repeat4/en_de_temp8/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 16  --repeat 4 >> results/repeat4/en_de_temp16/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 32 --repeat 4 >> results/repeat4/en_de_temp32/log_time_predictnoise.txt

python Test_on_experimental_data.py --batch_hyper 1 --repeat 4 >> results/repeat4/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 1 --repeat 4 >> three_dim/repeat4/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 2 --repeat 4 >> results/repeat4/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 2 --repeat 4 >> three_dim/repeat4/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 4 --repeat 4 >> results/repeat4/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 4 --repeat 4 >> three_dim/repeat4/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 8 --repeat 4 >> results/repeat4/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 8 --repeat 4 >> three_dim/repeat4/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 16 --repeat 4 >> results/repeat4/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 16 --repeat 4 >> three_dim/repeat4/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 32 --repeat 4 >> results/repeat4/en_de_temp32/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 32 --repeat 4 >> three_dim/repeat4/en_de_temp32/log_time_predictexp.txt


python train_nSDE.py --batch_hyper 32  --repeat 5   >> results/repeat5/en_de_temp32/log_time_sde.txt
python train_nSDE_3dim.py  --batch_hyper 32  --repeat 5 >> three_dim/repeat5/en_de_temp32/log_time_sde.txt


python learn_noise.py --batch_hyper  1 --repeat 5 >> results/repeat5/en_de_temp1/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 2  --repeat 5 >> results/repeat5/en_de_temp2/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 4  --repeat 5 >> results/repeat5/en_de_temp4/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 8  --repeat 5 >> results/repeat5/en_de_temp8/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 16  --repeat 5 >> results/repeat5/en_de_temp16/log_time_predictnoise.txt
python learn_noise.py --batch_hyper 32 --repeat 5 >> results/repeat5/en_de_temp32/log_time_predictnoise.txt

python Test_on_experimental_data.py --batch_hyper 1 --repeat 5 >> results/repeat5/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 1 --repeat 5 >> three_dim/repeat5/en_de_temp1/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 2 --repeat 5 >> results/repeat5/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 2 --repeat 5 >> three_dim/repeat5/en_de_temp2/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 4 --repeat 5 >> results/repeat5/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 4 --repeat 5 >> three_dim/repeat5/en_de_temp4/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 8 --repeat 5 >> results/repeat5/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 8 --repeat 5 >> three_dim/repeat5/en_de_temp8/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 16 --repeat 5 >> results/repeat5/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 16 --repeat 5 >> three_dim/repeat5/en_de_temp16/log_time_predictexp.txt
python Test_on_experimental_data.py --batch_hyper 32 --repeat 5 >> results/repeat5/en_de_temp32/log_time_predictexp.txt
python Test_on_experimental_data_3dim.py  --batch_hyper 32 --repeat 5 >> three_dim/repeat5/en_de_temp32/log_time_predictexp.txt
