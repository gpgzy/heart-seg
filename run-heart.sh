#!/bin/bash
#SBATCH -J cac_heart           #作业名称
#SBATCH -p lzhgnormal            #指定队列名，网页平台"可访问队列"里面有队列名
#SBATCH -N 1                      #使用计算节点个数,跨节点一般需要调用mpi;python一般使用单节点
#SBATCH --ntasks-per-node=20     #每个计算节点使用的核心数，上限是单个计算节点的所有核心数
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -o 2CH%j.log                 #指定作业的标准输出,%j是作业ID
#SBATCH -e 2CH%j.err                 #指定作业的错误输出

#加载环境变量
#运行程序 srun --mpi=pmi2等同于mpirun -np 24,srun可以自动获取#SBATCH参数申请的核心数

python main.py