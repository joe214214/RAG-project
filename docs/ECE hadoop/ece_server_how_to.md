
Running the starter code
7. The ground truth (output of matrix multiplication) is computed using NumPy.
8. If you want to compare the output of your solution with the ground truth using 
the diff command, use the “-b” option to ignore white space change: “diff -b 
file_1, file_2“.
9. The “--host” option of “mpirun” sets the number of “slots” on a compute node 
which indicates how many processes can potentially execute on that node. For 
example, “--host $HOSTANME:16” means up to 16 MPI processes are allowed on 
the node. The number of slots should not be significantly larger than the number 
of CPU-cores available on a node.
ECE 454 - University of Waterloo 10
Interpreting output of the starter code▪ “output/result.txt” contains output of all test cases
Example: 
exptype, numproc, mtxorder, computetime, totaltime, correct
t, 16, 256, 0.0630541, 0.152243, 1
t, 16, 512, 0.348331, 0.612397, 1
t, 16, 1024, 7.59186, 8.63896, 1
s, 4, 1024, 7.59362, 8.64095, 1
s, 16, 1024, 7.60069, 8.64888, 1
• exptype: t - time-to-solution experiment, s - strong scaling experiment
• numproc: number of MPI processes used 
• mtxorder: order of input matrix (input size) 
• computetime: totaltime - the fraction of time spent in writing output to file 
• output correct: 1, output incorrect: 0 (determined by comparing with the ground truth)
ECE 454 - University of Waterloo 11
Standalone vs. multi-node experiments
▪The “buildrun_standalone.sh” script is for running experiment on a single host (with 
multiple CPUs/cores, e.g., ecetesla hosts have up to 24 CPUs).
▪The same MPI application can be run on both a single host and multiple interconnected 
hosts, no code recompilation is required.
▪Many parallel applications benefit from access to a large number of distributed 
processes. At the same time, performance of distributed processing is heavily influenced 
by the capacity of the interconnect (network) of a distributed platform.
▪We set up MPI in the ecehadoop cluster. Each ecehadoop node, however, only has a few 
cores, so we are not expecting any significant gain compared to running on a single host, 
e.g., on ecetesla using 16 processes. 
▪A second “buildrun” script for running your code on ecehaddop is available. It has the 
same test cases (as in “buildrun_standalone.sh”), however, runs in a multi-node setting.
▪The only modification in the second “builrun” script is a list of hosts, on which an an MPI 
application is run, as an argument to the “mpirun” command.
ECE 454 - University of Waterloo 12
Multi-node experiments
▪Log in to “ecehadoop”.
▪Run the following script: “sh copy_ssh_public_key.sh”. (Run this script only once.)
▪Then use the following script to run an MPI application on the ecehadoop 
cluster: “sh buildrun_multinode.sh “. The “mpi_ecehadoop_hosts” is the 
“hostfile” for the “mpirun” command.
▪Note: If you notice the MPI application is not progressing, make sure you can ssh 
to the hosts listed in the file “ecehadoop_hosts” without entering password.
▪Note: The following warning message can be ignored; it does not interfere with 
application execution or output. “Authorization required, but no authorization 
protocol specified”. (ECE sysadmins do not know how to fix this yet.)
ECE 454 - University of Waterloo 13
Apache Spark
▪We will create accounts on eceHadoop for everyone in the class.
▪The following set of slides describe how to run a Spark application on 
ecehadooop cluster.
ECE 454 - University of Waterloo 14
Using the starter code
▪The  starter  code  includes  an  implementation  of  word  counting  in  Spark.    Shell 
scripts are provided to build and run the code.
ECE 454 - University of Waterloo 15
Running the starter code
Please follow the steps below to run the sample Spark (scala) word counting 
program (available in the starter code tarball):
ssh userID@eceTerm3.uwaterloo.ca (userID is your ECE alphanumeric username)
ssh userID@eceHadoop.private.uwaterloo.ca // this is the eceHadoop master 
node (it runs the Spark driver program)
Make sure the following command completes without any error:
/opt/hadoop-latest/hadoop/bin/hdfs dfs -ls "/user/userID/" // your HDFS home 
directory
Assuming you copied the a2_starter code in your ECE home directory, 
i.e.,  /home/userID/a2_starter, cd into the a2_starter directory:
cd /home/userID/a2_starter
Now build/run the word counting program:
sh buildrun_wc_spark_cluster.sh
16
Running the starter code
The program will copy input data from a2_starter/sample_input to the HDFS 
directory /user/userID/
Upon successful completion of the Spark program, the output will be written to 
the HDFS directory /user/userID/a2_starter_code_output_spark/ 
Run the following command to view the output written to HDFS:
/opt/hadoop-latest/hadoop/bin/hdfs dfs -cat 
"/user/userID/a2_starter_code_output_spark/*"
Please do not put your application code/scripts in your HDFS directory, i.e., 
/user/userID/. This directory is only for input and output of a Spark program.
17
Cluster house rules
▪Storage space on ecehadoop cluster is limited, so please keep your HDFS usage 
(/user/userID/) below 1GB.
▪If you want to experiment with large inputs then consider placing them in HDFS 
under /tmp where they can be accessed by group members and avoid 
duplication.
▪Please limit yourself to running one Spark job at a time, especially during busy 
periods such as shortly before the deadline.
▪Please do not let any job run for more than 30 minutes on the cluster. 
ECE 454 - University of Waterloo 18
External computing resources
▪AMD University Program AI & HPC Cluster
▪Microsoft Azure for Students
▪AWS
▪Google Colab
▪IBM Cloud
▪Oracle Cloud
▪Sharcnet Canada
ECE 454 - University of Waterloo 19
