# SVM-MapReduce
Implementation of SVM Machine Learning algorithm on MapReduce using Hadoop Framework


The project is aimed at implementing the training and testing of the SVM machine learning problem on the Hadoop framework.
Why do we use Hadoop MapReduce?
  1) The size of the dataset is large and hence we want to use the MapReduce capabilities to exploit the parallel processing power of the commodity hardware on different machines connected in a hadoop cluster. Hence, we can speed up the process of computation.
  2) Also it is not always feasible to store such large amount of data in the memory of a single PC which again requires a big data platform like HDFS. Hence, we can store the dataset in clusters of commodity hardware such that space exhaustion is not a concern.

Now, we have trained the SVM using the Hinge Loss.
The datapoints in the dataset form a binary classification task with 2 features for each datapoint.

Note:
We have in our project made changes that are specific to the number of datapoints manually in the applyGradients function in the SVM.java class. Whenever the dataset is changed i.e. for training or testing, the number of datapoints must be changed manually. For Eg)- weights[i] -= w.get(i)/22500000; line must be changed in the applyGradients class.Hence, if in our dataset only 10,000 datapoints were present then we must change it to weights[i] -= w.get(i)/10000, the same must be done for the bias value as well. Also, the paths to weights and bias files must be changed according to the path of these files in one's own HDFS path for these files.

# Usage steps:
1) Add a dataset,weights and bias text files to the HDFS.
2) Make changes to the code to point to the paths of these files in both SVM.java and Prediction.java files.
3) Make changes to the code before training as mentioned in the note above.
4) Convert the project to a jar and run.

The weights must be initialized to the dimension of the features of datapoints. Also please refer to the docx file to see how the code is implemented. Also we have uploaded small datasets for reference on the format of dataset files.
