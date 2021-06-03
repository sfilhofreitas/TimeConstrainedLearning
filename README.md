# Installation

Download the project and place the root folder in some directory. I'll assume the location is /path/TimeConstrainedLearning.


# Dependencies

Package dependencies are listed in the file "requirements.txt" at /path/TimeConstrainedLearning/machine_teacher/requirements.txt. To install them, just open a terminal and run the command:

```bash
pip3 install -r path/TimeConstrainedLearning/machine_teacher/requirements.txt
```

# Repeat Experiments

## Downloading datasets

Download the pre-processed datasets and place them in the folder path/TimeConstrainedLearning/garage/datasets_train_test_split

Download link:
https://drive.google.com/file/d/1dWu7BQxD1LZn7AdHchbKkVh2KSV5uN7L/view?usp=sharing

## Configuration Files

All configuration files of the experiments performed are available at: /path/TimeConstrainedLearning/experiments/configs.


* Files are divided into folders. For example, the files needed to repeat the Decision Tree tests are in the folder: config_files_DecisionTree. The files of other learners are in folders identified with their names.

* There is also the config_files_SingleBatchTeacher folder that contains the files needed to get the Full results for every combination of dataset and learner.


## Log Table

At the end of the execution of each set of experiments, a folder with a table of logs .xls will be generated in /path/TimeConstrainedLearning/experiments/results. Each row in this table represents an iteration between teacher and learner in an experiment. Below we highlight information from the relevant columns to interpret the results of the experiments:

* Teacher: teacher's name. The values in this column can be 'FixedPercWrongTeacher', 'DoubleTeacher', 'WTFTeacher' and 'SingleBatchTeacher'; which correspond, respectively, to the teachers TCT, Double, OSCT and Full (full training).

* Learner: learner's name.

* Dataset: datasets's name.

* iter: iteration number.

* TS_size: teaching set size.

* dataset_accuracy: accuracy on training set.

* elapsed_time: time taken until the referred iteration.

* time_left: time left to time out (time_limit - elapsed_time).

* test_set_accuracy: accuracy on testing set.

* learner_selected: iteration of the learner that the method returns (a value from the iter column).

* accuracy_selected: accuracy (on testing set) of the learner that the method returns (a value from the test_set_accuracy column).


## Running the Experiments

At first, let's run the tests with the entire datasets or Full Training (Full) and Getting/Setting the Time Limits. For this, we will use the configuration files from the folder: config_files_SingleBatchTeacher.


From directory /path/TimeConstrainedLearning/experiments/ run the following command:

```bash
python main_from_folder.py config_files_SingleBatchTeacher
```

At the end of the execution of this command, it is possible to obtain in the logs table the time limits for each combination of learner and dataset. In our experiments we use the average over the 4 runs that will be in the table (elapsed_time column).

For each combination (L, D) we must replace the value of time_limit (in the last line of the file) in the configuration files: Double_L_D.conf, TCT_L_D.conf and OSCT_L_D.conf. More precisely, taking as an example the case where L=DecisionTree and D=mnist, in the files: Double_DecisionTreeLearner_mnist.conf, TCT_DecisionTreeLearner_mnist.conf and OSCT_DecisionTreeLearner_mnist.conf.

The other experiments can be performed (from directory /path/TimeConstrainedLearning/experiments/) as follows:

```bash
python main_from_folder.py config_files_DecisionTree
```

```bash
python main_from_folder.py config_files_RandomForest
```

```bash
python main_from_folder.py config_files_LGBM
```

```bash
python main_from_folder.py config_files_SVM
```

```bash
python main_from_folder.py config_files_LogisticRegression
```



