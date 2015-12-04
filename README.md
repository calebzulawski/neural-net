# Artificial Neural Network for ECE469
Caleb Zulawski

## Usage
This requires GCC 4.7 or later with C++11.  Other C++11 compatible compilers may work but no guarantees.

### Build from source
To build with make:
```bash
cd /path/to/neural-net
make
```

### Training
```bash
train initWeightsFile outputWeightsFile trainDataFile learnRate epochs
```

### Testing
```bash
test trainedWeightsFile testDataFile resultsOutputFile
```

## Additional Dataset
I downloaded a dataset from the UCI Machine Learning Repository.  Each sample corresponds to a person's information from the 1994 Census database.  The dataset contains various demographic information predictors.  The categories correspond to whether or not that person makes over $50,000 a year.  The original dataset can be accessed at:

```url
https://archive.ics.uci.edu/ml/datasets/Adult
```

### Predictors and Modifications
The original predictors in the dataset were age, working status, U.S. Census sample weight, education, education number, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours worked per week, and native country.

There were several things that make this data set interesting.  First, there is a mixture of categorical and continuous data.  Second, none of the data is normalized.  Finally, the database contained some information regarding the accuracy of several classifiers as tested by the authors.

The database contained some unknowns, so all of those records were removed.  In addition, the predictors for capital gains and losses were fairly sparse, so they were not included.  The following is the list of the predictors and categories, in the order they are contained in the training and testing files:

* Age, normalized by dividing by 100
* Working status, mapped to fractions between 0 and 1
* Census weight, unsure what this meant so divided by 2,000,000
* Education, mapped to fractions between 0 and 1
* Education number, normalized to fractions between 0 and 1
* Marital status, mapped to fractions between 0 and 1
* Occupation, mapped to fractions between 0 and 1
* Relationship, mapped to fractions between 0 and 1
* Race, mapped to fractions between 0 and 1,
* Sex, mapped to 0 or 1
* Hours worked per week, divided by 168 (number of hours in a week)
* Native country, mapped to fractions between 0 and 1

### Configuration

This dataset was tested on a network with 5 hidden nodes, trained with a learning rate of 0.1 over 100 epochs.  This was a reasonable balance between training wall time and performance, as this data set is much larger than the provided ones, with thousands of samples.  The starting weights were simply a truncated version of the weights supplied for the WDBC neural network. 

### Results

I was surprised to find that the performance was fairly good.  The overall accuracy was 82.5%.  This performance was not quite as good as the authors' C4.5 or Naive Bayes classifiers, which had an accuracy of nearly 86%.  The neural network did seem to outperform K nearest neighbors, however, with just under 80% accuracy.

The other metrics (precision, recall, and F1) were not nearly as high, but had no results to be compared to.