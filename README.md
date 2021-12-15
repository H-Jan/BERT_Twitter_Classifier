<h1> BERT Classifier for 2020 India Election Sentiment via Tweets </h1>

> Tweets have quickly become a representation of voice across many countries, and as such, a prime depository for exploring NLP’s and their potential. The following is a notebook which explores the sentiment of tweets during the 2020 Indian election, and, using the BERT or Bidirectional Encoder Representations from Transformers model developed by Google, classifies tweets into one of three categories. 

<h2> **Required Imports** </h2>

These include:

Torch

An open-source machine learning library which has necesssary algorithms for deep learning we will use.
Tqdm

A simple visual aid to show progress bar of operations
Pandas

A software library used for data manipulation and analysis of tabulated datasets
Numpy

A library to support large arrays and functions, which will be required in analyzing our dataset
MatplotLib

A plotting library to visualize certain features
Sklearn

A Machine Learning library with classification algorithms among others.



<h2> **Data Analysis and Exploration** </h2>
Local loading in of the dataset in the form of a csv file. From there, we note:

The sum of null values in the dataset

The size and shape of the dataset

We note that the dataset totals over 162,980 tweets, and is composed with two columns, one containing the tweet, and the other it's category
A peak into the data

Understanding how the data is formatted




<h3> **Distribution of the dataset** </h3>
Understanding how the dataset is distributed across possible categories. Here we can note:

72,250 positive tweets
55,213 neutral tweets
35,510 negative tweets





<h3> **Data Cleaning** </h3>
We will clean our data to ignore null values, seen as "nan"





<h3> **Target Encoding** </h3>
Implementation of Target Encoding to convert our tweets into numbers for the computer to understand. 
Target encoding doesn't add to the diemnsionality of the dataset, and benefits well here.








<h3> **Data Preparation for BERT Modeling*** </h3>
First, we create a label for the sentence list, and then check the distribution of the data based on the labels.
As we can note, the distribution is nearly identical to our dataset above, with a few missing values 
(most likely nan values). We then set the maximum length of any sequence to 280 (for 280 characters in a tweet)
and import our BERT tokenizer to convert our text into tokens corresponding to the BERT Library.






**Setting the parameter** for our input after encoded with the tokenizer. The parameters are:

add_special_tokens

Use of a special classification token found in BERT.
max_length

Limiting the size of each input to our aforementioned tweet length
truncation

Simple shortening/cutting






**Applying simple labels** to our sentences before and after tokenization.

Next, we apply an attention mask, which is used while batching sequences together by indicating to the model which tokens should and should not be attended to in order to ensure they do not violate our maximum length.

Will determine if the first sequence needs to be padded up to length, or the second truncated down
We create a mask of 1 for all input tokens and 0 for all padding tokens.






**Converting our data into torch tensors** which is the required data type for our model. We as well specify batch size while training, and define our iterator using torch DataLoader.

Use of torch DataLoader is simply to help on memory during training since this prevents loading in our entire dataset into memory.







** <h3> Loading BERT for Sequence Classification</h3>**
We load BERT for Sequence Classifiation, specifically using a pretrained BERT model with a single linear classification layer on top, with 3 total labels (positive, neutral, and negative). We follow this with basic tuning of parameters, such as learning rate, epochs, and AdamW with epsilon, a variation of the Adam optimizer.








**###Training**
Now we train our BERT. First we begin by making empty arrays to store our loss and accuracy for plotting.

Next, we set up a function to calculate the loss for each epoch in our range using tnrange (like use of range, but has a tqdm wrapper)

Afterwards, we set our model to train and unpack the inputs from our dataloader and pass it forward and backwards (hence the B. in BERT)

Then, we update the parameters and take a step using the computed gradient and update the learning rate schedule using a scheduler.

A scheduler functions by taking the epoch index and current learning rate as inputs to return a new learning rate.
In the same step, we also update the tracking variables and calculate the average loss over the training data and store the current learning rate.

***Model Evaluation*** In this section of the notebook, we set metrics to evaluate our loss on the validation set, evaluating the data for each epoch. As each batch 
is working, we unpack the inputs from our data model and tell the model not to compute or store gradients in order to save memory and encourage validation speed 
(This is a computationally heavy step, and may require heavy hardware to compute). This is followed up by the forward pass and logit predictions, which contain the 
probabilistic output in our final layer, and evaluate our accuracy, both traditional and MCC (our Matthews Correlation Coefficient. It produces a high score only if 
the prediction obtained good results for true positives, false negatives, true negatives, and false positives).






**###Analysis**
Here we plot a confusion matrix using blue coloring. Included is a function that normalizes our confusion matrix, though normalization does not have a large effect on our data analysis.
Lastly, we add labels to convey the emotional sentiment alongwith our confusion matrix to view how our model performed.
