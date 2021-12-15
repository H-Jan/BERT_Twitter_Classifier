<h1> BERT Classifier for 2020 India Election Sentiment via Tweets </h1>

> Tweets have quickly become a representation of voice across mMany countries, and as such, a prime depository for exploring NLP’s and their potential. The following is a notebook which explores the sentiment of tweets during the 2020 Indian election, and, using the BERT or Bidirectional Encoder Representations from Transformers model developed by Google, classifies tweets into one of three categories. 

<h2> Required Imports </h2>
These include:

- Torch
  >An open-source machine learning library which has necesssary algorithms for deep learning we will use.

- Tqdm
  >A simple visual aid to show progress bar of operations

- Pandas
  >A software library used for data manipulation and analysis of tabulated datasets

- Numpy
  >A library to support large arrays and functions, which will be required in analyzing our dataset

- MatplotLib
  >A plotting library to visualize certain features

- Sklearn
  >A Machine Learning library with classification algorithms among others.




<h2> Data Analysis and Exploration </h2>
Local loading in of the dataset in the form of a csv file. For access to the dataset, it can be found [here](https://www.kaggle.com/praveengovi/bert-twitter-sentiment-classifier/data). 
From there, we note:

  - The sum of null values in the dataset
  - The size and shape of the dataset
  
We also note that the dataset totals over 162,980 tweets, and is composed with two columns, one containing the tweet, and the other it's category

A peak into the data
<img width="560" alt="Peep_Into_Data" src="https://user-images.githubusercontent.com/69823896/146152464-6ea9abbf-0fc1-46c3-b1fc-3c036b6c2fc4.png">


<h2> Distribution of the dataset </h2>
Understanding how the dataset is distributed across possible categories. Here we can note:

  - 72,250 positive tweets
  - 55,213 neutral tweets
  - 35,510 negative tweets


<h2> Data Cleaning </h2>
We will clean our data to ignore null values, seen as "nan"


<h2> Target Encoding </h2>
Implementation of Target Encoding to convert our tweets into numbers for the computer to understand. 
Target encoding doesn't add to the dimensionality of the dataset, and benefits well here.
<img width="307" alt="Target_Encoding" src="https://user-images.githubusercontent.com/69823896/146149957-b605c9be-33b0-45d9-a809-912414e25723.png">



<h2> Data Preparation for BERT Modeling </h2>
First, we create a label for the sentence list, and then check the distribution of the data based on the labels.
As we can note, the distribution is nearly identical to our dataset above, with a few missing values 
(most likely nan values). We then set the maximum length of any sequence to 280 (for 280 characters in a tweet)
and import our BERT tokenizer to convert our text into tokens corresponding to the BERT Library.


**Setting the parameter** for our input after encoded with the tokenizer. The parameters are:
  * add_special_tokens
  * Use of a special classification token found in BERT.
  * Setting of max_length
  * Limiting the size of each input to our aforementioned tweet length truncation
  * Simple shortening/cutting


**Applying simple labels** to our sentences before and after tokenization.
Next, we apply an attention mask, which is used while batching sequences together by indicating to the model which tokens should and should not be attended to in order to ensure they do not violate our maximum length.
Will determine if the first sequence needs to be padded up to length, or the second truncated down
We create a mask of 1 for all input tokens and 0 for all padding tokens.
<img width="1054" alt="Data_Prep_Before_Tokenization" src="https://user-images.githubusercontent.com/69823896/146150062-0a84bb9e-82ab-49e4-b86a-e0eb236c0c19.png">


**Converting our data into torch tensors** which is the required data type for our model. We as well specify batch size while training, and define our iterator using torch DataLoader. Use of torch DataLoader is simply to help on memory during training since this prevents loading in our entire dataset into memory.
<img width="995" alt="Trained_Data" src="https://user-images.githubusercontent.com/69823896/146150277-ece161c4-63c5-41cb-8464-a09ad490e3fb.png">


<h2> Loading BERT for Sequence Classification </h2>
We load BERT for Sequence Classifiation, specifically using a pretrained BERT model with a single linear classification layer on top, with 3 total labels (positive, neutral, and negative). We follow this with basic tuning of parameters, such as learning rate, epochs, and AdamW with epsilon, a variation of the Adam optimizer.

<h2> Training </h2>
Now we train our BERT. First we begin by making empty arrays to store our loss and accuracy for plotting.
Next, we set up a function to calculate the loss for each epoch in our range using tnrange (*like use of range, but has a tqdm wrapper*) Afterwards, we set our model to train and unpack the inputs from our dataloader and pass it forward and backwards (*hence the B. in BERT*)

Then, we update the parameters and take a step using the computed gradient and update the learning rate schedule using a scheduler. A scheduler functions by taking the epoch index and current learning rate as inputs to return a new learning rate. In the same step, we also update the tracking variables and calculate the average loss over the training data and store the current learning rate.

<h2> Problem Encountered </h2> 
<img width="1425" alt="Problem_1" src="https://user-images.githubusercontent.com/69823896/146150722-72c6bb49-6657-4729-b2c8-bfcc15c10a4c.png">
A chief problem encountered was the computational power necessary to train our BERT. Using cloud resources with 30 Gb of ram and 8 CPU's proved too much for the program, which eventually timed out during the session. The results were only attainable running on GPU from a much more powerful gaming computer that was luckily on hand.

***Model Evaluation*** 
<img width="807" alt="Screen Shot 2021-12-15 at 5 57 29 AM" src="https://user-images.githubusercontent.com/69823896/146147423-f4e93934-f86c-4c78-be9c-c798374444f3.png">
In this section of the notebook, we set metrics to evaluate our loss on the validation set, evaluating the data for each epoch. As each batch is working, we unpack the inputs from our data model and tell the model not to compute or store gradients in order to save memory and encourage validation speed 
(This is a computationally heavy step, and may require heavy hardware to compute). This is followed up by the forward pass and logit predictions, which contain the 
probabilistic output in our final layer, and evaluate our accuracy, both traditional and MCC (our Matthews Correlation Coefficient. It produces a high score only if 
the prediction obtained good results for true positives, false negatives, true negatives, and false positives).

<h2> Analysis </h2>
Here we plot a confusion matrix using blue coloring. Included is a function that normalizes our confusion matrix, though normalization does not have a large effect on our data analysis.
Lastly, we add labels to convey the emotional sentiment alongwith our confusion matrix to view how our model performed.
<img width="754" alt="Classification_Report" src="https://user-images.githubusercontent.com/69823896/146150427-c3d5b4a3-199d-43a6-8685-6d54a65a146a.png">



<h2> Conclusion </h2>
In conclusion, our model performed exceedingly well, though it most likely resulted in overfitment. The overall precision (A measure of positively predicted values), recall (or sensitivity), and f1-score all equated to 1.000. In fact, every metric equated to 1, indicating that the BERT was either fantastic in it's classification or it overfitted. 

<h2> Resources </h2>

Coursera Course on BERT, found [here](https://www.coursera.org/lecture/attention-models-in-nlp/bidirectional-encoder-representations-from-transformers-bert-lZX7F)

Medium Articles on BERT, found [here](https://medium.com/swlh/bert-pre-training-of-transformers-for-language-understanding-5214fba4a9af)

Github Articles, found
  - [here](https://github.com/google-research/bert)
  - [here](https://github.com/GU-DataLab/stance-detection-KE-MLM)
  - [here](https://jalammar.github.io/illustrated-transformer/)
  - [here](https://jalammar.github.io/illustrated-bert/)

