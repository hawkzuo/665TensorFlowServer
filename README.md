# CONTENT-BASED DJANGO SERVER
## CSCE665 Advanced Network Security Mini-Research Project
## AUTHORS: Jianyu Zuo; Tsao Yuan Chang
### WHAT THIS IS -- A BRIEF INTRODUCTION
This is a Django-based server which is designed to communicate with our chrome extension product. 
It contains a pre-trained machine learning model that can give predictions on spam with the given corpus. 
Currently it supports only 2 kinds of spams, one for the email spam and the other for Youtube/Facebook comment spam.
The model reached 99.9% training accuracy and 99.6% dev-test accuracy and 97% accuracy when tested on previous spam examples.

### A FEW WORDS ABOUT DEPLOYMENT
We'll finally push our system onto Heroku domain once we finished our chrome extension registration.
The Heroku domain is designed to only perform content-based prediction without saving the information locally.
However, if you do not trust us, you can use the source code version and build a localhost server for the predicting purposes.
We'll add the options on our chrome extension in the real future work.
Meanwhile, we're currently working on Docker settings to simplify the installation processes.
For now, the installation requires some technical backgrounds

### INSTALLATION

```Bash
pip install --ignore-installed --upgrade package-list.txt   
python -m nltk.downloader all
cd src/server/mysite
python manage.py runserver locoalhost:port
```

### INDIVIDUALIZATION
The LSTM model will be loaded by default, since it has the best performance.
Nonetheless, you could choose other models like Logistic Regression and 3-layer Neural Network Model.
Those models are all pre-trained for you, but you can even develop your own models to our system!
For example, if you want to replace our email model with your own, you can put your model predict script into folder
```Bash
/src/server/mysite/spam/model/gmail
```
And also add other folders/files needed to this path.
Then you can change the predicting service used for the server by the file at:
```Bash
/src/server/mysite/spam/views.py
```

### EXPLANATION 
The model we trained is based on all the spam emails collected from
```html
<a href="http://untroubled.org/spam/">Untroubled</a>
```
It contains spam emails from year 1998 to year 2018, with total up to 10M mails
Currently we've used Year 2018 emails as our spam examples to train our model, and use rest of spam emails to test

For all the emails, we first do a filtering process. We select emails meet our criterion showed below:
* UTF-8 and acceptable English Charsets
* Content Length < 50 examples are filtered 

The reason for 1st rule is that we believe blending different charset characters into training will impair the model performance,
instead of combining, the model can be trained separately. Another reason is that the product need for this extension. 
Most of the users are conceivably in the US.
The reason for 2nd rule is that we believe too short emails may be too rare in our daily use.
Almost all of the mails we receive today exceed this length. 
Since the dataset is from a 3rd party, we have to make sure they are not weird examples in life.

For the email parsing, a high-level idea is that we tried to make the content more meaningful.
That is, we carefully implemented our data parser parsing email contents(plain/html) to best representing useful data. 
For example, we gracefully removed the URLs in raw emails. We removed stopwords as well. 
 
For feature extraction, we finally adopted word-level N-gram features with URL count in email.
However, there're multiple candidate features that we tried as well.
* Character-level N-gram: 
The benefit of this feature is helpful on not so-well tokenized corpus,
however, we carefully tokenize our dataset to overcome this. 
The overhead of this feature is less meaningful with respect to the corpus contents.
* Top Rank Words / Most frequent words:
Those words may seem useful in the sense that they are frequently seen. 
However, it might actually both popular in spams and hams, 
let alone our feature extraction can select the word out if it only popular on one label.
* Emotional Words Lexicon / Hot trends on words
Those words may be indirectly related to the spams. We may consider add those into our system in the future.

### EVALUATION
#### Configurations
There are several configurations on the training process
 
 
#### Results


#### Conclusions


### ACKNOWLEDGEMENT
We give our special thanks to those helpful references. 
Some of the references used for this project is listed below:









