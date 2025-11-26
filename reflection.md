# Reflection

I used Python to solve this.

First, I had to find the hash of my ID. I just used the hashlib library for that. Once I had the hash, I used pandas to read the csv files and find the review that contained that hash. That gave me the book title and the fake review.

For the machine learning part, I used scikit-learn and SHAP. I wasn't totally sure how to label the data, but the instructions said suspicious reviews are short and have superlatives, so I wrote a function to check for that. Then I trained a logistic regression model.

The SHAP part was interesting. I used it to see which words were most important for the "genuine" class. It turned out to be words like "perfect" and "excellent".

Overall it was a good assignment to practice pandas and ML stuff.
