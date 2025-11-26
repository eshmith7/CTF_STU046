# CTF Assignment - STU046

Hi, this is my submission for the CTF challenge.

## Steps I took

1.  **Flag 1**:
    I calculated the hash for my ID (STU046) using python. It came out to be `8CB8D714`.
    I searched for this hash in the reviews file and found a review that had it.
    The review was for the book "Sister Sister".
    I checked the ratings and it had 1234 ratings and 5 stars, so it was the right one.
    I took the first 8 letters of the title to make the first flag.

2.  **Flag 2**:
    I already found the review with the hash in the previous step, so I just used that hash for the second flag.

3.  **Flag 3**:
    For the last part, I used the SHAP library.
    I wrote a script to label reviews as suspicious if they were short and had words like "best" or "amazing".
    Then I trained a model and used SHAP to find which words make a review look real.
    The top words were perfect, excellent, and good.
    I combined these with my ID number to get the last flag.

## Running the code

I put everything in `solver.py`. You can run it like this:

```
python3 solver.py
```

It will print the flags and save them to a text file.
