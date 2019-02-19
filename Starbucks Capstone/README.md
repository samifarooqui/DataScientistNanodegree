# StarbucksCapstoneProject
# Motivation:
As part of the final project for Udacitys Data Science Nanodegree, we were given Starbucks data from their rewards app, and tasked to use any methods learned in the nanodegree to analyze the data and make answer any questions we had from it. The questions I had going into this project were to see the key demographics that use the Starbucks Rewards app, and see if clustering can help determine whether a certain demographic would respond more positively to a BOGO (Buy One Get One) or discount type of rewards.
I also wrote a blog post on Medium giving my thoughts on the results. The post can be found here:  https://medium.com/@samifarooqui4/clustering-starbucks-rewards-members-1e5e55ce64a1
# Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

