# A/B Testing Dataset
This is a synthetic dataset that simulates a Randomized Controlled Trial (RCT).
In this imaginary scenario, suppose that an online retail store wants to answer
the question _do customers spend more time browsing our site if its background
color is dark?_

> Your job is to estimate the causal effect of switching the site's background
color from white (control) to black (treatment).

## Columns
- User ID: Identifier for each user.
- Group: Contains both the control group (A) and treatment group (B).
- Page Views: Number of pages the user viewed during their session.
- Time Spent: The total amount of time, in seconds, that the user spent on the
site during the session.
- Conversion: Indicates whether a user has completed a desired action (Yes/No).
- Device: Type of device used to access the website.
- Location: The country in UK where the user is based in.

## Source
This dataset was obtained from Kaggle. You can find the [original dataset here](
    https://www.kaggle.com/datasets/adarsh0806/ab-testing-practice
).
