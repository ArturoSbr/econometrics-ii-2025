# A/B Testing Dataset
This is a synthetic dataset that simulates a Randomized Controlled Trial (RCT).
In this imaginary scenario, a UK retailer wants to know if switching their
site's theme from light to dark mode improves user engagement. Namely, the
retailer randomly assigned the themes to users and wants to compare the
conversion rate of both groups.

> Your job is to estimate the causal effect of switching the site's background
color from white (control) to black (treatment) on users' conversion rate.

## Columns
- User ID: Identifier for each user.
- Group: Contains both the control group (A) and treatment group (B).
- Page Views: Number of pages the user viewed during their session.
- Time Spent: The total amount of time, in seconds, that the user spent on the
site during the session.
- Conversion: Indicates whether a user purchased an item during their session.
- Device: Type of device used to access the website.
- Location: The country in UK where the user is based in.

## Source
This dataset was obtained from Kaggle. You can find the [original dataset here](
    https://www.kaggle.com/datasets/adarsh0806/ab-testing-practice
).
